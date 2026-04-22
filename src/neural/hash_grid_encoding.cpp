#include "hash_grid_encoding.h"

#include <random>
#include <cmath>
#include <stdexcept>

REGISTER_ENCODING_CPP(HashGridEncoding, "hashgrid");

struct HashGridForwardPushConstants
{
    uint64_t inputBuffer;
    uint64_t outputBuffer;
    uint64_t hashTable;
    uint32_t sampleCount;
    uint32_t inputDim;
    uint32_t numLevels;
    uint32_t featuresPerLevel;
    uint32_t tableSize;
    float    coarsestResolution;
    float    perLevelScale;
    uint32_t inputFieldOffset;
    uint32_t inputStride;
    uint32_t outputFieldOffset;
    uint32_t outputStride;
};

struct HashGridBackwardPushConstants
{
    uint64_t dEncodedBuffer;
    uint64_t inputBuffer;
    uint64_t hashTableGrad;
    uint32_t sampleCount;
    uint32_t inputDim;
    uint32_t numLevels;
    uint32_t featuresPerLevel;
    uint32_t tableSize;
    float    coarsestResolution;
    float    perLevelScale;
    uint32_t dEncodedFieldOffset;
    uint32_t dEncodedStride;
    uint32_t inputFieldOffset;
    uint32_t inputStride;
};

struct AdamPushConstants
{
    uint64_t adamStates;
    uint64_t params;
    uint64_t gradients;
    uint32_t count;
};

static constexpr size_t kAdamStateSize = sizeof(float) * 2 + sizeof(int32_t);

HashGridEncoding::HashGridEncoding(Device &_d, const json &params)
    : device(_d)
{
    if (params.contains("numLevels"))          numLevels = params["numLevels"].get<int>();
    if (params.contains("featuresPerLevel"))   featuresPerLevel = params["featuresPerLevel"].get<int>();
    if (params.contains("tableSize"))          tableSize = params["tableSize"].get<int>();
    if (params.contains("coarsestResolution")) coarsestResolution = params["coarsestResolution"].get<int>();
    if (params.contains("finestResolution"))   finestResolution = params["finestResolution"].get<int>();
    if (params.contains("inputDim"))           inputDim = params["inputDim"].get<int>();

    if (numLevels <= 0 || numLevels > 32)
        throw std::runtime_error("HashGridEncoding: numLevels must be in (0, 32]");
    if (featuresPerLevel != 1 && featuresPerLevel != 2 && featuresPerLevel != 4)
        throw std::runtime_error("HashGridEncoding: featuresPerLevel must be 1, 2, or 4");
    if (tableSize <= 0 || (tableSize & (tableSize - 1)) != 0)
        throw std::runtime_error("HashGridEncoding: tableSize must be a power of two");
    if (coarsestResolution <= 0 || finestResolution < coarsestResolution)
        throw std::runtime_error("HashGridEncoding: invalid resolution range");
    if (inputDim < 1 || inputDim > 4)
        throw std::runtime_error("HashGridEncoding: inputDim must be in [1, 4]");

    perLevelScale = (numLevels > 1)
        ? std::exp(std::log((float)finestResolution / (float)coarsestResolution)
                   / (float)(numLevels - 1))
        : 1.0f;

    tableBufferSize = (VkDeviceSize)getTotalFeatures() * sizeof(float);

    tableBuffer = std::make_unique<StorageBufferResource>(
        device, tableBufferSize,
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
            VK_BUFFER_USAGE_TRANSFER_DST_BIT |
            VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
            VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT);

    tableGradBuffer = std::make_unique<StorageBufferResource>(
        device, tableBufferSize,
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
            VK_BUFFER_USAGE_TRANSFER_DST_BIT |
            VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT);

    tableAddr = device.getBufferDeviceAddress(tableBuffer->getBuffer());
    tableGradAddr = device.getBufferDeviceAddress(tableGradBuffer->getBuffer());

    adamState = std::make_unique<StorageBufferResource>(
        device, (VkDeviceSize)getTotalFeatures() * kAdamStateSize,
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
            VK_BUFFER_USAGE_TRANSFER_DST_BIT |
            VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT);
}

void HashGridEncoding::initParams(unsigned int seed)
{
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(-1e-4f, 1e-4f);

    std::vector<float> data(getTotalFeatures());
    for (auto &v : data)
        v = dist(rng);

    tableBuffer->update(data.data());
}

void HashGridEncoding::resetAdamState()
{
    VkCommandBuffer cmd = device.beginSingleTimeCommands();
    vkCmdFillBuffer(cmd, adamState->getBuffer(), 0,
                    (VkDeviceSize)getTotalFeatures() * kAdamStateSize, 0);
    device.endSingleTimeCommands(cmd);
}

void HashGridEncoding::createPipelines()
{
    std::vector<DescriptorLayoutBinding> emptyBindings{};

    forwardPipeline = std::make_unique<ComputePipeline>(
        device, 1, emptyBindings,
        shaderPath("neural/hashgrid_forward.spv"),
        sizeof(HashGridForwardPushConstants));

    backwardPipeline = std::make_unique<ComputePipeline>(
        device, 1, emptyBindings,
        shaderPath("neural/hashgrid_backward.spv"),
        sizeof(HashGridBackwardPushConstants));

    adamPipeline = std::make_unique<ComputePipeline>(
        device, 1, emptyBindings,
        shaderPath("neural/adam_kernel.spv"),
        sizeof(AdamPushConstants));
}

void HashGridEncoding::recordForward(VkCommandBuffer cmd,
                                     VkBuffer rawInput, uint32_t inputOffset, uint32_t inputStride,
                                     VkBuffer encodedOutput, uint32_t outputOffset, uint32_t outputStride,
                                     uint32_t sampleCount)
{
    HashGridForwardPushConstants pc{};
    pc.inputBuffer = device.getBufferDeviceAddress(rawInput);
    pc.outputBuffer = device.getBufferDeviceAddress(encodedOutput);
    pc.hashTable = tableAddr;
    pc.sampleCount = sampleCount;
    pc.inputDim = (uint32_t)inputDim;
    pc.numLevels = (uint32_t)numLevels;
    pc.featuresPerLevel = (uint32_t)featuresPerLevel;
    pc.tableSize = (uint32_t)tableSize;
    pc.coarsestResolution = (float)coarsestResolution;
    pc.perLevelScale = perLevelScale;
    pc.inputFieldOffset = inputOffset;
    pc.inputStride = inputStride;
    pc.outputFieldOffset = outputOffset;
    pc.outputStride = outputStride;

    forwardPipeline->bindPipeline(cmd);
    forwardPipeline->pushConstants(cmd, &pc);
    vkCmdDispatch(cmd, (sampleCount + 255) / 256, 1, 1);
}

void HashGridEncoding::recordForwardWithParams(VkCommandBuffer cmd,
                                               uint64_t paramAddr,
                                               VkBuffer rawInput, uint32_t inputOffset, uint32_t inputStride,
                                               VkBuffer encodedOutput, uint32_t outputOffset, uint32_t outputStride,
                                               uint32_t sampleCount)
{
    HashGridForwardPushConstants pc{};
    pc.inputBuffer = device.getBufferDeviceAddress(rawInput);
    pc.outputBuffer = device.getBufferDeviceAddress(encodedOutput);
    pc.hashTable = paramAddr;
    pc.sampleCount = sampleCount;
    pc.inputDim = (uint32_t)inputDim;
    pc.numLevels = (uint32_t)numLevels;
    pc.featuresPerLevel = (uint32_t)featuresPerLevel;
    pc.tableSize = (uint32_t)tableSize;
    pc.coarsestResolution = (float)coarsestResolution;
    pc.perLevelScale = perLevelScale;
    pc.inputFieldOffset = inputOffset;
    pc.inputStride = inputStride;
    pc.outputFieldOffset = outputOffset;
    pc.outputStride = outputStride;

    forwardPipeline->bindPipeline(cmd);
    forwardPipeline->pushConstants(cmd, &pc);
    vkCmdDispatch(cmd, (sampleCount + 255) / 256, 1, 1);
}

void HashGridEncoding::recordBackward(VkCommandBuffer cmd,
                                      VkBuffer dEncoded, uint32_t dOffset, uint32_t dStride,
                                      VkBuffer rawInput, uint32_t inputOffset, uint32_t inputStride,
                                      uint32_t sampleCount)
{
    HashGridBackwardPushConstants pc{};
    pc.dEncodedBuffer = device.getBufferDeviceAddress(dEncoded);
    pc.inputBuffer = device.getBufferDeviceAddress(rawInput);
    pc.hashTableGrad = tableGradAddr;
    pc.sampleCount = sampleCount;
    pc.inputDim = (uint32_t)inputDim;
    pc.numLevels = (uint32_t)numLevels;
    pc.featuresPerLevel = (uint32_t)featuresPerLevel;
    pc.tableSize = (uint32_t)tableSize;
    pc.coarsestResolution = (float)coarsestResolution;
    pc.perLevelScale = perLevelScale;
    pc.dEncodedFieldOffset = dOffset;
    pc.dEncodedStride = dStride;
    pc.inputFieldOffset = inputOffset;
    pc.inputStride = inputStride;

    backwardPipeline->bindPipeline(cmd);
    backwardPipeline->pushConstants(cmd, &pc);
    vkCmdDispatch(cmd, (sampleCount + 255) / 256, 1, 1);
}

void HashGridEncoding::recordZeroGrads(VkCommandBuffer cmd)
{
    vkCmdFillBuffer(cmd, tableGradBuffer->getBuffer(), 0, tableBufferSize, 0);
}

void HashGridEncoding::recordAdam(VkCommandBuffer cmd)
{
    AdamPushConstants pc{};
    pc.adamStates = device.getBufferDeviceAddress(adamState->getBuffer());
    pc.params = tableAddr;
    pc.gradients = tableGradAddr;
    pc.count = (uint32_t)getTotalFeatures();

    adamPipeline->bindPipeline(cmd);
    adamPipeline->pushConstants(cmd, &pc);
    vkCmdDispatch(cmd, (pc.count + 255) / 256, 1, 1);
}
