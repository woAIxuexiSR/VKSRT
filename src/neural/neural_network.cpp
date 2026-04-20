#include "neural_network.h"

#include <cstring>
#include <cmath>

struct TrainMlpPushConstants
{
    uint64_t layerAddressBuffer;
    uint64_t inputBuffer;
    uint64_t gtBuffer;
    uint64_t lossBuffer;
    uint32_t sampleCount;
    uint32_t hiddenLayerCount;
    uint32_t inputSize;
    uint32_t outputSize;
};

struct TrainHashMlpPushConstants
{
    uint64_t layerAddressBuffer;
    uint64_t inputBuffer;
    uint64_t gtBuffer;
    uint64_t lossBuffer;
    uint64_t hashTable;
    uint64_t hashTableGrad;
    uint32_t sampleCount;
    uint32_t hiddenLayerCount;
    uint32_t inputSize;
    uint32_t outputSize;
    uint32_t rawInputDim;
    uint32_t numLevels;
    uint32_t featuresPerLevel;
    uint32_t tableSize;
    float    coarsestResolution;
    float    perLevelScale;
};

struct AdamPushConstants
{
    uint64_t adamStates;
    uint64_t params;
    uint64_t gradients;
    uint32_t count;
};

static constexpr size_t kAdamStateSize = sizeof(float) * 2 + sizeof(int32_t);

NeuralNetwork::NeuralNetwork(Device &_d, const std::vector<LayerConfig> &_layers)
    : NeuralNetwork(_d, Config{_layers, false, {}})
{
}

NeuralNetwork::NeuralNetwork(Device &_d, const Config &cfg)
    : device(_d)
{
    vkGetBufferDeviceAddressKHR = device.loadDeviceFunction<PFN_vkGetBufferDeviceAddressKHR>(
        "vkGetBufferDeviceAddressKHR");

    mlp = std::make_unique<MLP>(device, cfg.layers);

    if (cfg.useEncoding)
    {
        hashGrid = std::make_unique<HashGridEncoding>(device, cfg.encoding);
        if (mlp->getInputSize() != hashGrid->getEncodedDim())
            throw std::runtime_error("NeuralNetwork: MLP inputSize must equal HashGrid encodedDim");
    }

    allocateTrainingBuffers();
    createPipelines();
}

NeuralNetwork::~NeuralNetwork()
{
    if (lossReadbackMapped)
        vkUnmapMemory(device.getDevice(), lossReadbackMemory);
    if (lossReadbackBuffer != VK_NULL_HANDLE)
        vkDestroyBuffer(device.getDevice(), lossReadbackBuffer, nullptr);
    if (lossReadbackMemory != VK_NULL_HANDLE)
        vkFreeMemory(device.getDevice(), lossReadbackMemory, nullptr);
}

uint64_t NeuralNetwork::getBufferDeviceAddress(VkBuffer buffer)
{
    VkBufferDeviceAddressInfoKHR info{};
    info.sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO_KHR;
    info.buffer = buffer;
    return vkGetBufferDeviceAddressKHR(device.getDevice(), &info);
}

void NeuralNetwork::allocateTrainingBuffers()
{
    adamStateMlp = std::make_unique<StorageBufferResource>(
        device, (VkDeviceSize)mlp->getTotalParams() * kAdamStateSize,
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
            VK_BUFFER_USAGE_TRANSFER_DST_BIT |
            VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT);

    if (hashGrid)
    {
        adamStateHash = std::make_unique<StorageBufferResource>(
            device, (VkDeviceSize)hashGrid->getTotalFeatures() * kAdamStateSize,
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                VK_BUFFER_USAGE_TRANSFER_DST_BIT |
                VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT);
    }

    lossGpuBuffer = std::make_unique<StorageBufferResource>(
        device, sizeof(float),
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
            VK_BUFFER_USAGE_TRANSFER_DST_BIT |
            VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
            VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT);

    device.createBuffer(sizeof(float),
                        VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                        lossReadbackBuffer, lossReadbackMemory);
    vkMapMemory(device.getDevice(), lossReadbackMemory, 0, sizeof(float), 0, &lossReadbackMapped);
    memset(lossReadbackMapped, 0, sizeof(float));
}

void NeuralNetwork::createPipelines()
{
    std::vector<DescriptorLayoutBinding> emptyBindings{};

    if (hashGrid)
    {
        trainPipeline = std::make_unique<ComputePipeline>(
            device, 1, emptyBindings,
            "build/shaders/neural/train_hashmlp.slang.spv",
            sizeof(TrainHashMlpPushConstants));
    }
    else
    {
        trainPipeline = std::make_unique<ComputePipeline>(
            device, 1, emptyBindings,
            "build/shaders/neural/train_mlp.slang.spv",
            sizeof(TrainMlpPushConstants));
    }

    adamPipeline = std::make_unique<ComputePipeline>(
        device, 1, emptyBindings,
        "build/shaders/neural/adam_kernel.slang.spv",
        sizeof(AdamPushConstants));
}

void NeuralNetwork::initWeights(unsigned int seed)
{
    mlp->initWeights(seed);
    if (hashGrid)
        hashGrid->initTable(seed ^ 0x9E3779B9u);

    VkCommandBuffer cmd = device.beginSingleTimeCommands();
    vkCmdFillBuffer(cmd, adamStateMlp->getBuffer(), 0,
                    (VkDeviceSize)mlp->getTotalParams() * kAdamStateSize, 0);
    if (adamStateHash)
    {
        vkCmdFillBuffer(cmd, adamStateHash->getBuffer(), 0,
                        (VkDeviceSize)hashGrid->getTotalFeatures() * kAdamStateSize, 0);
    }
    device.endSingleTimeCommands(cmd);
}

void NeuralNetwork::computeBarrier(VkCommandBuffer cmd)
{
    device.memoryBarrier(cmd,
                         VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT | VK_ACCESS_2_SHADER_WRITE_BIT,
                         VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_SHADER_READ_BIT |
                             VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                         VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                         VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT);
}

float NeuralNetwork::readLoss() const
{
    float totalLoss = *static_cast<const float *>(lossReadbackMapped);
    return lastSampleCount > 0 ? totalLoss / (float)lastSampleCount : 0.0f;
}

void NeuralNetwork::recordTrain(VkCommandBuffer cmd, VkBuffer inputBuffer, VkBuffer gtBuffer, uint32_t sampleCount)
{
    lastSampleCount = sampleCount;
    uint64_t paramAddr = mlp->getParamBufferAddress();
    VkDeviceSize gradientOffset = mlp->getGradientOffset();
    VkBuffer paramBuf = mlp->getParamBuffer();
    VkDeviceSize totalBufferSize = mlp->getParamBufferSize();

    // Zero gradients (MLP + hash table if present) + loss
    VkDeviceSize gradSize = totalBufferSize - gradientOffset;
    vkCmdFillBuffer(cmd, paramBuf, gradientOffset, gradSize, 0);
    vkCmdFillBuffer(cmd, lossGpuBuffer->getBuffer(), 0, sizeof(float), 0);
    if (hashGrid)
    {
        vkCmdFillBuffer(cmd, hashGrid->getTableGradBuffer(), 0,
                        hashGrid->getTableBufferSize(), 0);
    }

    device.memoryBarrier(cmd,
                         VK_ACCESS_2_TRANSFER_WRITE_BIT,
                         VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                         VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                         VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT);

    // Train dispatch
    if (hashGrid)
    {
        const auto &hc = hashGrid->getConfig();
        float perLevelScale = (hc.numLevels > 1)
            ? std::exp(std::log((float)hc.finestResolution / (float)hc.coarsestResolution)
                       / (float)(hc.numLevels - 1))
            : 1.0f;

        TrainHashMlpPushConstants pc{};
        pc.layerAddressBuffer = getBufferDeviceAddress(mlp->getLayerAddressBuffer());
        pc.inputBuffer = getBufferDeviceAddress(inputBuffer);
        pc.gtBuffer = getBufferDeviceAddress(gtBuffer);
        pc.lossBuffer = getBufferDeviceAddress(lossGpuBuffer->getBuffer());
        pc.hashTable = hashGrid->getTableBufferAddress();
        pc.hashTableGrad = hashGrid->getTableGradBufferAddress();
        pc.sampleCount = sampleCount;
        pc.hiddenLayerCount = (uint32_t)mlp->getHiddenLayerCount();
        pc.inputSize = (uint32_t)mlp->getInputSize();
        pc.outputSize = (uint32_t)mlp->getOutputSize();
        pc.rawInputDim = (uint32_t)hc.inputDim;
        pc.numLevels = (uint32_t)hc.numLevels;
        pc.featuresPerLevel = (uint32_t)hc.featuresPerLevel;
        pc.tableSize = (uint32_t)hc.tableSize;
        pc.coarsestResolution = (float)hc.coarsestResolution;
        pc.perLevelScale = perLevelScale;

        trainPipeline->bindPipeline(cmd);
        trainPipeline->pushConstants(cmd, &pc);
        vkCmdDispatch(cmd, (sampleCount + 255) / 256, 1, 1);
    }
    else
    {
        TrainMlpPushConstants pc{};
        pc.layerAddressBuffer = getBufferDeviceAddress(mlp->getLayerAddressBuffer());
        pc.inputBuffer = getBufferDeviceAddress(inputBuffer);
        pc.gtBuffer = getBufferDeviceAddress(gtBuffer);
        pc.lossBuffer = getBufferDeviceAddress(lossGpuBuffer->getBuffer());
        pc.sampleCount = sampleCount;
        pc.hiddenLayerCount = (uint32_t)mlp->getHiddenLayerCount();
        pc.inputSize = (uint32_t)mlp->getInputSize();
        pc.outputSize = (uint32_t)mlp->getOutputSize();

        trainPipeline->bindPipeline(cmd);
        trainPipeline->pushConstants(cmd, &pc);
        vkCmdDispatch(cmd, (sampleCount + 255) / 256, 1, 1);
    }

    computeBarrier(cmd);

    // Adam update (MLP params)
    {
        AdamPushConstants pc{};
        pc.adamStates = getBufferDeviceAddress(adamStateMlp->getBuffer());
        pc.params = paramAddr;
        pc.gradients = paramAddr + gradientOffset;
        pc.count = (uint32_t)mlp->getTotalParams();

        adamPipeline->bindPipeline(cmd);
        adamPipeline->pushConstants(cmd, &pc);
        vkCmdDispatch(cmd, (pc.count + 255) / 256, 1, 1);
    }

    // Adam update (hash table)
    if (hashGrid)
    {
        AdamPushConstants pc{};
        pc.adamStates = getBufferDeviceAddress(adamStateHash->getBuffer());
        pc.params = hashGrid->getTableBufferAddress();
        pc.gradients = hashGrid->getTableGradBufferAddress();
        pc.count = (uint32_t)hashGrid->getTotalFeatures();

        adamPipeline->bindPipeline(cmd);
        adamPipeline->pushConstants(cmd, &pc);
        vkCmdDispatch(cmd, (pc.count + 255) / 256, 1, 1);
    }

    // Copy loss to readback buffer
    device.memoryBarrier(cmd,
                         VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                         VK_ACCESS_2_TRANSFER_READ_BIT,
                         VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                         VK_PIPELINE_STAGE_2_TRANSFER_BIT);

    VkBufferCopy copyRegion{};
    copyRegion.size = sizeof(float);
    vkCmdCopyBuffer(cmd, lossGpuBuffer->getBuffer(), lossReadbackBuffer, 1, &copyRegion);

    device.memoryBarrier(cmd,
                         VK_ACCESS_2_TRANSFER_WRITE_BIT,
                         VK_ACCESS_2_HOST_READ_BIT,
                         VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                         VK_PIPELINE_STAGE_2_HOST_BIT);
}
