#include "neural_network.h"

#include <random>
#include <cmath>
#include <cstring>
#include <stdexcept>

struct TrainPushConstants
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

struct AdamPushConstants
{
    uint64_t adamStates;
    uint64_t params;
    uint64_t gradients;
    uint32_t count;
};

static size_t align64(size_t size)
{
    return (size + 63) & ~63;
}

NeuralNetwork::NeuralNetwork(Device &_d, const std::vector<LayerConfig> &_layers)
    : device(_d), layers(_layers)
{
    vkGetBufferDeviceAddressKHR = device.loadDeviceFunction<PFN_vkGetBufferDeviceAddressKHR>(
        "vkGetBufferDeviceAddressKHR");

    if (layers.size() < 2)
        throw std::runtime_error("NeuralNetwork requires at least 2 layers (input + output)");
    if (layers.front().inputSize <= 0 || layers.front().inputSize > 64)
        throw std::runtime_error("Input layer inputSize must be in (0, 64]");
    if (layers.back().outputSize <= 0 || layers.back().outputSize > 64)
        throw std::runtime_error("Output layer outputSize must be in (0, 64]");
    if (layers.front().outputSize != 64)
        throw std::runtime_error("Input layer must output 64 (hidden width)");
    if (layers.back().inputSize != 64)
        throw std::runtime_error("Output layer must input 64 (hidden width)");
    for (size_t i = 1; i + 1 < layers.size(); i++)
    {
        if (layers[i].inputSize != 64 || layers[i].outputSize != 64)
            throw std::runtime_error("Hidden layers must be 64->64");
    }

    hiddenLayerCount = (int)layers.size() - 2;

    allocateStorage();
    buildConstantBuffer();
    buildLayerAddressBuffer();
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

void NeuralNetwork::allocateStorage()
{
    totalBufferSize = 0;
    totalParamCount = 0;
    allocations.resize(layers.size());

    auto alloc = [&](size_t size) -> size_t
    {
        size_t aligned = align64(size);
        size_t offset = totalBufferSize;
        totalBufferSize += aligned;
        return offset;
    };

    for (size_t i = 0; i < layers.size(); i++)
    {
        int weightCount = layers[i].inputSize * layers[i].outputSize;
        int biasCount = layers[i].outputSize;

        allocations[i].weightsSize = weightCount * sizeof(float);
        allocations[i].weightsOffset = alloc(allocations[i].weightsSize);
        allocations[i].biasSize = biasCount * sizeof(float);
        allocations[i].biasOffset = alloc(allocations[i].biasSize);

        totalParamCount += weightCount + biasCount;
    }

    gradientOffset = totalBufferSize;
    for (size_t i = 0; i < layers.size(); i++)
    {
        allocations[i].weightsGradOffset = alloc(allocations[i].weightsSize);
        allocations[i].biasGradOffset = alloc(allocations[i].biasSize);
    }

    paramBuffer = std::make_unique<StorageBufferResource>(
        device, totalBufferSize,
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
            VK_BUFFER_USAGE_TRANSFER_DST_BIT |
            VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
            VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT);

    static constexpr size_t kAdamStateSize = sizeof(float) * 2 + sizeof(int32_t);
    adamStateBuffer = std::make_unique<StorageBufferResource>(
        device, (VkDeviceSize)totalParamCount * kAdamStateSize,
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
            VK_BUFFER_USAGE_TRANSFER_DST_BIT |
            VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT);

    // Loss GPU buffer (atomic float)
    lossGpuBuffer = std::make_unique<StorageBufferResource>(
        device, sizeof(float),
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
            VK_BUFFER_USAGE_TRANSFER_DST_BIT |
            VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
            VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT);

    // Loss readback buffer (host-visible, persistently mapped)
    device.createBuffer(sizeof(float),
                        VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                        lossReadbackBuffer, lossReadbackMemory);
    vkMapMemory(device.getDevice(), lossReadbackMemory, 0, sizeof(float), 0, &lossReadbackMapped);
    memset(lossReadbackMapped, 0, sizeof(float));
}

void NeuralNetwork::buildConstantBuffer()
{
    networkConstantBufferSize = layers.size() * 4 * sizeof(uint64_t);
    networkConstantBuffer = std::make_unique<UniformBufferResource>(
        device, networkConstantBufferSize, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT);

    std::vector<uint64_t> addresses(layers.size() * 4);
    uint64_t baseAddr = getBufferDeviceAddress(paramBuffer->getBuffer());

    for (size_t i = 0; i < layers.size(); i++)
    {
        addresses[i * 4 + 0] = baseAddr + allocations[i].weightsOffset;
        addresses[i * 4 + 1] = baseAddr + allocations[i].weightsGradOffset;
        addresses[i * 4 + 2] = baseAddr + allocations[i].biasOffset;
        addresses[i * 4 + 3] = baseAddr + allocations[i].biasGradOffset;
    }

    networkConstantBuffer->update(addresses.data());
}

void NeuralNetwork::buildLayerAddressBuffer()
{
    VkDeviceSize bufSize = layers.size() * 4 * sizeof(uint64_t);
    layerAddressBuffer = std::make_unique<StorageBufferResource>(
        device, bufSize,
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
            VK_BUFFER_USAGE_TRANSFER_DST_BIT |
            VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT);

    std::vector<uint64_t> addresses(layers.size() * 4);
    uint64_t baseAddr = getBufferDeviceAddress(paramBuffer->getBuffer());

    for (size_t i = 0; i < layers.size(); i++)
    {
        addresses[i * 4 + 0] = baseAddr + allocations[i].weightsOffset;
        addresses[i * 4 + 1] = baseAddr + allocations[i].weightsGradOffset;
        addresses[i * 4 + 2] = baseAddr + allocations[i].biasOffset;
        addresses[i * 4 + 3] = baseAddr + allocations[i].biasGradOffset;
    }

    layerAddressBuffer->update(addresses.data());
}

void NeuralNetwork::createPipelines()
{
    std::vector<DescriptorLayoutBinding> emptyBindings{};

    trainPipeline = std::make_unique<ComputePipeline>(
        device, 1, emptyBindings,
        "build/shaders/neural/train_kernel.slang.spv",
        sizeof(TrainPushConstants));

    adamPipeline = std::make_unique<ComputePipeline>(
        device, 1, emptyBindings,
        "build/shaders/neural/adam_kernel.slang.spv",
        sizeof(AdamPushConstants));
}

void NeuralNetwork::initWeights(unsigned int seed)
{
    std::mt19937 rng(seed);
    std::vector<float> data(totalBufferSize / sizeof(float), 0.0f);

    for (size_t i = 0; i < layers.size(); i++)
    {
        int fanIn = layers[i].inputSize;
        int fanOut = layers[i].outputSize;
        float stddev = std::sqrt(2.0f / (float)(fanIn + fanOut));
        std::normal_distribution<float> dist(0.0f, stddev);

        int weightCount = fanIn * fanOut;
        float *wPtr = data.data() + allocations[i].weightsOffset / sizeof(float);
        for (int j = 0; j < weightCount; j++)
            wPtr[j] = dist(rng);
    }

    paramBuffer->update(data.data());

    VkCommandBuffer cmd = device.beginSingleTimeCommands();
    static constexpr size_t kAdamStateSize = sizeof(float) * 2 + sizeof(int32_t);
    vkCmdFillBuffer(cmd, adamStateBuffer->getBuffer(), 0,
                    (VkDeviceSize)totalParamCount * kAdamStateSize, 0);
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
    uint64_t paramAddr = getBufferDeviceAddress(paramBuffer->getBuffer());

    // Zero gradients + loss
    VkDeviceSize gradSize = totalBufferSize - gradientOffset;
    vkCmdFillBuffer(cmd, paramBuffer->getBuffer(), gradientOffset, gradSize, 0);
    vkCmdFillBuffer(cmd, lossGpuBuffer->getBuffer(), 0, sizeof(float), 0);

    device.memoryBarrier(cmd,
                         VK_ACCESS_2_TRANSFER_WRITE_BIT,
                         VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                         VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                         VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT);

    // Train dispatch
    {
        TrainPushConstants pc{};
        pc.layerAddressBuffer = getBufferDeviceAddress(layerAddressBuffer->getBuffer());
        pc.inputBuffer = getBufferDeviceAddress(inputBuffer);
        pc.gtBuffer = getBufferDeviceAddress(gtBuffer);
        pc.lossBuffer = getBufferDeviceAddress(lossGpuBuffer->getBuffer());
        pc.sampleCount = sampleCount;
        pc.hiddenLayerCount = (uint32_t)hiddenLayerCount;
        pc.inputSize = (uint32_t)layers.front().inputSize;
        pc.outputSize = (uint32_t)layers.back().outputSize;

        trainPipeline->bindPipeline(cmd);
        trainPipeline->pushConstants(cmd, &pc);
        vkCmdDispatch(cmd, (sampleCount + 255) / 256, 1, 1);
    }

    computeBarrier(cmd);

    // Adam update
    {
        AdamPushConstants pc{};
        pc.adamStates = getBufferDeviceAddress(adamStateBuffer->getBuffer());
        pc.params = paramAddr;
        pc.gradients = paramAddr + gradientOffset;
        pc.count = (uint32_t)totalParamCount;

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
