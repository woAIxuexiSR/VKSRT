#include "mlp.h"

#include <random>
#include <cmath>
#include <stdexcept>

static size_t align64(size_t size)
{
    return (size + 63) & ~63;
}

MLP::MLP(Device &_d, const std::vector<LayerConfig> &_layers)
    : device(_d), layers(_layers)
{
    vkGetBufferDeviceAddressKHR = device.loadDeviceFunction<PFN_vkGetBufferDeviceAddressKHR>(
        "vkGetBufferDeviceAddressKHR");

    if (layers.size() < 2)
        throw std::runtime_error("MLP requires at least 2 layers (input + output)");
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
    buildLayerAddressBuffer();
}

uint64_t MLP::getBufferDeviceAddress(VkBuffer buffer)
{
    VkBufferDeviceAddressInfoKHR info{};
    info.sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO_KHR;
    info.buffer = buffer;
    return vkGetBufferDeviceAddressKHR(device.getDevice(), &info);
}

void MLP::allocateStorage()
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

    paramBufferAddr = getBufferDeviceAddress(paramBuffer->getBuffer());
}

void MLP::buildLayerAddressBuffer()
{
    VkDeviceSize bufSize = layers.size() * 4 * sizeof(uint64_t);
    layerAddressBuffer = std::make_unique<StorageBufferResource>(
        device, bufSize,
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
            VK_BUFFER_USAGE_TRANSFER_DST_BIT |
            VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT);

    std::vector<uint64_t> addresses(layers.size() * 4);
    for (size_t i = 0; i < layers.size(); i++)
    {
        addresses[i * 4 + 0] = paramBufferAddr + allocations[i].weightsOffset;
        addresses[i * 4 + 1] = paramBufferAddr + allocations[i].weightsGradOffset;
        addresses[i * 4 + 2] = paramBufferAddr + allocations[i].biasOffset;
        addresses[i * 4 + 3] = paramBufferAddr + allocations[i].biasGradOffset;
    }

    layerAddressBuffer->update(addresses.data());
}

void MLP::initWeights(unsigned int seed)
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
}
