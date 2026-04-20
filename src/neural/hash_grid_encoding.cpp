#include "hash_grid_encoding.h"

#include <random>
#include <stdexcept>

HashGridEncoding::HashGridEncoding(Device &_d, const Config &cfg)
    : device(_d), config(cfg)
{
    if (cfg.numLevels <= 0 || cfg.numLevels > 32)
        throw std::runtime_error("HashGridEncoding: numLevels must be in (0, 32]");
    if (cfg.featuresPerLevel != 1 && cfg.featuresPerLevel != 2 && cfg.featuresPerLevel != 4)
        throw std::runtime_error("HashGridEncoding: featuresPerLevel must be 1, 2, or 4");
    if (cfg.tableSize <= 0 || (cfg.tableSize & (cfg.tableSize - 1)) != 0)
        throw std::runtime_error("HashGridEncoding: tableSize must be a power of two");
    if (cfg.coarsestResolution <= 0 || cfg.finestResolution < cfg.coarsestResolution)
        throw std::runtime_error("HashGridEncoding: invalid resolution range");
    if (cfg.inputDim < 1 || cfg.inputDim > 4)
        throw std::runtime_error("HashGridEncoding: inputDim must be in [1, 4]");

    vkGetBufferDeviceAddressKHR = device.loadDeviceFunction<PFN_vkGetBufferDeviceAddressKHR>(
        "vkGetBufferDeviceAddressKHR");

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

    tableAddr = getBufferDeviceAddress(tableBuffer->getBuffer());
    tableGradAddr = getBufferDeviceAddress(tableGradBuffer->getBuffer());
}

uint64_t HashGridEncoding::getBufferDeviceAddress(VkBuffer buffer)
{
    VkBufferDeviceAddressInfoKHR info{};
    info.sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO_KHR;
    info.buffer = buffer;
    return vkGetBufferDeviceAddressKHR(device.getDevice(), &info);
}

void HashGridEncoding::initTable(unsigned int seed)
{
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(-1e-4f, 1e-4f);

    std::vector<float> data(getTotalFeatures());
    for (auto &v : data)
        v = dist(rng);

    tableBuffer->update(data.data());
}
