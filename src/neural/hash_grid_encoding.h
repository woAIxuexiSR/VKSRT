#pragma once

#include "device.h"
#include "resource.h"

#include <memory>
#include <cstdint>

// Multi-resolution hash grid encoding (NRC-style).
// Hash table: float[numLevels * tableSize * featuresPerLevel], packed by level.
// Level l occupies [l * tableSize * featuresPerLevel, (l+1) * tableSize * featuresPerLevel).
class HashGridEncoding
{
public:
    struct Config
    {
        int numLevels{16};
        int featuresPerLevel{2};
        int tableSize{1 << 19};     // per-level entries (524288)
        int coarsestResolution{16};
        int finestResolution{2048};
        int inputDim{2};            // 2 for images, 3 for volumes
    };

    HashGridEncoding(Device &device, const Config &cfg);
    ~HashGridEncoding() = default;

    HashGridEncoding(const HashGridEncoding &) = delete;
    HashGridEncoding &operator=(const HashGridEncoding &) = delete;

    void initTable(unsigned int seed = 1337);

    const Config &getConfig() const { return config; }
    int getEncodedDim() const { return config.numLevels * config.featuresPerLevel; }
    int getTotalFeatures() const { return config.numLevels * config.tableSize * config.featuresPerLevel; }

    VkBuffer getTableBuffer() const { return tableBuffer->getBuffer(); }
    VkBuffer getTableGradBuffer() const { return tableGradBuffer->getBuffer(); }
    uint64_t getTableBufferAddress() const { return tableAddr; }
    uint64_t getTableGradBufferAddress() const { return tableGradAddr; }
    VkDeviceSize getTableBufferSize() const { return tableBufferSize; }

private:
    Device &device;
    Config config;

    std::unique_ptr<StorageBufferResource> tableBuffer;
    std::unique_ptr<StorageBufferResource> tableGradBuffer;
    uint64_t tableAddr{0};
    uint64_t tableGradAddr{0};
    VkDeviceSize tableBufferSize{0};

    PFN_vkGetBufferDeviceAddressKHR vkGetBufferDeviceAddressKHR{nullptr};
    uint64_t getBufferDeviceAddress(VkBuffer buffer);
};
