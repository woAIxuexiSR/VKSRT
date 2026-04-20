#pragma once

#include "device.h"
#include "resource.h"
#include "pipeline.h"

#include <memory>
#include <cstdint>

// Multi-resolution hash grid encoding (NRC-style).
// Owns hash table + gradient buffers, forward/backward compute pipelines,
// and Adam optimizer state.
class HashGridEncoding
{
public:
    struct Config
    {
        int numLevels{16};
        int featuresPerLevel{2};
        int tableSize{1 << 19};
        int coarsestResolution{16};
        int finestResolution{2048};
        int inputDim{2};
    };

    HashGridEncoding(Device &device, const Config &cfg);
    ~HashGridEncoding() = default;

    HashGridEncoding(const HashGridEncoding &) = delete;
    HashGridEncoding &operator=(const HashGridEncoding &) = delete;

    void initTable(unsigned int seed = 1337);
    void createPipelines();

    // Record forward: rawInput → encodedOutput.
    void recordForward(VkCommandBuffer cmd, VkBuffer rawInput, VkBuffer encodedOutput,
                       uint32_t sampleCount);

    // Record backward: dEncoded + rawInput → splat gradients to tableGrad.
    void recordBackward(VkCommandBuffer cmd, VkBuffer dEncoded, VkBuffer rawInput,
                        uint32_t sampleCount);

    // Record zero-fill of table gradient buffer.
    void recordZeroGrads(VkCommandBuffer cmd);

    // Record Adam optimizer step on hash table parameters.
    void recordAdam(VkCommandBuffer cmd);

    // Reset Adam state (call once after initTable).
    void resetAdamState();

    const Config &getConfig() const { return config; }
    int getEncodedDim() const { return config.numLevels * config.featuresPerLevel; }
    int getTotalFeatures() const { return config.numLevels * config.tableSize * config.featuresPerLevel; }
    float getPerLevelScale() const { return perLevelScale; }

    VkBuffer getTableBuffer() const { return tableBuffer->getBuffer(); }
    VkBuffer getTableGradBuffer() const { return tableGradBuffer->getBuffer(); }
    uint64_t getTableBufferAddress() const { return tableAddr; }
    uint64_t getTableGradBufferAddress() const { return tableGradAddr; }
    VkDeviceSize getTableBufferSize() const { return tableBufferSize; }

private:
    Device &device;
    Config config;
    float perLevelScale{1.0f};

    std::unique_ptr<StorageBufferResource> tableBuffer;
    std::unique_ptr<StorageBufferResource> tableGradBuffer;
    uint64_t tableAddr{0};
    uint64_t tableGradAddr{0};
    VkDeviceSize tableBufferSize{0};

    // Adam optimizer state
    std::unique_ptr<StorageBufferResource> adamState;

    // Compute pipelines
    std::unique_ptr<ComputePipeline> forwardPipeline;
    std::unique_ptr<ComputePipeline> backwardPipeline;
    std::unique_ptr<ComputePipeline> adamPipeline;

    PFN_vkGetBufferDeviceAddressKHR vkGetBufferDeviceAddressKHR{nullptr};
    uint64_t getBufferDeviceAddress(VkBuffer buffer);
};
