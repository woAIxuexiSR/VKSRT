#pragma once

#include "encoding.h"

#include <memory>
#include <cstdint>

class HashGridEncoding : public Encoding
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

    int getInputDim() const override { return config.inputDim; }
    int getOutputDim() const override { return config.numLevels * config.featuresPerLevel; }
    bool hasTrainableParams() const override { return true; }
    int getTrainableParamCount() const override { return getTotalFeatures(); }
    std::string typeName() const override { return "hashgrid"; }

    void createPipelines() override;
    void initParams(unsigned int seed) override;
    void resetAdamState() override;

    void recordForward(VkCommandBuffer cmd,
                       VkBuffer rawInput, uint32_t inputOffset, uint32_t inputStride,
                       VkBuffer encodedOutput, uint32_t outputOffset, uint32_t outputStride,
                       uint32_t sampleCount) override;

    void recordBackward(VkCommandBuffer cmd,
                        VkBuffer dEncoded, uint32_t dOffset, uint32_t dStride,
                        VkBuffer rawInput, uint32_t inputOffset, uint32_t inputStride,
                        uint32_t sampleCount) override;

    void recordZeroGrads(VkCommandBuffer cmd) override;
    void recordAdam(VkCommandBuffer cmd) override;

    const Config &getConfig() const { return config; }
    int getEncodedDim() const { return config.numLevels * config.featuresPerLevel; }
    int getTotalFeatures() const { return config.numLevels * config.tableSize * config.featuresPerLevel; }
    float getPerLevelScale() const { return perLevelScale; }

    VkBuffer getTableBuffer() const { return tableBuffer->getBuffer(); }
    VkBuffer getTableGradBuffer() const { return tableGradBuffer->getBuffer(); }

    uint64_t getParamBufferAddress() const override { return tableAddr; }
    VkDeviceSize getParamBufferSize() const override { return tableBufferSize; }

    void recordForwardWithParams(VkCommandBuffer cmd,
                                 uint64_t paramAddr,
                                 VkBuffer rawInput, uint32_t inputOffset, uint32_t inputStride,
                                 VkBuffer encodedOutput, uint32_t outputOffset, uint32_t outputStride,
                                 uint32_t sampleCount) override;

private:
    Device &device;
    Config config;
    float perLevelScale{1.0f};

    std::unique_ptr<StorageBufferResource> tableBuffer;
    std::unique_ptr<StorageBufferResource> tableGradBuffer;
    uint64_t tableAddr{0};
    uint64_t tableGradAddr{0};
    VkDeviceSize tableBufferSize{0};

    std::unique_ptr<StorageBufferResource> adamState;

    std::unique_ptr<ComputePipeline> forwardPipeline;
    std::unique_ptr<ComputePipeline> backwardPipeline;
    std::unique_ptr<ComputePipeline> adamPipeline;

    PFN_vkGetBufferDeviceAddressKHR vkGetBufferDeviceAddressKHR{nullptr};
    uint64_t getBufferDeviceAddress(VkBuffer buffer);
};
