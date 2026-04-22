#pragma once

#include "encoding.h"

#include <memory>
#include <cstdint>

class HashGridEncoding : public Encoding
{
    REGISTER_ENCODING(HashGridEncoding);

public:
    HashGridEncoding(Device &device, const json &params);
    ~HashGridEncoding() = default;

    HashGridEncoding(const HashGridEncoding &) = delete;
    HashGridEncoding &operator=(const HashGridEncoding &) = delete;

    int getInputDim() const override { return inputDim; }
    int getOutputDim() const override { return numLevels * featuresPerLevel; }
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

    uint64_t getParamBufferAddress() const override { return tableAddr; }
    VkDeviceSize getParamBufferSize() const override { return tableBufferSize; }
    VkBuffer getParamBuffer() const override { return tableBuffer->getBuffer(); }

    void recordForwardWithParams(VkCommandBuffer cmd,
                                 uint64_t paramAddr,
                                 VkBuffer rawInput, uint32_t inputOffset, uint32_t inputStride,
                                 VkBuffer encodedOutput, uint32_t outputOffset, uint32_t outputStride,
                                 uint32_t sampleCount) override;

private:
    Device &device;

    int numLevels{16};
    int featuresPerLevel{2};
    int tableSize{1 << 19};
    int coarsestResolution{16};
    int finestResolution{2048};
    int inputDim{2};

    float perLevelScale{1.0f};

    int getTotalFeatures() const { return numLevels * tableSize * featuresPerLevel; }

    std::unique_ptr<StorageBufferResource> tableBuffer;
    std::unique_ptr<StorageBufferResource> tableGradBuffer;
    uint64_t tableAddr{0};
    uint64_t tableGradAddr{0};
    VkDeviceSize tableBufferSize{0};

    std::unique_ptr<StorageBufferResource> adamState;

    std::unique_ptr<ComputePipeline> forwardPipeline;
    std::unique_ptr<ComputePipeline> backwardPipeline;
    std::unique_ptr<ComputePipeline> adamPipeline;
};
