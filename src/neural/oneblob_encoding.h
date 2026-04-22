#pragma once

#include "encoding.h"

class OneBlobEncoding : public Encoding
{
    REGISTER_ENCODING(OneBlobEncoding);

public:
    OneBlobEncoding(Device &device, const json &params);
    ~OneBlobEncoding() = default;

    int getInputDim() const override { return inputDim; }
    int getOutputDim() const override { return inputDim * numBins; }
    bool hasTrainableParams() const override { return false; }
    std::string typeName() const override { return "oneblob"; }

    void createPipelines() override;

    void recordForward(VkCommandBuffer cmd,
                       VkBuffer rawInput, uint32_t inputOffset, uint32_t inputStride,
                       VkBuffer encodedOutput, uint32_t outputOffset, uint32_t outputStride,
                       uint32_t sampleCount) override;

    void recordBackward(VkCommandBuffer cmd,
                        VkBuffer dEncoded, uint32_t dOffset, uint32_t dStride,
                        VkBuffer rawInput, uint32_t inputOffset, uint32_t inputStride,
                        uint32_t sampleCount) override {}

private:
    Device &device;
    int inputDim{3};
    int numBins{16};
    float sigma{0.0f};
    std::unique_ptr<ComputePipeline> forwardPipeline;
};
