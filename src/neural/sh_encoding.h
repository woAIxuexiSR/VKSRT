#pragma once

#include "encoding.h"

class SHEncoding : public Encoding
{
    REGISTER_ENCODING(SHEncoding);

public:
    SHEncoding(Device &device, const json &params);
    ~SHEncoding() = default;

    int getInputDim() const override { return 3; }
    int getOutputDim() const override { return (degree + 1) * (degree + 1); }
    bool hasTrainableParams() const override { return false; }
    std::string typeName() const override { return "sh"; }

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
    int degree{4};
    std::unique_ptr<ComputePipeline> forwardPipeline;
};
