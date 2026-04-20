#pragma once

#include "encoding.h"

class IdentityEncoding : public Encoding
{
public:
    struct Config
    {
        int inputDim{3};
    };

    IdentityEncoding(Device &device, const Config &cfg);
    ~IdentityEncoding() = default;

    int getInputDim() const override { return config.inputDim; }
    int getOutputDim() const override { return config.inputDim; }
    bool hasTrainableParams() const override { return false; }
    std::string typeName() const override { return "identity"; }

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
    Config config;
    std::unique_ptr<ComputePipeline> forwardPipeline;

    PFN_vkGetBufferDeviceAddressKHR vkGetBufferDeviceAddressKHR{nullptr};
    uint64_t getBufferDeviceAddress(VkBuffer buffer);
};
