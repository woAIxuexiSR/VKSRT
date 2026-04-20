#pragma once

#include "device.h"
#include "resource.h"
#include "pipeline.h"

#include <memory>
#include <string>
#include <cstdint>

class Encoding
{
public:
    virtual ~Encoding() = default;

    virtual int getInputDim() const = 0;
    virtual int getOutputDim() const = 0;
    virtual bool hasTrainableParams() const = 0;
    virtual int getTrainableParamCount() const { return 0; }
    virtual std::string typeName() const = 0;

    virtual void createPipelines() = 0;
    virtual void initParams(unsigned int seed) {}
    virtual void resetAdamState() {}

    virtual void recordForward(VkCommandBuffer cmd,
                               VkBuffer rawInput, uint32_t inputOffset, uint32_t inputStride,
                               VkBuffer encodedOutput, uint32_t outputOffset, uint32_t outputStride,
                               uint32_t sampleCount) = 0;

    virtual void recordBackward(VkCommandBuffer cmd,
                                VkBuffer dEncoded, uint32_t dOffset, uint32_t dStride,
                                VkBuffer rawInput, uint32_t inputOffset, uint32_t inputStride,
                                uint32_t sampleCount) = 0;

    virtual void recordZeroGrads(VkCommandBuffer cmd) {}
    virtual void recordAdam(VkCommandBuffer cmd) {}
};
