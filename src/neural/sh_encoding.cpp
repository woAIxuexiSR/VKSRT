#include "sh_encoding.h"

#include <stdexcept>

REGISTER_ENCODING_CPP(SHEncoding, "sh");

struct SHForwardPushConstants
{
    uint64_t inputBuffer;
    uint64_t outputBuffer;
    uint32_t sampleCount;
    uint32_t degree;
    uint32_t inputFieldOffset;
    uint32_t inputStride;
    uint32_t outputFieldOffset;
    uint32_t outputStride;
};

SHEncoding::SHEncoding(Device &_d, const json &params)
    : device(_d)
{
    int inputDim = 3;
    if (params.contains("inputDim")) inputDim = params["inputDim"].get<int>();
    if (params.contains("degree"))   degree = params["degree"].get<int>();

    if (inputDim != 3)
        throw std::runtime_error("SHEncoding: inputDim must be 3");
    if (degree < 0 || degree > 4)
        throw std::runtime_error("SHEncoding: degree must be in [0, 4]");
}

void SHEncoding::createPipelines()
{
    std::vector<DescriptorLayoutBinding> emptyBindings{};
    forwardPipeline = std::make_unique<ComputePipeline>(
        device, 1, emptyBindings,
        shaderPath("neural/sh_forward.spv"),
        sizeof(SHForwardPushConstants));
}

void SHEncoding::recordForward(VkCommandBuffer cmd,
                               VkBuffer rawInput, uint32_t inputOffset, uint32_t inputStride,
                               VkBuffer encodedOutput, uint32_t outputOffset, uint32_t outputStride,
                               uint32_t sampleCount)
{
    SHForwardPushConstants pc{};
    pc.inputBuffer = device.getBufferDeviceAddress(rawInput);
    pc.outputBuffer = device.getBufferDeviceAddress(encodedOutput);
    pc.sampleCount = sampleCount;
    pc.degree = (uint32_t)degree;
    pc.inputFieldOffset = inputOffset;
    pc.inputStride = inputStride;
    pc.outputFieldOffset = outputOffset;
    pc.outputStride = outputStride;

    forwardPipeline->bindPipeline(cmd);
    forwardPipeline->pushConstants(cmd, &pc);
    vkCmdDispatch(cmd, (sampleCount + 255) / 256, 1, 1);
}
