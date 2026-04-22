#include "identity_encoding.h"

REGISTER_ENCODING_CPP(IdentityEncoding, "identity");

struct IdentityForwardPushConstants
{
    uint64_t inputBuffer;
    uint64_t outputBuffer;
    uint32_t sampleCount;
    uint32_t dim;
    uint32_t inputFieldOffset;
    uint32_t inputStride;
    uint32_t outputFieldOffset;
    uint32_t outputStride;
};

IdentityEncoding::IdentityEncoding(Device &_d, const json &params)
    : device(_d)
{
    if (params.contains("inputDim")) inputDim = params["inputDim"].get<int>();
}

void IdentityEncoding::createPipelines()
{
    std::vector<DescriptorLayoutBinding> emptyBindings{};
    forwardPipeline = std::make_unique<ComputePipeline>(
        device, 1, emptyBindings,
        shaderPath("neural/identity_forward.spv"),
        sizeof(IdentityForwardPushConstants));
}

void IdentityEncoding::recordForward(VkCommandBuffer cmd,
                                     VkBuffer rawInput, uint32_t inputOffset, uint32_t inputStride,
                                     VkBuffer encodedOutput, uint32_t outputOffset, uint32_t outputStride,
                                     uint32_t sampleCount)
{
    IdentityForwardPushConstants pc{};
    pc.inputBuffer = device.getBufferDeviceAddress(rawInput);
    pc.outputBuffer = device.getBufferDeviceAddress(encodedOutput);
    pc.sampleCount = sampleCount;
    pc.dim = (uint32_t)inputDim;
    pc.inputFieldOffset = inputOffset;
    pc.inputStride = inputStride;
    pc.outputFieldOffset = outputOffset;
    pc.outputStride = outputStride;

    forwardPipeline->bindPipeline(cmd);
    forwardPipeline->pushConstants(cmd, &pc);
    vkCmdDispatch(cmd, (sampleCount + 255) / 256, 1, 1);
}
