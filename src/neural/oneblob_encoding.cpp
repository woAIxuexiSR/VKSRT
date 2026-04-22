#include "oneblob_encoding.h"

#include <stdexcept>
#include <cmath>

REGISTER_ENCODING_CPP(OneBlobEncoding, "oneblob");

struct OneBlobForwardPushConstants
{
    uint64_t inputBuffer;
    uint64_t outputBuffer;
    uint32_t sampleCount;
    uint32_t inputDim;
    uint32_t numBins;
    float    sigma;
    uint32_t inputFieldOffset;
    uint32_t inputStride;
    uint32_t outputFieldOffset;
    uint32_t outputStride;
};

OneBlobEncoding::OneBlobEncoding(Device &_d, const json &params)
    : device(_d)
{
    if (params.contains("inputDim")) inputDim = params["inputDim"].get<int>();
    if (params.contains("numBins"))  numBins  = params["numBins"].get<int>();
    if (params.contains("sigma"))    sigma    = params["sigma"].get<float>();

    if (inputDim < 1 || inputDim > 4)
        throw std::runtime_error("OneBlobEncoding: inputDim must be in [1, 4]");
    if (numBins < 2 || numBins > 32)
        throw std::runtime_error("OneBlobEncoding: numBins must be in [2, 32]");

    if (sigma <= 0.0f)
        sigma = 1.0f / (float)(numBins - 1);
}

void OneBlobEncoding::createPipelines()
{
    std::vector<DescriptorLayoutBinding> emptyBindings{};
    forwardPipeline = std::make_unique<ComputePipeline>(
        device, 1, emptyBindings,
        shaderPath("neural/oneblob_forward.spv"),
        sizeof(OneBlobForwardPushConstants));
}

void OneBlobEncoding::recordForward(VkCommandBuffer cmd,
                                    VkBuffer rawInput, uint32_t inputOffset, uint32_t inputStride,
                                    VkBuffer encodedOutput, uint32_t outputOffset, uint32_t outputStride,
                                    uint32_t sampleCount)
{
    OneBlobForwardPushConstants pc{};
    pc.inputBuffer = device.getBufferDeviceAddress(rawInput);
    pc.outputBuffer = device.getBufferDeviceAddress(encodedOutput);
    pc.sampleCount = sampleCount;
    pc.inputDim = (uint32_t)inputDim;
    pc.numBins = (uint32_t)numBins;
    pc.sigma = sigma;
    pc.inputFieldOffset = inputOffset;
    pc.inputStride = inputStride;
    pc.outputFieldOffset = outputOffset;
    pc.outputStride = outputStride;

    forwardPipeline->bindPipeline(cmd);
    forwardPipeline->pushConstants(cmd, &pc);
    vkCmdDispatch(cmd, (sampleCount + 255) / 256, 1, 1);
}
