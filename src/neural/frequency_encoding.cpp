#include "frequency_encoding.h"

#include <stdexcept>

REGISTER_ENCODING_CPP(FrequencyEncoding, "frequency");

struct FrequencyForwardPushConstants
{
    uint64_t inputBuffer;
    uint64_t outputBuffer;
    uint32_t sampleCount;
    uint32_t inputDim;
    uint32_t numFreqs;
    uint32_t inputFieldOffset;
    uint32_t inputStride;
    uint32_t outputFieldOffset;
    uint32_t outputStride;
};

FrequencyEncoding::FrequencyEncoding(Device &_d, const json &params)
    : device(_d)
{
    if (params.contains("inputDim")) inputDim = params["inputDim"].get<int>();
    if (params.contains("numFreqs")) numFreqs = params["numFreqs"].get<int>();

    if (inputDim < 1 || inputDim > 4)
        throw std::runtime_error("FrequencyEncoding: inputDim must be in [1, 4]");
    if (numFreqs < 1 || numFreqs > 12)
        throw std::runtime_error("FrequencyEncoding: numFreqs must be in [1, 12]");
}

void FrequencyEncoding::createPipelines()
{
    std::vector<DescriptorLayoutBinding> emptyBindings{};
    forwardPipeline = std::make_unique<ComputePipeline>(
        device, 1, emptyBindings,
        shaderPath("neural/frequency_forward.spv"),
        sizeof(FrequencyForwardPushConstants));
}

void FrequencyEncoding::recordForward(VkCommandBuffer cmd,
                                      VkBuffer rawInput, uint32_t inputOffset, uint32_t inputStride,
                                      VkBuffer encodedOutput, uint32_t outputOffset, uint32_t outputStride,
                                      uint32_t sampleCount)
{
    FrequencyForwardPushConstants pc{};
    pc.inputBuffer = device.getBufferDeviceAddress(rawInput);
    pc.outputBuffer = device.getBufferDeviceAddress(encodedOutput);
    pc.sampleCount = sampleCount;
    pc.inputDim = (uint32_t)inputDim;
    pc.numFreqs = (uint32_t)numFreqs;
    pc.inputFieldOffset = inputOffset;
    pc.inputStride = inputStride;
    pc.outputFieldOffset = outputOffset;
    pc.outputStride = outputStride;

    forwardPipeline->bindPipeline(cmd);
    forwardPipeline->pushConstants(cmd, &pc);
    vkCmdDispatch(cmd, (sampleCount + 255) / 256, 1, 1);
}
