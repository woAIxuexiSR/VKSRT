#include "frequency_encoding.h"

#include <stdexcept>

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

FrequencyEncoding::FrequencyEncoding(Device &_d, const Config &cfg)
    : device(_d), config(cfg)
{
    if (cfg.inputDim < 1 || cfg.inputDim > 4)
        throw std::runtime_error("FrequencyEncoding: inputDim must be in [1, 4]");
    if (cfg.numFreqs < 1 || cfg.numFreqs > 12)
        throw std::runtime_error("FrequencyEncoding: numFreqs must be in [1, 12]");

    vkGetBufferDeviceAddressKHR = device.loadDeviceFunction<PFN_vkGetBufferDeviceAddressKHR>(
        "vkGetBufferDeviceAddressKHR");
}

uint64_t FrequencyEncoding::getBufferDeviceAddress(VkBuffer buffer)
{
    VkBufferDeviceAddressInfoKHR info{};
    info.sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO_KHR;
    info.buffer = buffer;
    return vkGetBufferDeviceAddressKHR(device.getDevice(), &info);
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
    pc.inputBuffer = getBufferDeviceAddress(rawInput);
    pc.outputBuffer = getBufferDeviceAddress(encodedOutput);
    pc.sampleCount = sampleCount;
    pc.inputDim = (uint32_t)config.inputDim;
    pc.numFreqs = (uint32_t)config.numFreqs;
    pc.inputFieldOffset = inputOffset;
    pc.inputStride = inputStride;
    pc.outputFieldOffset = outputOffset;
    pc.outputStride = outputStride;

    forwardPipeline->bindPipeline(cmd);
    forwardPipeline->pushConstants(cmd, &pc);
    vkCmdDispatch(cmd, (sampleCount + 255) / 256, 1, 1);
}
