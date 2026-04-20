#include "sh_encoding.h"

#include <stdexcept>

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

SHEncoding::SHEncoding(Device &_d, const Config &cfg)
    : device(_d), config(cfg)
{
    if (cfg.inputDim != 3)
        throw std::runtime_error("SHEncoding: inputDim must be 3");
    if (cfg.degree < 0 || cfg.degree > 4)
        throw std::runtime_error("SHEncoding: degree must be in [0, 4]");

    vkGetBufferDeviceAddressKHR = device.loadDeviceFunction<PFN_vkGetBufferDeviceAddressKHR>(
        "vkGetBufferDeviceAddressKHR");
}

uint64_t SHEncoding::getBufferDeviceAddress(VkBuffer buffer)
{
    VkBufferDeviceAddressInfoKHR info{};
    info.sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO_KHR;
    info.buffer = buffer;
    return vkGetBufferDeviceAddressKHR(device.getDevice(), &info);
}

void SHEncoding::createPipelines()
{
    std::vector<DescriptorLayoutBinding> emptyBindings{};
    forwardPipeline = std::make_unique<ComputePipeline>(
        device, 1, emptyBindings,
        "build/shaders/neural/sh_forward.slang.spv",
        sizeof(SHForwardPushConstants));
}

void SHEncoding::recordForward(VkCommandBuffer cmd,
                               VkBuffer rawInput, uint32_t inputOffset, uint32_t inputStride,
                               VkBuffer encodedOutput, uint32_t outputOffset, uint32_t outputStride,
                               uint32_t sampleCount)
{
    SHForwardPushConstants pc{};
    pc.inputBuffer = getBufferDeviceAddress(rawInput);
    pc.outputBuffer = getBufferDeviceAddress(encodedOutput);
    pc.sampleCount = sampleCount;
    pc.degree = (uint32_t)config.degree;
    pc.inputFieldOffset = inputOffset;
    pc.inputStride = inputStride;
    pc.outputFieldOffset = outputOffset;
    pc.outputStride = outputStride;

    forwardPipeline->bindPipeline(cmd);
    forwardPipeline->pushConstants(cmd, &pc);
    vkCmdDispatch(cmd, (sampleCount + 255) / 256, 1, 1);
}
