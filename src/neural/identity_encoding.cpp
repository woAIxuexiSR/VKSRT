#include "identity_encoding.h"

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

IdentityEncoding::IdentityEncoding(Device &_d, const Config &cfg)
    : device(_d), config(cfg)
{
    vkGetBufferDeviceAddressKHR = device.loadDeviceFunction<PFN_vkGetBufferDeviceAddressKHR>(
        "vkGetBufferDeviceAddressKHR");
}

uint64_t IdentityEncoding::getBufferDeviceAddress(VkBuffer buffer)
{
    VkBufferDeviceAddressInfoKHR info{};
    info.sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO_KHR;
    info.buffer = buffer;
    return vkGetBufferDeviceAddressKHR(device.getDevice(), &info);
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
    pc.inputBuffer = getBufferDeviceAddress(rawInput);
    pc.outputBuffer = getBufferDeviceAddress(encodedOutput);
    pc.sampleCount = sampleCount;
    pc.dim = (uint32_t)config.inputDim;
    pc.inputFieldOffset = inputOffset;
    pc.inputStride = inputStride;
    pc.outputFieldOffset = outputOffset;
    pc.outputStride = outputStride;

    forwardPipeline->bindPipeline(cmd);
    forwardPipeline->pushConstants(cmd, &pc);
    vkCmdDispatch(cmd, (sampleCount + 255) / 256, 1, 1);
}
