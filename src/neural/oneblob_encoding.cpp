#include "oneblob_encoding.h"

#include <stdexcept>
#include <cmath>

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

OneBlobEncoding::OneBlobEncoding(Device &_d, const Config &cfg)
    : device(_d), config(cfg)
{
    if (cfg.inputDim < 1 || cfg.inputDim > 4)
        throw std::runtime_error("OneBlobEncoding: inputDim must be in [1, 4]");
    if (cfg.numBins < 2 || cfg.numBins > 32)
        throw std::runtime_error("OneBlobEncoding: numBins must be in [2, 32]");

    if (config.sigma <= 0.0f)
        config.sigma = 1.0f / (float)(config.numBins - 1);

    vkGetBufferDeviceAddressKHR = device.loadDeviceFunction<PFN_vkGetBufferDeviceAddressKHR>(
        "vkGetBufferDeviceAddressKHR");
}

uint64_t OneBlobEncoding::getBufferDeviceAddress(VkBuffer buffer)
{
    VkBufferDeviceAddressInfoKHR info{};
    info.sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO_KHR;
    info.buffer = buffer;
    return vkGetBufferDeviceAddressKHR(device.getDevice(), &info);
}

void OneBlobEncoding::createPipelines()
{
    std::vector<DescriptorLayoutBinding> emptyBindings{};
    forwardPipeline = std::make_unique<ComputePipeline>(
        device, 1, emptyBindings,
        "build/shaders/neural/oneblob_forward.slang.spv",
        sizeof(OneBlobForwardPushConstants));
}

void OneBlobEncoding::recordForward(VkCommandBuffer cmd,
                                    VkBuffer rawInput, uint32_t inputOffset, uint32_t inputStride,
                                    VkBuffer encodedOutput, uint32_t outputOffset, uint32_t outputStride,
                                    uint32_t sampleCount)
{
    OneBlobForwardPushConstants pc{};
    pc.inputBuffer = getBufferDeviceAddress(rawInput);
    pc.outputBuffer = getBufferDeviceAddress(encodedOutput);
    pc.sampleCount = sampleCount;
    pc.inputDim = (uint32_t)config.inputDim;
    pc.numBins = (uint32_t)config.numBins;
    pc.sigma = config.sigma;
    pc.inputFieldOffset = inputOffset;
    pc.inputStride = inputStride;
    pc.outputFieldOffset = outputOffset;
    pc.outputStride = outputStride;

    forwardPipeline->bindPipeline(cmd);
    forwardPipeline->pushConstants(cmd, &pc);
    vkCmdDispatch(cmd, (sampleCount + 255) / 256, 1, 1);
}
