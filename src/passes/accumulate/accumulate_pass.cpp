#include "accumulate_pass.h"

#include "imgui.h"

REGISTER_RENDER_PASS_CPP(AccumulatePass, "accumulate");

AccumulatePass::AccumulatePass(Device &_d, SwapChain &_sc, const json &params)
    : PassBase(_d, _sc),
      accumImage{_d, VK_FORMAT_R32G32B32A32_SFLOAT, _sc.getExtent(),
                 VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT}
{
}

void AccumulatePass::init()
{
    accumulatePipeline = std::make_unique<ComputePipeline>(
        device, 1,
        std::vector<DescriptorLayoutBinding>{
            {VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, VK_SHADER_STAGE_COMPUTE_BIT},
            {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, VK_SHADER_STAGE_COMPUTE_BIT},
        },
        "../shaders/accumulate/accumulate.slang.spv",
        sizeof(AccumulatePushConstants));

    accumulatePipeline->updateDescriptorSets({
        {VkDescriptorImageInfo{initInputSlot.sampler, initInputSlot.imageView, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL}},
        {VkDescriptorImageInfo{accumImage.getSampler(), accumImage.getImageView(), VK_IMAGE_LAYOUT_GENERAL}},
    });
    lastBoundInputView = initInputSlot.imageView;
    lastBoundInputSampler = initInputSlot.sampler;
}

void AccumulatePass::drawUI()
{
    ImGui::Text("Frame Index: %d", pushConstants.frameIndex);
}

void AccumulatePass::update(uint32_t currentFrame, InputState &inputState)
{
    if (!enabled)
    {
        wasEnabled = false;
        return;
    }

    if (!inputState.isChanged() && wasEnabled)
        pushConstants.frameIndex++;
    else
        pushConstants.frameIndex = 0;

    wasEnabled = true;
}

PassImageSlot AccumulatePass::recordCommand(VkCommandBuffer commandBuffer,
                                             const PassImageSlot &inputSlot,
                                             uint32_t currentFrame, uint32_t imageIndex)
{
    if (!enabled)
        return inputSlot;

    if (inputSlot.imageView != lastBoundInputView || inputSlot.sampler != lastBoundInputSampler)
    {
        vkDeviceWaitIdle(device.getDevice());
        accumulatePipeline->updateDescriptorSets({
            {VkDescriptorImageInfo{inputSlot.sampler, inputSlot.imageView, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL}},
            {VkDescriptorImageInfo{accumImage.getSampler(), accumImage.getImageView(), VK_IMAGE_LAYOUT_GENERAL}},
        });
        lastBoundInputView = inputSlot.imageView;
        lastBoundInputSampler = inputSlot.sampler;
    }

    auto extent = accumImage.getExtent();

    // Transition RT output: GENERAL -> SHADER_READ_ONLY for compute read
    device.imageBarrier(commandBuffer, inputSlot.image,
                        VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                        VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT, VK_ACCESS_2_SHADER_READ_BIT,
                        VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, 1);

    // Transition accumulation buffer to GENERAL for compute read/write
    VkImageLayout oldLayout = firstFrame ? VK_IMAGE_LAYOUT_UNDEFINED : VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    VkAccessFlags2 srcAccess = firstFrame ? (VkAccessFlags2)0 : VK_ACCESS_2_SHADER_READ_BIT;
    VkPipelineStageFlags2 srcStage = firstFrame ? VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT : VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;

    device.imageBarrier(commandBuffer, accumImage.getImage(),
                        oldLayout, VK_IMAGE_LAYOUT_GENERAL,
                        srcAccess, VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                        srcStage, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, 1);

    firstFrame = false;

    // Dispatch accumulation compute shader
    accumulatePipeline->bindPipeline(commandBuffer);
    accumulatePipeline->bindDescriptorSets(commandBuffer, currentFrame);
    accumulatePipeline->pushConstants(commandBuffer, &pushConstants);
    uint32_t groupsX = (extent.width + 15) / 16;
    uint32_t groupsY = (extent.height + 15) / 16;
    vkCmdDispatch(commandBuffer, groupsX, groupsY, 1);

    return getOutputSlot();
}

PassImageSlot AccumulatePass::getOutputSlot() const
{
    return {
        accumImage.getImage(),
        accumImage.getImageView(),
        accumImage.getSampler(),
        VK_FORMAT_R32G32B32A32_SFLOAT,
        accumImage.getExtent(),
        VK_IMAGE_LAYOUT_GENERAL,
    };
}
