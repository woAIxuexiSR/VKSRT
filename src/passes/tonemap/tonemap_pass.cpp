#include "tonemap_pass.h"

#include "imgui.h"

REGISTER_RENDER_PASS_CPP(TonemapPass, "tonemap");

TonemapPass::TonemapPass(Device &_d, SwapChain &_sc, const json &params)
    : PassBase(_d, _sc),
      ldrImage{_d, VK_FORMAT_R8G8B8A8_UNORM, _sc.getExtent(),
               VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT}
{
    if (params.contains("exposure"))
        pushConstants.exposure = params["exposure"].get<float>();
}

void TonemapPass::init()
{
    tonemapPipeline = std::make_unique<ComputePipeline>(
        device, 1,
        std::vector<DescriptorLayoutBinding>{
            {VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, VK_SHADER_STAGE_COMPUTE_BIT},
            {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, VK_SHADER_STAGE_COMPUTE_BIT},
        },
        "build/shaders/tonemap/tonemap.slang.spv",
        sizeof(TonemapPushConstants));

    tonemapPipeline->updateDescriptorSets({
        {VkDescriptorImageInfo{initInputSlot.sampler, initInputSlot.imageView, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL}},
        {VkDescriptorImageInfo{ldrImage.getSampler(), ldrImage.getImageView(), VK_IMAGE_LAYOUT_GENERAL}},
    });
    lastBoundInputView = initInputSlot.imageView;
    lastBoundInputSampler = initInputSlot.sampler;
}

void TonemapPass::drawUI()
{
    ImGui::SliderFloat("Exposure", &pushConstants.exposure, 0.1f, 10.0f);
}

PassImageSlot TonemapPass::recordCommand(VkCommandBuffer commandBuffer,
                                          const PassImageSlot &inputSlot,
                                          uint32_t currentFrame, uint32_t imageIndex)
{
    if (!enabled)
        return inputSlot;

    if (inputSlot.imageView != lastBoundInputView || inputSlot.sampler != lastBoundInputSampler)
    {
        vkDeviceWaitIdle(device.getDevice());
        tonemapPipeline->updateDescriptorSets({
            {VkDescriptorImageInfo{inputSlot.sampler, inputSlot.imageView, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL}},
            {VkDescriptorImageInfo{ldrImage.getSampler(), ldrImage.getImageView(), VK_IMAGE_LAYOUT_GENERAL}},
        });
        lastBoundInputView = inputSlot.imageView;
        lastBoundInputSampler = inputSlot.sampler;
    }

    auto extent = ldrImage.getExtent();

    // Transition input -> SHADER_READ_ONLY for compute read
    device.imageBarrier(commandBuffer, inputSlot.image,
                        inputSlot.layout, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                        VK_ACCESS_2_MEMORY_WRITE_BIT, VK_ACCESS_2_SHADER_READ_BIT,
                        VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, 1);

    // Transition LDR image to GENERAL for compute write
    device.imageBarrier(commandBuffer, ldrImage.getImage(),
                        VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL,
                        0, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                        VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, 1);

    // Dispatch tonemap compute shader
    tonemapPipeline->bindPipeline(commandBuffer);
    tonemapPipeline->bindDescriptorSets(commandBuffer, currentFrame);
    tonemapPipeline->pushConstants(commandBuffer, &pushConstants);
    uint32_t groupsX = (extent.width + 15) / 16;
    uint32_t groupsY = (extent.height + 15) / 16;
    vkCmdDispatch(commandBuffer, groupsX, groupsY, 1);

    return getOutputSlot();
}

PassImageSlot TonemapPass::getOutputSlot() const
{
    return {
        ldrImage.getImage(),
        ldrImage.getImageView(),
        ldrImage.getSampler(),
        VK_FORMAT_R8G8B8A8_UNORM,
        ldrImage.getExtent(),
        VK_IMAGE_LAYOUT_GENERAL,
    };
}
