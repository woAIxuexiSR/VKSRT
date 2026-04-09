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

    outputs["color"] = {
        ldrImage.getImage(),
        ldrImage.getImageView(),
        ldrImage.getSampler(),
        VK_FORMAT_R8G8B8A8_UNORM,
        ldrImage.getExtent(),
        VK_IMAGE_LAYOUT_GENERAL,
    };
}

void TonemapPass::init()
{
    auto &input = inputs.at("color");

    tonemapPipeline = std::make_unique<ComputePipeline>(
        device, 1,
        std::vector<DescriptorLayoutBinding>{
            {VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, VK_SHADER_STAGE_COMPUTE_BIT},
            {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, VK_SHADER_STAGE_COMPUTE_BIT},
        },
        "../shaders/tonemap/tonemap.slang.spv",
        sizeof(TonemapPushConstants));

    tonemapPipeline->updateDescriptorSets({
        {VkDescriptorImageInfo{input.sampler, input.imageView, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL}},
        {VkDescriptorImageInfo{ldrImage.getSampler(), ldrImage.getImageView(), VK_IMAGE_LAYOUT_GENERAL}},
    });
}

void TonemapPass::drawUI()
{
    ImGui::SliderFloat("Exposure", &pushConstants.exposure, 0.1f, 10.0f);
}

void TonemapPass::recordCommand(VkCommandBuffer commandBuffer,
                                uint32_t currentFrame, uint32_t imageIndex)
{
    auto &input = inputs.at("color");
    auto extent = ldrImage.getExtent();

    if (!enabled)
    {
        // Bypass: blit input directly to output (handles format conversion)
        device.imageBarrier(commandBuffer, input.image,
                            VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                            VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT, VK_ACCESS_2_TRANSFER_READ_BIT,
                            VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR, VK_PIPELINE_STAGE_2_BLIT_BIT, 1);

        device.imageBarrier(commandBuffer, ldrImage.getImage(),
                            VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                            0, VK_ACCESS_2_TRANSFER_WRITE_BIT,
                            VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_2_BLIT_BIT, 1);

        VkImageBlit region{};
        region.srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
        region.srcOffsets[0] = {0, 0, 0};
        region.srcOffsets[1] = {(int32_t)extent.width, (int32_t)extent.height, 1};
        region.dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
        region.dstOffsets[0] = {0, 0, 0};
        region.dstOffsets[1] = {(int32_t)extent.width, (int32_t)extent.height, 1};

        vkCmdBlitImage(commandBuffer,
                       input.image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                       ldrImage.getImage(), VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                       1, &region, VK_FILTER_NEAREST);

        // Transition input back to GENERAL for next frame's RT accumulation
        device.imageBarrier(commandBuffer, input.image,
                            VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                            VK_ACCESS_2_TRANSFER_READ_BIT, VK_ACCESS_2_SHADER_READ_BIT,
                            VK_PIPELINE_STAGE_2_BLIT_BIT, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, 1);

        // Transition LDR to GENERAL (matches output slot layout)
        device.imageBarrier(commandBuffer, ldrImage.getImage(),
                            VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_GENERAL,
                            VK_ACCESS_2_TRANSFER_WRITE_BIT, VK_ACCESS_2_SHADER_READ_BIT,
                            VK_PIPELINE_STAGE_2_BLIT_BIT, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, 1);
        return;
    }

    // Transition RT output from GENERAL to SHADER_READ_ONLY for compute read
    device.imageBarrier(commandBuffer, input.image,
                        VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                        VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT, VK_ACCESS_2_SHADER_READ_BIT,
                        VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, 1);

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
}
