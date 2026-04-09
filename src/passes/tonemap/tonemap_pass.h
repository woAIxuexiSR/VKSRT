#pragma once

#include "pass_base.h"
#include "pipeline.h"
#include "resource.h"

#include <memory>

class TonemapPass : public PassBase
{
private:
    std::unique_ptr<ComputePipeline> tonemapPipeline;
    std::unique_ptr<GraphicsPipeline> blitPipeline;

    ImageResource &inputImage;
    ImageResource ldrImage;

public:
    TonemapPass(Device &_d, SwapChain &swapChain, ImageResource &_inputImage)
        : PassBase(_d),
          inputImage(_inputImage),
          ldrImage{_d, VK_FORMAT_R8G8B8A8_UNORM, swapChain.getExtent(),
                   VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT}
    {
        // Compute tonemap pipeline: binding 0 = input (sampled), binding 1 = output (storage)
        tonemapPipeline = std::make_unique<ComputePipeline>(
            device, 1,
            std::vector<DescriptorLayoutBinding>{
                {VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, VK_SHADER_STAGE_COMPUTE_BIT},
                {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, VK_SHADER_STAGE_COMPUTE_BIT},
            },
            "../shaders/tonemap/tonemap.slang.spv");

        tonemapPipeline->updateDescriptorSets({
            {VkDescriptorImageInfo{inputImage.getSampler(), inputImage.getImageView(), VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL}},
            {VkDescriptorImageInfo{ldrImage.getSampler(), ldrImage.getImageView(), VK_IMAGE_LAYOUT_GENERAL}},
        });

        // Blit pipeline: fullscreen triangle sampling the tonemapped LDR output
        blitPipeline = std::make_unique<GraphicsPipeline>(
            device, 1,
            std::vector<DescriptorLayoutBinding>{
                {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_FRAGMENT_BIT},
            },
            std::vector<VkVertexInputBindingDescription>{},
            std::vector<VkVertexInputAttributeDescription>{},
            "../shaders/ray_tracing/blit.slang.spv",
            std::vector<VkFormat>{swapChain.getImageFormat()});

        blitPipeline->updateDescriptorSets({
            {VkDescriptorImageInfo{ldrImage.getSampler(), ldrImage.getImageView(), VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL}},
        });
    }

    std::string getName() const override { return "Tonemap"; }

    void recordCommand(VkCommandBuffer commandBuffer, SwapChain &swapChain,
                       uint32_t currentFrame, uint32_t imageIndex) override
    {
        auto extent = ldrImage.getExtent();

        // Transition RT output from GENERAL to SHADER_READ_ONLY for compute read
        device.imageBarrier(commandBuffer, inputImage.getImage(),
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
        uint32_t groupsX = (extent.width + 15) / 16;
        uint32_t groupsY = (extent.height + 15) / 16;
        vkCmdDispatch(commandBuffer, groupsX, groupsY, 1);

        // Transition LDR image to SHADER_READ_ONLY for blit fragment read
        device.imageBarrier(commandBuffer, ldrImage.getImage(),
                            VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                            VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT, VK_ACCESS_2_SHADER_READ_BIT,
                            VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT, 1);

        // Transition swapchain image for rendering
        device.imageBarrier(commandBuffer, swapChain.getImage(imageIndex),
                            VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                            0, VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
                            VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT, 1);

        // Fullscreen blit via dynamic rendering
        VkRenderingAttachmentInfo colorAttachment{};
        colorAttachment.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
        colorAttachment.imageView = swapChain.getImageView(imageIndex);
        colorAttachment.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
        colorAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
        colorAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
        colorAttachment.clearValue.color = {{0.0f, 0.0f, 0.0f, 1.0f}};

        VkRenderingInfo renderingInfo{};
        renderingInfo.sType = VK_STRUCTURE_TYPE_RENDERING_INFO;
        renderingInfo.renderArea = {{0, 0}, swapChain.getExtent()};
        renderingInfo.layerCount = 1;
        renderingInfo.colorAttachmentCount = 1;
        renderingInfo.pColorAttachments = &colorAttachment;
        renderingInfo.pDepthAttachment = VK_NULL_HANDLE;

        vkCmdBeginRendering(commandBuffer, &renderingInfo);

        device.bindViewport(commandBuffer, swapChain.getExtent());
        blitPipeline->bindPipeline(commandBuffer);
        blitPipeline->bindDescriptorSets(commandBuffer, currentFrame);
        vkCmdDraw(commandBuffer, 3, 1, 0, 0);

        vkCmdEndRendering(commandBuffer);
    }
};
