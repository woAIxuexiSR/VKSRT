#pragma once

#include "pass_base.h"
#include "pipeline.h"

#include <memory>

class TrianglePass : public PassBase
{
private:
    std::unique_ptr<GraphicsPipeline> pipeline;

public:
    TrianglePass(Device &_d, VkFormat swapChainFormat)
        : PassBase(_d)
    {
        pipeline = std::make_unique<GraphicsPipeline>(
            device,
            1, // descriptor set count
            std::vector<DescriptorLayoutBinding>{}, // no descriptors
            std::vector<VkVertexInputBindingDescription>{},   // no vertex input
            std::vector<VkVertexInputAttributeDescription>{},
            "../shaders/triangle/triangle.slang.spv",
            std::vector<VkFormat>{swapChainFormat});
    }

    std::string getName() const override { return "Triangle"; }

    void recordCommand(VkCommandBuffer commandBuffer, SwapChain &swapChain,
                       uint32_t currentFrame, uint32_t imageIndex) override
    {
        auto extent = swapChain.getExtent();

        // Transition swapchain image to color attachment
        device.imageBarrier(commandBuffer, swapChain.getImage(imageIndex),
                            VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                            0, VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
                            VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT, 1);

        // Begin dynamic rendering
        VkRenderingAttachmentInfo colorAttachment{};
        colorAttachment.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
        colorAttachment.imageView = swapChain.getImageView(imageIndex);
        colorAttachment.imageLayout = VK_IMAGE_LAYOUT_ATTACHMENT_OPTIMAL;
        colorAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
        colorAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
        colorAttachment.clearValue.color = {{0.1f, 0.1f, 0.1f, 1.0f}};

        VkRenderingAttachmentInfo depthAttachment{};
        depthAttachment.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
        depthAttachment.imageView = swapChain.getDepthImageView();
        depthAttachment.imageLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
        depthAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
        depthAttachment.storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        depthAttachment.clearValue.depthStencil.depth = 1.0f;

        VkRenderingInfo renderingInfo{};
        renderingInfo.sType = VK_STRUCTURE_TYPE_RENDERING_INFO;
        renderingInfo.renderArea.offset = {0, 0};
        renderingInfo.renderArea.extent = extent;
        renderingInfo.layerCount = 1;
        renderingInfo.colorAttachmentCount = 1;
        renderingInfo.pColorAttachments = &colorAttachment;
        renderingInfo.pDepthAttachment = &depthAttachment;

        vkCmdBeginRendering(commandBuffer, &renderingInfo);

        pipeline->bindPipeline(commandBuffer);
        device.bindViewport(commandBuffer, extent);

        // Draw 3 vertices (fullscreen triangle with colors generated in shader)
        vkCmdDraw(commandBuffer, 3, 1, 0, 0);

        vkCmdEndRendering(commandBuffer);
    }
};
