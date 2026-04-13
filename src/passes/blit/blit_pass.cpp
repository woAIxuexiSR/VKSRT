#include "blit_pass.h"

#include "imgui.h"

REGISTER_RENDER_PASS_CPP(BlitPass, "blit");

BlitPass::BlitPass(Device &_d, SwapChain &_sc, const json &params)
    : PassBase(_d, _sc)
{
}

void BlitPass::init()
{
    blitPipeline = std::make_unique<GraphicsPipeline>(
        device, 1,
        std::vector<DescriptorLayoutBinding>{
            {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_FRAGMENT_BIT},
        },
        std::vector<VkVertexInputBindingDescription>{},
        std::vector<VkVertexInputAttributeDescription>{},
        "../shaders/blit/blit.slang.spv",
        std::vector<VkFormat>{swapChain.getImageFormat()});

    blitPipeline->updateDescriptorSets({
        {VkDescriptorImageInfo{initInputSlot.sampler, initInputSlot.imageView, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL}},
    });
    lastBoundInputView = initInputSlot.imageView;
    lastBoundInputSampler = initInputSlot.sampler;
}

void BlitPass::drawUI()
{
    auto extent = swapChain.getExtent();
    ImGui::Text("Output: %ux%u", extent.width, extent.height);
}

PassImageSlot BlitPass::recordCommand(VkCommandBuffer commandBuffer,
                                       const PassImageSlot &inputSlot,
                                       uint32_t currentFrame, uint32_t imageIndex)
{
    if (inputSlot.imageView != lastBoundInputView || inputSlot.sampler != lastBoundInputSampler)
    {
        vkDeviceWaitIdle(device.getDevice());
        blitPipeline->updateDescriptorSets({
            {VkDescriptorImageInfo{inputSlot.sampler, inputSlot.imageView, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL}},
        });
        lastBoundInputView = inputSlot.imageView;
        lastBoundInputSampler = inputSlot.sampler;
    }

    // Transition input -> SHADER_READ_ONLY for fragment read
    device.imageBarrier(commandBuffer, inputSlot.image,
                        inputSlot.layout, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                        VK_ACCESS_2_MEMORY_WRITE_BIT, VK_ACCESS_2_SHADER_READ_BIT,
                        VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT, 1);

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

    return {};
}
