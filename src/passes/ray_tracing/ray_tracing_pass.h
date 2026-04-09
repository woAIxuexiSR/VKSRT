#pragma once

#include "pass_base.h"
#include "pipeline.h"
#include "resource.h"
#include "cornell_box.h"
#include "camera.h"

#include <memory>
#include <chrono>

struct RTUniform
{
    alignas(4) int frameIndex{0};
    alignas(4) int _pad0{0};
    alignas(4) int _pad1{0};
    alignas(4) int _pad2{0};
    alignas(16) glm::mat4 viewInverse;
    alignas(16) glm::mat4 projInverse;
};

class RayTracingPass : public PassBase
{
private:
    std::unique_ptr<RayTracingPipeline> rtPipeline;
    std::unique_ptr<GraphicsPipeline> blitPipeline;

    ImageResource colorImage;
    RTUniform ubo;
    UniformBufferResource uniformBuffer;

    CornellBox scene;
    Camera camera;
    float lastTime{0.0f};

public:
    RayTracingPass(Device &_d, SwapChain &swapChain)
        : PassBase(_d),
          colorImage{_d, VK_FORMAT_R8G8B8A8_UNORM, swapChain.getExtent(),
                     VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT},
          uniformBuffer{_d, sizeof(RTUniform), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT},
          scene{_d},
          camera{{0.0f, -4.5f, 1.2f}, {0.0f, 0.0f, 1.2f},
                 swapChain.getExtent().width / (float)swapChain.getExtent().height, 45.0f, 0.1f, 100.0f}
    {
        // RT pipeline: bindings 0-1 are pass-owned, 2-6 come from scene
        std::vector<DescriptorLayoutBinding> bindings = {
            {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, VK_SHADER_STAGE_RAYGEN_BIT_KHR},
            {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, VK_SHADER_STAGE_RAYGEN_BIT_KHR},
        };
        auto sceneBindings = scene.getDescriptorLayoutBindings();
        bindings.insert(bindings.end(), sceneBindings.begin(), sceneBindings.end());

        rtPipeline = std::make_unique<RayTracingPipeline>(
            device, 1, bindings,
            "../shaders/ray_tracing/ray_tracing.slang.spv",
            "raygenMain", "missMain", "closesthitMain",
            scene.getHitSBTRecords());

        // Blit pipeline: fullscreen triangle sampling the RT output
        blitPipeline = std::make_unique<GraphicsPipeline>(
            device, 1,
            std::vector<DescriptorLayoutBinding>{
                {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_FRAGMENT_BIT},
            },
            std::vector<VkVertexInputBindingDescription>{},
            std::vector<VkVertexInputAttributeDescription>{},
            "../shaders/ray_tracing/blit.slang.spv",
            std::vector<VkFormat>{swapChain.getImageFormat()});

        // Update RT pipeline descriptors
        std::vector<std::vector<DescriptorInfo>> infos = {
            {VkDescriptorImageInfo{colorImage.getSampler(), colorImage.getImageView(), VK_IMAGE_LAYOUT_GENERAL}},
            {VkDescriptorBufferInfo{uniformBuffer.getBuffer(), 0, sizeof(RTUniform)}},
        };
        auto sceneInfos = scene.getDescriptorInfos();
        infos.insert(infos.end(), sceneInfos.begin(), sceneInfos.end());
        rtPipeline->updateDescriptorSets(infos);

        // Update blit pipeline descriptors
        blitPipeline->updateDescriptorSets({
            {VkDescriptorImageInfo{colorImage.getSampler(), colorImage.getImageView(), VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL}},
        });
    }

    std::string getName() const override { return "Ray Tracing"; }

    void drawUI() override
    {
        ImGui::Text("Frame Index: %d", ubo.frameIndex);
    }

    void update(uint32_t currentFrame, InputState &inputState) override
    {
        static auto startTime = std::chrono::high_resolution_clock::now();
        auto currentTime = std::chrono::high_resolution_clock::now();
        float time = std::chrono::duration<float, std::chrono::seconds::period>(currentTime - startTime).count();

        camera.processInput(inputState, time - lastTime);
        lastTime = time;

        ubo.viewInverse = camera.getInverseViewMatrix();
        ubo.projInverse = camera.getInverseProjectionMatrix();

        if (!inputState.isChanged())
            ubo.frameIndex++;
        else
            ubo.frameIndex = 0;

        uniformBuffer.update(&ubo);
    }

    void recordCommand(VkCommandBuffer commandBuffer, SwapChain &swapChain,
                       uint32_t currentFrame, uint32_t imageIndex) override
    {
        auto swapChainExtent = swapChain.getExtent();
        auto colorExtent = colorImage.getExtent();

        // Transition storage image to GENERAL for RT writes
        device.imageBarrier(commandBuffer, colorImage.getImage(),
                            VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL,
                            0, VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                            VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR, 1);

        // Trace rays
        rtPipeline->bindPipeline(commandBuffer);
        rtPipeline->bindDescriptorSets(commandBuffer, currentFrame);
        rtPipeline->traceRays(commandBuffer, {colorExtent.width, colorExtent.height, 1});

        // Transition storage image to SHADER_READ_ONLY for blit fragment read
        device.imageBarrier(commandBuffer, colorImage.getImage(),
                            VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                            VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT, VK_ACCESS_2_SHADER_READ_BIT,
                            VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR, VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT, 1);

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
        renderingInfo.renderArea = {{0, 0}, swapChainExtent};
        renderingInfo.layerCount = 1;
        renderingInfo.colorAttachmentCount = 1;
        renderingInfo.pColorAttachments = &colorAttachment;
        renderingInfo.pDepthAttachment = VK_NULL_HANDLE;

        vkCmdBeginRendering(commandBuffer, &renderingInfo);

        device.bindViewport(commandBuffer, swapChainExtent);
        blitPipeline->bindPipeline(commandBuffer);
        blitPipeline->bindDescriptorSets(commandBuffer, currentFrame);
        vkCmdDraw(commandBuffer, 3, 1, 0, 0);

        vkCmdEndRendering(commandBuffer);
    }
};
