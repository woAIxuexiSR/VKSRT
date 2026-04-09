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

    ImageResource colorImage;
    RTUniform ubo;
    UniformBufferResource uniformBuffer;

    CornellBox scene;
    Camera camera;
    float lastTime{0.0f};

public:
    RayTracingPass(Device &_d, SwapChain &swapChain)
        : PassBase(_d),
          colorImage{_d, VK_FORMAT_R32G32B32A32_SFLOAT, swapChain.getExtent(),
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

        // Update RT pipeline descriptors
        std::vector<std::vector<DescriptorInfo>> infos = {
            {VkDescriptorImageInfo{colorImage.getSampler(), colorImage.getImageView(), VK_IMAGE_LAYOUT_GENERAL}},
            {VkDescriptorBufferInfo{uniformBuffer.getBuffer(), 0, sizeof(RTUniform)}},
        };
        auto sceneInfos = scene.getDescriptorInfos();
        infos.insert(infos.end(), sceneInfos.begin(), sceneInfos.end());
        rtPipeline->updateDescriptorSets(infos);
    }

    std::string getName() const override { return "Ray Tracing"; }

    ImageResource &getColorImage() { return colorImage; }

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
    }
};
