#pragma once

#include "pass_base.h"
#include "pipeline.h"
#include "resource.h"
#include "ray_tracing_model.h"
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

    RayTracingModel model;
    Camera camera;
    float lastTime{0.0f};
    bool firstFrame{true};

    void createScene()
    {
        // Simple triangle for testing
        std::vector<glm::vec3> vertices = {
            {0.0f, 0.0f, 0.0f},
            {1.0f, 0.0f, 0.0f},
            {0.5f, 1.0f, 0.0f},
        };
        std::vector<glm::uvec3> indices = {{0, 1, 2}};
        model.insertMesh(vertices, indices, glm::vec4(0.0f, 1.0f, 0.0f, 0.0f)); // green triangle

        model.buildAccelerationStructures();
    }

public:
    RayTracingPass(Device &_d, SwapChain &swapChain)
        : PassBase(_d),
          colorImage{_d, VK_FORMAT_R32G32B32A32_SFLOAT, swapChain.getExtent(),
                     VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT},
          uniformBuffer{_d, sizeof(RTUniform), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT},
          model{_d},
          camera{{0.5f, -2.0f, 0.5f}, {0.5f, 0.0f, 0.5f},
                 swapChain.getExtent().width / (float)swapChain.getExtent().height, 45.0f, 0.1f, 100.0f}
    {
        createScene();

        // RT pipeline: bindings 0-1 are pass-owned, 2-5 come from model
        std::vector<DescriptorLayoutBinding> bindings = {
            {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, VK_SHADER_STAGE_RAYGEN_BIT_KHR},
            {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, VK_SHADER_STAGE_RAYGEN_BIT_KHR},
        };
        auto modelBindings = model.getDescriptorLayoutBindings();
        bindings.insert(bindings.end(), modelBindings.begin(), modelBindings.end());

        rtPipeline = std::make_unique<RayTracingPipeline>(
            device, 1, bindings,
            "../shaders/ray_tracing/raygen.spv",
            "../shaders/ray_tracing/miss.spv",
            "../shaders/ray_tracing/closesthit.spv",
            model.getHitSBTRecords());

        // Update RT pipeline descriptors
        std::vector<std::vector<DescriptorInfo>> infos = {
            {VkDescriptorImageInfo{colorImage.getSampler(), colorImage.getImageView(), VK_IMAGE_LAYOUT_GENERAL}},
            {VkDescriptorBufferInfo{uniformBuffer.getBuffer(), 0, sizeof(RTUniform)}},
        };
        auto modelInfos = model.getDescriptorInfos();
        infos.insert(infos.end(), modelInfos.begin(), modelInfos.end());
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
        // First frame: UNDEFINED (no prior content). Subsequent frames: SHADER_READ_ONLY_OPTIMAL
        // (left by tonemap pass), preserving data for progressive accumulation.
        VkImageLayout oldLayout = firstFrame ? VK_IMAGE_LAYOUT_UNDEFINED : VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        VkAccessFlags2 srcAccess = firstFrame ? (VkAccessFlags2)0 : VK_ACCESS_2_SHADER_READ_BIT;
        VkPipelineStageFlags2 srcStage = firstFrame ? VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT : VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;

        device.imageBarrier(commandBuffer, colorImage.getImage(),
                            oldLayout, VK_IMAGE_LAYOUT_GENERAL,
                            srcAccess, VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                            srcStage, VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR, 1);

        firstFrame = false;

        // Trace rays
        rtPipeline->bindPipeline(commandBuffer);
        rtPipeline->bindDescriptorSets(commandBuffer, currentFrame);
        rtPipeline->traceRays(commandBuffer, {colorExtent.width, colorExtent.height, 1});
    }
};
