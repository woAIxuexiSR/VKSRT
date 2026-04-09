#include "ray_tracing_pass.h"

#include <chrono>
#include "imgui.h"

REGISTER_RENDER_PASS_CPP(RayTracingPass, "ray_tracing");

void RayTracingPass::createScene()
{
    std::vector<glm::vec3> vertices = {
        {0.0f, 0.0f, 0.0f},
        {1.0f, 0.0f, 0.0f},
        {0.5f, 1.0f, 0.0f},
    };
    std::vector<glm::uvec3> indices = {{0, 1, 2}};
    model.insertMesh(vertices, indices, glm::vec4(0.0f, 1.0f, 0.0f, 0.0f));

    model.buildAccelerationStructures();
}

RayTracingPass::RayTracingPass(Device &_d, SwapChain &_sc, const json &params)
    : PassBase(_d, _sc),
      colorImage{_d, VK_FORMAT_R32G32B32A32_SFLOAT, _sc.getExtent(),
                 VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT},
      uniformBuffer{_d, sizeof(RTUniform), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT},
      model{_d},
      camera{{0.5f, -2.0f, 0.5f}, {0.5f, 0.0f, 0.5f},
             _sc.getExtent().width / (float)_sc.getExtent().height, 45.0f, 0.1f, 100.0f}
{
    createScene();

    // Register output slot
    outputs["color"] = {
        colorImage.getImage(),
        colorImage.getImageView(),
        colorImage.getSampler(),
        VK_FORMAT_R32G32B32A32_SFLOAT,
        colorImage.getExtent(),
        VK_IMAGE_LAYOUT_GENERAL,
    };
}

void RayTracingPass::init()
{
    std::vector<DescriptorLayoutBinding> bindings = {
        {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, VK_SHADER_STAGE_RAYGEN_BIT_KHR},
        {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, VK_SHADER_STAGE_RAYGEN_BIT_KHR},
    };
    auto modelBindings = model.getDescriptorLayoutBindings();
    bindings.insert(bindings.end(), modelBindings.begin(), modelBindings.end());

    rtPipeline = std::make_unique<RayTracingPipeline>(
        device, 1, bindings,
        "../shaders/ray_tracing/ray_tracing.spv",
        model.getHitSBTRecords());

    std::vector<std::vector<DescriptorInfo>> infos = {
        {VkDescriptorImageInfo{colorImage.getSampler(), colorImage.getImageView(), VK_IMAGE_LAYOUT_GENERAL}},
        {VkDescriptorBufferInfo{uniformBuffer.getBuffer(), 0, sizeof(RTUniform)}},
    };
    auto modelInfos = model.getDescriptorInfos();
    infos.insert(infos.end(), modelInfos.begin(), modelInfos.end());
    rtPipeline->updateDescriptorSets(infos);
}

void RayTracingPass::drawUI()
{
    ImGui::Text("Frame Index: %d", ubo.frameIndex);
}

void RayTracingPass::update(uint32_t currentFrame, InputState &inputState)
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

void RayTracingPass::recordCommand(VkCommandBuffer commandBuffer,
                                   uint32_t currentFrame, uint32_t imageIndex)
{
    auto colorExtent = colorImage.getExtent();

    VkImageLayout oldLayout = firstFrame ? VK_IMAGE_LAYOUT_UNDEFINED : VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    VkAccessFlags2 srcAccess = firstFrame ? (VkAccessFlags2)0 : VK_ACCESS_2_SHADER_READ_BIT;
    VkPipelineStageFlags2 srcStage = firstFrame ? VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT : VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;

    device.imageBarrier(commandBuffer, colorImage.getImage(),
                        oldLayout, VK_IMAGE_LAYOUT_GENERAL,
                        srcAccess, VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                        srcStage, VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR, 1);

    firstFrame = false;

    rtPipeline->bindPipeline(commandBuffer);
    rtPipeline->bindDescriptorSets(commandBuffer, currentFrame);
    rtPipeline->traceRays(commandBuffer, {colorExtent.width, colorExtent.height, 1});
}
