#include "ray_tracing_pass.h"

#include "camera.h"
#include "imgui.h"

REGISTER_RENDER_PASS_CPP(RayTracingPass, "ray_tracing");

RayTracingPass::RayTracingPass(Device &_d, SwapChain &_sc, const json &params)
    : PassBase(_d, _sc),
      colorImage{_d, VK_FORMAT_R32G32B32A32_SFLOAT, _sc.getExtent(),
                 VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT},
      uniformBuffer{_d, sizeof(RTUniform), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT},
      model{_d}
{
    SceneLoader::loadScene(params, model);

    model.buildAccelerationStructures();
}

void RayTracingPass::init()
{
    std::vector<DescriptorLayoutBinding> bindings = {
        {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, VK_SHADER_STAGE_RAYGEN_BIT_KHR},
        {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, VK_SHADER_STAGE_RAYGEN_BIT_KHR},
    };
    VkShaderStageFlags hitStages = VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR;
    auto modelBindings = model.getDescriptorBindings(hitStages);
    bindings.insert(bindings.end(), modelBindings.begin(), modelBindings.end());

    rtPipeline = std::make_unique<RayTracingPipeline>(
        device, 1, bindings,
        "build/shaders/ray_tracing/ray_tracing.spv",
        model.getHitSBTRecords(),
        "raygenMain", "missMain", "closestHitMain",
        sizeof(RTPushConstants));

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
    const char *modes[] = {"Material", "Position", "Normal", "UV"};
    ImGui::Combo("Shading Mode", &pushConstants.shadingMode, modes, IM_ARRAYSIZE(modes));
}

void RayTracingPass::update(uint32_t currentFrame, InputState &inputState)
{
    ubo.viewInverse = camera->getInverseViewMatrix();
    ubo.projInverse = camera->getInverseProjectionMatrix();

    uniformBuffer.update(&ubo);
}

PassImageSlot RayTracingPass::recordCommand(VkCommandBuffer commandBuffer,
                                             const PassImageSlot &inputSlot,
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
    rtPipeline->pushConstants(commandBuffer, &pushConstants);
    rtPipeline->traceRays(commandBuffer, {colorExtent.width, colorExtent.height, 1});

    return getOutputSlot();
}

PassImageSlot RayTracingPass::getOutputSlot() const
{
    return {
        colorImage.getImage(),
        colorImage.getImageView(),
        colorImage.getSampler(),
        VK_FORMAT_R32G32B32A32_SFLOAT,
        colorImage.getExtent(),
        VK_IMAGE_LAYOUT_GENERAL,
    };
}
