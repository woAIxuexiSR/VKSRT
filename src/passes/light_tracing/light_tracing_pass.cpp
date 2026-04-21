#include "light_tracing_pass.h"

#include "camera.h"
#include "gbuffer.h"
#include "imgui.h"

REGISTER_RENDER_PASS_CPP(LightTracingPass, "light_tracing");

LightTracingPass::LightTracingPass(Device &_d, SwapChain &_sc, const json &params)
    : PassBase(_d, _sc),
      splatR{_d, VK_FORMAT_R32_SFLOAT, _sc.getExtent(),
             VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT},
      splatG{_d, VK_FORMAT_R32_SFLOAT, _sc.getExtent(),
             VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT},
      splatB{_d, VK_FORMAT_R32_SFLOAT, _sc.getExtent(),
             VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT},
      colorImage{_d, VK_FORMAT_R32G32B32A32_SFLOAT, _sc.getExtent(),
                 VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT},
      uniformBuffer{_d, sizeof(LTUniform), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT},
      model{_d}
{
    if (params.contains("maxDepth"))
        pushConstants.maxDepth = params["maxDepth"].get<int>();
    if (params.contains("rrDepth"))
        pushConstants.rrDepth = params["rrDepth"].get<int>();

    SceneLoader::loadScene(params, model);

    model.buildAccelerationStructures();
    pushConstants.lightCount = model.getLightCount();
}

void LightTracingPass::init()
{
    // RT pipeline: TLAS(0), uniform(1), vertices(2), indices(3), materials(4),
    //              normals(5), texcoords(6), lights(7), instanceTransforms(8),
    //              splatR(9), splatG(10), splatB(11)
    VkShaderStageFlags hitStages = VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR | VK_SHADER_STAGE_RAYGEN_BIT_KHR;
    std::vector<DescriptorLayoutBinding> rtBindings = {
        {VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR, VK_SHADER_STAGE_RAYGEN_BIT_KHR},
        {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, VK_SHADER_STAGE_RAYGEN_BIT_KHR},
        {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, hitStages},
        {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, hitStages},
        {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, hitStages},
        {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, hitStages},
        {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, hitStages},
        {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_RAYGEN_BIT_KHR},
        {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_RAYGEN_BIT_KHR},
        {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, VK_SHADER_STAGE_RAYGEN_BIT_KHR},
        {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, VK_SHADER_STAGE_RAYGEN_BIT_KHR},
        {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, VK_SHADER_STAGE_RAYGEN_BIT_KHR},
    };

    rtPipeline = std::make_unique<RayTracingPipeline>(
        device, 1, rtBindings,
        shaderPath("light_tracing/light_tracing.spv"),
        model.getHitSBTRecords(),
        "raygenMain", "missMain", "closestHitMain",
        sizeof(LTPushConstants));

    // Descriptor infos for RT pipeline
    auto modelInfos = model.getDescriptorInfos();
    // modelInfos order: TLAS, vertices, indices, materials, normals, texcoords, lights, instanceTransforms
    std::vector<std::vector<DescriptorInfo>> rtInfos;
    rtInfos.push_back(modelInfos[0]); // TLAS (binding 0)
    rtInfos.push_back({VkDescriptorBufferInfo{uniformBuffer.getBuffer(), 0, sizeof(LTUniform)}}); // uniform (binding 1)
    for (size_t i = 1; i < modelInfos.size(); i++) // vertices through instanceTransforms (bindings 2-8)
        rtInfos.push_back(modelInfos[i]);
    // splat images (bindings 9-11)
    rtInfos.push_back({VkDescriptorImageInfo{splatR.getSampler(), splatR.getImageView(), VK_IMAGE_LAYOUT_GENERAL}});
    rtInfos.push_back({VkDescriptorImageInfo{splatG.getSampler(), splatG.getImageView(), VK_IMAGE_LAYOUT_GENERAL}});
    rtInfos.push_back({VkDescriptorImageInfo{splatB.getSampler(), splatB.getImageView(), VK_IMAGE_LAYOUT_GENERAL}});
    rtPipeline->updateDescriptorSets(rtInfos);

    // Compose compute pipeline: splatR(0), splatG(1), splatB(2), colorImage(3)
    std::vector<DescriptorLayoutBinding> composeBindings = {
        {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, VK_SHADER_STAGE_COMPUTE_BIT},
        {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, VK_SHADER_STAGE_COMPUTE_BIT},
        {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, VK_SHADER_STAGE_COMPUTE_BIT},
        {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, VK_SHADER_STAGE_COMPUTE_BIT},
    };

    composePipeline = std::make_unique<ComputePipeline>(
        device, 1, composeBindings,
        shaderPath("light_tracing/lt_compose.spv"));

    std::vector<std::vector<DescriptorInfo>> composeInfos = {
        {VkDescriptorImageInfo{splatR.getSampler(), splatR.getImageView(), VK_IMAGE_LAYOUT_GENERAL}},
        {VkDescriptorImageInfo{splatG.getSampler(), splatG.getImageView(), VK_IMAGE_LAYOUT_GENERAL}},
        {VkDescriptorImageInfo{splatB.getSampler(), splatB.getImageView(), VK_IMAGE_LAYOUT_GENERAL}},
        {VkDescriptorImageInfo{colorImage.getSampler(), colorImage.getImageView(), VK_IMAGE_LAYOUT_GENERAL}},
    };
    composePipeline->updateDescriptorSets(composeInfos);
}

void LightTracingPass::drawUI()
{
    ImGui::SliderInt("Max Depth", &pushConstants.maxDepth, 1, 32);
    ImGui::SliderInt("RR Depth", &pushConstants.rrDepth, 1, 16);
    ImGui::Text("Frame Index: %d", pushConstants.frameIndex);
    ImGui::Text("Light Count: %d", pushConstants.lightCount);
}

void LightTracingPass::update(uint32_t currentFrame, InputState &inputState)
{
    static int lastMaxDepth = pushConstants.maxDepth;
    static int lastRRDepth = pushConstants.rrDepth;
    if (pushConstants.maxDepth != lastMaxDepth || pushConstants.rrDepth != lastRRDepth)
    {
        inputState.keyboardChanged = true;
        lastMaxDepth = pushConstants.maxDepth;
        lastRRDepth = pushConstants.rrDepth;
    }

    if (!inputState.isChanged())
        pushConstants.frameIndex++;
    else
        pushConstants.frameIndex = 0;

    ubo.viewProj = camera->getProjectionMatrix() * camera->getViewMatrix();
    ubo.viewInverse = camera->getInverseViewMatrix();
    auto extent = colorImage.getExtent();
    ubo.screenWidth = extent.width;
    ubo.screenHeight = extent.height;
    ubo.tanHalfFov = tan(glm::radians(camera->fov) * 0.5f);
    uniformBuffer.update(&ubo);
}

PassImageSlot LightTracingPass::recordCommand(VkCommandBuffer commandBuffer,
                                               const PassImageSlot &inputSlot,
                                               uint32_t currentFrame, uint32_t imageIndex)
{
    auto extent = colorImage.getExtent();

    // 1. Clear splat images to 0
    VkClearColorValue clearValue = {{0.0f, 0.0f, 0.0f, 0.0f}};

    // Barrier: previous layout -> TRANSFER_DST for clear
    VkImageLayout splatOldLayout = firstFrame ? VK_IMAGE_LAYOUT_UNDEFINED : VK_IMAGE_LAYOUT_GENERAL;
    VkAccessFlags2 splatSrcAccess = firstFrame ? (VkAccessFlags2)0 : VK_ACCESS_2_SHADER_STORAGE_READ_BIT;
    VkPipelineStageFlags2 splatSrcStage = firstFrame ? VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT : VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
    for (auto *img : {&splatR, &splatG, &splatB})
    {
        device.imageBarrier(commandBuffer, img->getImage(),
                            splatOldLayout, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                            splatSrcAccess, VK_ACCESS_2_TRANSFER_WRITE_BIT,
                            splatSrcStage, VK_PIPELINE_STAGE_2_TRANSFER_BIT, 1);
    }
    for (auto *img : {&splatR, &splatG, &splatB})
    {
        VkImageSubresourceRange range{VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        vkCmdClearColorImage(commandBuffer, img->getImage(),
                             VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &clearValue, 1, &range);
    }
    // Barrier: TRANSFER_DST -> GENERAL for RT atomic writes
    for (auto *img : {&splatR, &splatG, &splatB})
    {
        device.imageBarrier(commandBuffer, img->getImage(),
                            VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_GENERAL,
                            VK_ACCESS_2_TRANSFER_WRITE_BIT,
                            VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                            VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR, 1);
    }

    // 2. Trace rays (light tracing with float atomic splat)
    rtPipeline->bindPipeline(commandBuffer);
    rtPipeline->bindDescriptorSets(commandBuffer, currentFrame);
    rtPipeline->pushConstants(commandBuffer, &pushConstants);
    uint32_t ps = static_cast<uint32_t>(pushConstants.photonSize);
    rtPipeline->traceRays(commandBuffer, {ps, ps, 1});

    // 3. Barrier: wait for RT atomic writes before compose reads
    for (auto *img : {&splatR, &splatG, &splatB})
    {
        device.imageBarrier(commandBuffer, img->getImage(),
                            VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_GENERAL,
                            VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                            VK_ACCESS_2_SHADER_STORAGE_READ_BIT,
                            VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR,
                            VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, 1);
    }
    // Transition color output to GENERAL for compose write
    VkImageLayout colorOldLayout = firstFrame ? VK_IMAGE_LAYOUT_UNDEFINED : VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    device.imageBarrier(commandBuffer, colorImage.getImage(),
                        colorOldLayout, VK_IMAGE_LAYOUT_GENERAL,
                        0, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                        VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, 1);

    // 4. Compose: read splatR/G/B -> write colorImage
    composePipeline->bindPipeline(commandBuffer);
    composePipeline->bindDescriptorSets(commandBuffer, currentFrame);
    vkCmdDispatch(commandBuffer, (extent.width + 15) / 16, (extent.height + 15) / 16, 1);

    // 5. Barrier: compose output -> SHADER_READ_ONLY for downstream TAA
    device.imageBarrier(commandBuffer, colorImage.getImage(),
                        VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                        VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT, VK_ACCESS_2_SHADER_READ_BIT,
                        VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, 1);

    firstFrame = false;
    return getOutputSlot();
}

PassImageSlot LightTracingPass::getOutputSlot() const
{
    return {
        colorImage.getImage(),
        colorImage.getImageView(),
        colorImage.getSampler(),
        VK_FORMAT_R32G32B32A32_SFLOAT,
        colorImage.getExtent(),
        VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
    };
}
