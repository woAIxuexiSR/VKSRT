#include "path_tracing_pass.h"

#include "camera.h"
#include "gbuffer.h"
#include "imgui.h"

REGISTER_RENDER_PASS_CPP(PathTracingPass, "path_tracing");

PathTracingPass::PathTracingPass(Device &_d, SwapChain &_sc, const json &params)
    : PassBase(_d, _sc),
      colorImage{_d, VK_FORMAT_R32G32B32A32_SFLOAT, _sc.getExtent(),
                 VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT},
      uniformBuffer{_d, sizeof(PTUniform), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT}
{
    if (params.contains("maxDepth"))
        pushConstants.maxDepth = params["maxDepth"].get<int>();
    if (params.contains("rrDepth"))
        pushConstants.rrDepth = params["rrDepth"].get<int>();
}

void PathTracingPass::init()
{
    pushConstants.lightCount = scene->getLightCount();

    std::vector<DescriptorLayoutBinding> bindings = {
        {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, VK_SHADER_STAGE_RAYGEN_BIT_KHR},
        {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, VK_SHADER_STAGE_RAYGEN_BIT_KHR},
    };
    VkShaderStageFlags hitStages = VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR;
    auto modelBindings = scene->getDescriptorBindings(hitStages);
    bindings.insert(bindings.end(), modelBindings.begin(), modelBindings.end());
    // G-buffer: normal (binding 10), position (binding 11), albedo (binding 12)
    bindings.push_back({VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, VK_SHADER_STAGE_RAYGEN_BIT_KHR});
    bindings.push_back({VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, VK_SHADER_STAGE_RAYGEN_BIT_KHR});
    bindings.push_back({VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, VK_SHADER_STAGE_RAYGEN_BIT_KHR});

    rtPipeline = std::make_unique<RayTracingPipeline>(
        device, 1, bindings,
        shaderPath("path_tracing/path_tracing.spv"),
        scene->getHitSBTRecords(),
        "raygenMain", "missMain", "closestHitMain",
        sizeof(PTPushConstants));

    std::vector<std::vector<DescriptorInfo>> infos = {
        {VkDescriptorImageInfo{colorImage.getSampler(), colorImage.getImageView(), VK_IMAGE_LAYOUT_GENERAL}},
        {VkDescriptorBufferInfo{uniformBuffer.getBuffer(), 0, sizeof(PTUniform)}},
    };
    auto modelInfos = scene->getDescriptorInfos();
    infos.insert(infos.end(), modelInfos.begin(), modelInfos.end());
    // G-buffer descriptors (App-managed)
    infos.push_back({VkDescriptorImageInfo{gbuffer->getNormalSampler(), gbuffer->getNormalImageView(), VK_IMAGE_LAYOUT_GENERAL}});
    infos.push_back({VkDescriptorImageInfo{gbuffer->getPositionSampler(), gbuffer->getPositionImageView(), VK_IMAGE_LAYOUT_GENERAL}});
    infos.push_back({VkDescriptorImageInfo{gbuffer->getAlbedoSampler(), gbuffer->getAlbedoImageView(), VK_IMAGE_LAYOUT_GENERAL}});
    rtPipeline->updateDescriptorSets(infos);
}

void PathTracingPass::drawUI()
{
    ImGui::SliderInt("Max Depth", &pushConstants.maxDepth, 1, 32);
    ImGui::SliderInt("RR Depth", &pushConstants.rrDepth, 1, 16);
    ImGui::Text("Frame Index: %d", pushConstants.frameIndex);

    ImGui::Separator();
    ImGui::Text("Light Count: %d", pushConstants.lightCount);
    bool nee = pushConstants.useNEE != 0;
    if (ImGui::Checkbox("Use NEE", &nee))
        pushConstants.useNEE = nee ? 1 : 0;
    if (pushConstants.useNEE)
    {
        bool mis = pushConstants.useMIS != 0;
        if (ImGui::Checkbox("Use MIS", &mis))
            pushConstants.useMIS = mis ? 1 : 0;
    }
}

void PathTracingPass::update(uint32_t currentFrame, InputState &inputState)
{
    // Detect NEE/MIS parameter changes to trigger reset
    static int lastNEE = pushConstants.useNEE;
    static int lastMIS = pushConstants.useMIS;
    static int lastMaxDepth = pushConstants.maxDepth;
    static int lastRRDepth = pushConstants.rrDepth;
    if (pushConstants.useNEE != lastNEE || pushConstants.useMIS != lastMIS ||
        pushConstants.maxDepth != lastMaxDepth || pushConstants.rrDepth != lastRRDepth)
    {
        inputState.keyboardChanged = true;
        lastNEE = pushConstants.useNEE;
        lastMIS = pushConstants.useMIS;
        lastMaxDepth = pushConstants.maxDepth;
        lastRRDepth = pushConstants.rrDepth;
    }

    if (!inputState.isChanged())
        pushConstants.frameIndex++;
    else
        pushConstants.frameIndex = 0;

    ubo.viewInverse = camera->getInverseViewMatrix();
    ubo.projInverse = camera->getInverseProjectionMatrix();

    uniformBuffer.update(&ubo);
}

PassImageSlot PathTracingPass::recordCommand(VkCommandBuffer commandBuffer,
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

    // Transition G-buffer images to GENERAL for RT write
    // Non-first-frame: use GENERAL->GENERAL since bilateral may be disabled (no layout change)
    VkImageLayout gbOldLayout = firstFrame ? VK_IMAGE_LAYOUT_UNDEFINED : VK_IMAGE_LAYOUT_GENERAL;
    VkAccessFlags2 gbSrcAccess = firstFrame ? (VkAccessFlags2)0 : (VK_ACCESS_2_SHADER_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT);
    VkPipelineStageFlags2 gbSrcStage = firstFrame ? VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT : (VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR);

    device.imageBarrier(commandBuffer, gbuffer->getNormalImage(),
                        gbOldLayout, VK_IMAGE_LAYOUT_GENERAL,
                        gbSrcAccess, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                        gbSrcStage, VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR, 1);
    device.imageBarrier(commandBuffer, gbuffer->getPositionImage(),
                        gbOldLayout, VK_IMAGE_LAYOUT_GENERAL,
                        gbSrcAccess, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                        gbSrcStage, VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR, 1);
    device.imageBarrier(commandBuffer, gbuffer->getAlbedoImage(),
                        gbOldLayout, VK_IMAGE_LAYOUT_GENERAL,
                        gbSrcAccess, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                        gbSrcStage, VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR, 1);

    firstFrame = false;

    rtPipeline->bindPipeline(commandBuffer);
    rtPipeline->bindDescriptorSets(commandBuffer, currentFrame);
    rtPipeline->pushConstants(commandBuffer, &pushConstants);
    rtPipeline->traceRays(commandBuffer, {colorExtent.width, colorExtent.height, 1});

    gbuffer->markWritten();

    return getOutputSlot();
}

PassImageSlot PathTracingPass::getOutputSlot() const
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
