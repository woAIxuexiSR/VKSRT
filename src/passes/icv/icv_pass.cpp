#include "icv_pass.h"

#include "camera.h"
#include "gbuffer.h"
#include "imgui.h"

REGISTER_RENDER_PASS_CPP(ICVPass, "icv");

ICVPass::ICVPass(Device &_d, SwapChain &_sc, const json &params)
    : PassBase(_d, _sc),
      anchorImage{_d, VK_FORMAT_R32G32B32A32_SFLOAT, _sc.getExtent(),
                  VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT},
      hEdgeLeft{_d, VK_FORMAT_R32G32B32A32_SFLOAT, _sc.getExtent(),
                VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT},
      hEdgeRight{_d, VK_FORMAT_R32G32B32A32_SFLOAT, _sc.getExtent(),
                 VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT},
      vEdgeUp{_d, VK_FORMAT_R32G32B32A32_SFLOAT, _sc.getExtent(),
              VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT},
      vEdgeDown{_d, VK_FORMAT_R32G32B32A32_SFLOAT, _sc.getExtent(),
                VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT},
      outputImage{_d, VK_FORMAT_R32G32B32A32_SFLOAT, _sc.getExtent(),
                  VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT},
      uniformBuffer{_d, sizeof(ICVUniform), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT}
{
    if (params.contains("maxDepth"))
        pushConstants.maxDepth = params["maxDepth"].get<int>();
    if (params.contains("rrDepth"))
        pushConstants.rrDepth = params["rrDepth"].get<int>();
    if (params.contains("a"))
        pushConstants.a = params["a"].get<float>();
    if (params.contains("gaussianSigma"))
        pushConstants.gaussianSigma = params["gaussianSigma"].get<float>();
    if (params.contains("useNEE"))
        pushConstants.useNEE = params["useNEE"].get<int>();
    if (params.contains("useMIS"))
        pushConstants.useMIS = params["useMIS"].get<int>();

    pushConstants.screenWidth = (int)_sc.getExtent().width;
    pushConstants.screenHeight = (int)_sc.getExtent().height;
}

void ICVPass::init()
{
    pushConstants.lightCount = scene->getLightCount();

    VkShaderStageFlags hitStages = VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR;
    auto modelBindings = scene->getDescriptorBindings(hitStages);
    auto modelInfos = scene->getDescriptorInfos();

    std::vector<DescriptorLayoutBinding> anchorBindings = {
        {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, VK_SHADER_STAGE_RAYGEN_BIT_KHR},
        {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, VK_SHADER_STAGE_RAYGEN_BIT_KHR},
    };
    anchorBindings.insert(anchorBindings.end(), modelBindings.begin(), modelBindings.end());
    anchorBindings.push_back({VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, VK_SHADER_STAGE_RAYGEN_BIT_KHR});
    anchorBindings.push_back({VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, VK_SHADER_STAGE_RAYGEN_BIT_KHR});
    anchorBindings.push_back({VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, VK_SHADER_STAGE_RAYGEN_BIT_KHR});

    anchorPipeline = std::make_unique<RayTracingPipeline>(
        device, 1, anchorBindings,
        shaderPath("icv/icv_anchor.spv"),
        scene->getHitSBTRecords(),
        "raygenMain", "missMain", "closestHitMain",
        sizeof(ICVPushConstants));

    std::vector<std::vector<DescriptorInfo>> anchorInfos = {
        {VkDescriptorImageInfo{anchorImage.getSampler(), anchorImage.getImageView(), VK_IMAGE_LAYOUT_GENERAL}},
        {VkDescriptorBufferInfo{uniformBuffer.getBuffer(), 0, sizeof(ICVUniform)}},
    };
    anchorInfos.insert(anchorInfos.end(), modelInfos.begin(), modelInfos.end());
    anchorInfos.push_back({VkDescriptorImageInfo{gbuffer->getNormalSampler(), gbuffer->getNormalImageView(), VK_IMAGE_LAYOUT_GENERAL}});
    anchorInfos.push_back({VkDescriptorImageInfo{gbuffer->getPositionSampler(), gbuffer->getPositionImageView(), VK_IMAGE_LAYOUT_GENERAL}});
    anchorInfos.push_back({VkDescriptorImageInfo{gbuffer->getAlbedoSampler(), gbuffer->getAlbedoImageView(), VK_IMAGE_LAYOUT_GENERAL}});
    anchorPipeline->updateDescriptorSets(anchorInfos);

    std::vector<DescriptorLayoutBinding> edgeBindings = {
        {VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, VK_SHADER_STAGE_RAYGEN_BIT_KHR}, // 0: anchorImage
        {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, VK_SHADER_STAGE_RAYGEN_BIT_KHR}, // 1: hEdgeLeft
        {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, VK_SHADER_STAGE_RAYGEN_BIT_KHR}, // 2: hEdgeRight
        {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, VK_SHADER_STAGE_RAYGEN_BIT_KHR}, // 3: vEdgeUp
        {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, VK_SHADER_STAGE_RAYGEN_BIT_KHR}, // 4: vEdgeDown
        {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, VK_SHADER_STAGE_RAYGEN_BIT_KHR}, // 5: camera
    };
    edgeBindings.insert(edgeBindings.end(), modelBindings.begin(), modelBindings.end());

    edgePipeline = std::make_unique<RayTracingPipeline>(
        device, 1, edgeBindings,
        shaderPath("icv/icv_edge.spv"),
        scene->getHitSBTRecords(),
        "raygenMain", "missMain", "closestHitMain",
        sizeof(ICVPushConstants));

    std::vector<std::vector<DescriptorInfo>> edgeInfos = {
        {VkDescriptorImageInfo{anchorImage.getSampler(), anchorImage.getImageView(), VK_IMAGE_LAYOUT_GENERAL}},
        {VkDescriptorImageInfo{hEdgeLeft.getSampler(), hEdgeLeft.getImageView(), VK_IMAGE_LAYOUT_GENERAL}},
        {VkDescriptorImageInfo{hEdgeRight.getSampler(), hEdgeRight.getImageView(), VK_IMAGE_LAYOUT_GENERAL}},
        {VkDescriptorImageInfo{vEdgeUp.getSampler(), vEdgeUp.getImageView(), VK_IMAGE_LAYOUT_GENERAL}},
        {VkDescriptorImageInfo{vEdgeDown.getSampler(), vEdgeDown.getImageView(), VK_IMAGE_LAYOUT_GENERAL}},
        {VkDescriptorBufferInfo{uniformBuffer.getBuffer(), 0, sizeof(ICVUniform)}},
    };
    edgeInfos.insert(edgeInfos.end(), modelInfos.begin(), modelInfos.end());
    edgePipeline->updateDescriptorSets(edgeInfos);

    composePipeline = std::make_unique<ComputePipeline>(
        device, 1,
        std::vector<DescriptorLayoutBinding>{
            {VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, VK_SHADER_STAGE_COMPUTE_BIT}, // anchorImage
            {VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, VK_SHADER_STAGE_COMPUTE_BIT}, // hEdgeLeft
            {VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, VK_SHADER_STAGE_COMPUTE_BIT}, // hEdgeRight
            {VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, VK_SHADER_STAGE_COMPUTE_BIT}, // vEdgeUp
            {VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, VK_SHADER_STAGE_COMPUTE_BIT}, // vEdgeDown
            {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, VK_SHADER_STAGE_COMPUTE_BIT}, // outputImage
        },
        shaderPath("icv/icv_compose.spv"),
        sizeof(ICVPushConstants));

    composePipeline->updateDescriptorSets({
        {VkDescriptorImageInfo{anchorImage.getSampler(), anchorImage.getImageView(), VK_IMAGE_LAYOUT_GENERAL}},
        {VkDescriptorImageInfo{hEdgeLeft.getSampler(), hEdgeLeft.getImageView(), VK_IMAGE_LAYOUT_GENERAL}},
        {VkDescriptorImageInfo{hEdgeRight.getSampler(), hEdgeRight.getImageView(), VK_IMAGE_LAYOUT_GENERAL}},
        {VkDescriptorImageInfo{vEdgeUp.getSampler(), vEdgeUp.getImageView(), VK_IMAGE_LAYOUT_GENERAL}},
        {VkDescriptorImageInfo{vEdgeDown.getSampler(), vEdgeDown.getImageView(), VK_IMAGE_LAYOUT_GENERAL}},
        {VkDescriptorImageInfo{outputImage.getSampler(), outputImage.getImageView(), VK_IMAGE_LAYOUT_GENERAL}},
    });
}

void ICVPass::drawUI()
{
    ImGui::SliderInt("Max Depth", &pushConstants.maxDepth, 1, 32);
    ImGui::SliderInt("RR Depth", &pushConstants.rrDepth, 1, 16);
    ImGui::SliderFloat("Control Variate a", &pushConstants.a, 0.0f, 1.0f);
    ImGui::SliderFloat("Gaussian Sigma", &pushConstants.gaussianSigma, 0.1f, 4.0f);
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

void ICVPass::update(uint32_t currentFrame, InputState &inputState)
{
    static int lastNEE = pushConstants.useNEE;
    static int lastMIS = pushConstants.useMIS;
    static int lastMaxDepth = pushConstants.maxDepth;
    static int lastRRDepth = pushConstants.rrDepth;
    static float lastA = pushConstants.a;
    static float lastSigma = pushConstants.gaussianSigma;

    if (pushConstants.useNEE != lastNEE || pushConstants.useMIS != lastMIS ||
        pushConstants.maxDepth != lastMaxDepth || pushConstants.rrDepth != lastRRDepth ||
        pushConstants.a != lastA || pushConstants.gaussianSigma != lastSigma)
    {
        inputState.keyboardChanged = true;
        lastNEE = pushConstants.useNEE;
        lastMIS = pushConstants.useMIS;
        lastMaxDepth = pushConstants.maxDepth;
        lastRRDepth = pushConstants.rrDepth;
        lastA = pushConstants.a;
        lastSigma = pushConstants.gaussianSigma;
    }

    if (!inputState.isChanged())
        pushConstants.frameIndex++;
    else
        pushConstants.frameIndex = 0;

    auto extent = outputImage.getExtent();
    pushConstants.screenWidth = (int)extent.width;
    pushConstants.screenHeight = (int)extent.height;

    ubo.viewInverse = camera->getInverseViewMatrix();
    ubo.projInverse = camera->getInverseProjectionMatrix();
    uniformBuffer.update(&ubo);
}

PassImageSlot ICVPass::recordCommand(VkCommandBuffer commandBuffer,
                                      const PassImageSlot &inputSlot,
                                      uint32_t currentFrame, uint32_t imageIndex)
{
    auto extent = outputImage.getExtent();

    VkImageLayout anchorOldLayout = firstFrame ? VK_IMAGE_LAYOUT_UNDEFINED : VK_IMAGE_LAYOUT_GENERAL;
    VkAccessFlags2 anchorSrcAccess = firstFrame ? (VkAccessFlags2)0 : VK_ACCESS_2_SHADER_READ_BIT;
    VkPipelineStageFlags2 anchorSrcStage = firstFrame ? VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT : VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;

    device.imageBarrier(commandBuffer, anchorImage.getImage(),
                        anchorOldLayout, VK_IMAGE_LAYOUT_GENERAL,
                        anchorSrcAccess, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                        anchorSrcStage, VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR, 1);

    VkImageLayout edgeOldLayout = firstFrame ? VK_IMAGE_LAYOUT_UNDEFINED : VK_IMAGE_LAYOUT_GENERAL;
    VkAccessFlags2 edgeSrcAccess = firstFrame ? (VkAccessFlags2)0 : VK_ACCESS_2_SHADER_READ_BIT;
    VkPipelineStageFlags2 edgeSrcStage = firstFrame ? VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT : VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
    auto edgeReadToWrite = [&](ImageResource &image) {
        device.imageBarrier(commandBuffer, image.getImage(),
                            edgeOldLayout, VK_IMAGE_LAYOUT_GENERAL,
                            edgeSrcAccess, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                            edgeSrcStage, VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR, 1);
    };
    edgeReadToWrite(hEdgeLeft);
    edgeReadToWrite(hEdgeRight);
    edgeReadToWrite(vEdgeUp);
    edgeReadToWrite(vEdgeDown);

    VkImageLayout outputOldLayout = firstFrame ? VK_IMAGE_LAYOUT_UNDEFINED : VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    VkAccessFlags2 outputSrcAccess = firstFrame ? (VkAccessFlags2)0 : VK_ACCESS_2_SHADER_READ_BIT;
    VkPipelineStageFlags2 outputSrcStage = firstFrame ? VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT : VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;

    device.imageBarrier(commandBuffer, outputImage.getImage(),
                        outputOldLayout, VK_IMAGE_LAYOUT_GENERAL,
                        outputSrcAccess, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                        outputSrcStage, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, 1);

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

    anchorPipeline->bindPipeline(commandBuffer);
    anchorPipeline->bindDescriptorSets(commandBuffer, currentFrame);
    anchorPipeline->pushConstants(commandBuffer, &pushConstants);
    anchorPipeline->traceRays(commandBuffer, {extent.width, extent.height, 1});

    gbuffer->markWritten();

    device.imageBarrier(commandBuffer, anchorImage.getImage(),
                        VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_GENERAL,
                        VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT, VK_ACCESS_2_SHADER_READ_BIT,
                        VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR,
                        VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR | VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, 1);

    edgePipeline->bindPipeline(commandBuffer);
    edgePipeline->bindDescriptorSets(commandBuffer, currentFrame);
    edgePipeline->pushConstants(commandBuffer, &pushConstants);
    edgePipeline->traceRays(commandBuffer, {extent.width, extent.height, 1});

    auto edgeWriteToRead = [&](ImageResource &image) {
        device.imageBarrier(commandBuffer, image.getImage(),
                            VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_GENERAL,
                            VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT, VK_ACCESS_2_SHADER_READ_BIT,
                            VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, 1);
    };
    edgeWriteToRead(hEdgeLeft);
    edgeWriteToRead(hEdgeRight);
    edgeWriteToRead(vEdgeUp);
    edgeWriteToRead(vEdgeDown);

    composePipeline->bindPipeline(commandBuffer);
    composePipeline->bindDescriptorSets(commandBuffer, currentFrame);
    composePipeline->pushConstants(commandBuffer, &pushConstants);
    uint32_t groupsX = (extent.width + 15) / 16;
    uint32_t groupsY = (extent.height + 15) / 16;
    vkCmdDispatch(commandBuffer, groupsX, groupsY, 1);

    firstFrame = false;
    return getOutputSlot();
}

PassImageSlot ICVPass::getOutputSlot() const
{
    return {
        outputImage.getImage(),
        outputImage.getImageView(),
        outputImage.getSampler(),
        VK_FORMAT_R32G32B32A32_SFLOAT,
        outputImage.getExtent(),
        VK_IMAGE_LAYOUT_GENERAL,
    };
}
