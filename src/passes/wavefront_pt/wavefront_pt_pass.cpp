#include "wavefront_pt_pass.h"

#include "camera.h"
#include "gbuffer.h"
#include "imgui.h"

REGISTER_RENDER_PASS_CPP(WavefrontPTPass, "wavefront_pt");

WavefrontPTPass::WavefrontPTPass(Device &_d, SwapChain &_sc, const json &params)
    : PassBase(_d, _sc),
      colorImage{_d, VK_FORMAT_R32G32B32A32_SFLOAT, _sc.getExtent(),
                 VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT},
      uniformBuffer{_d, sizeof(WFPTUniform), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT},
      model{_d}
{
    if (params.contains("maxDepth"))
        pushConstants.maxDepth = params["maxDepth"].get<int>();
    if (params.contains("rrDepth"))
        pushConstants.rrDepth = params["rrDepth"].get<int>();

    SceneLoader::loadScene(params, model);
    model.buildAccelerationStructures();
    pushConstants.lightCount = model.getLightCount();

    auto extent = _sc.getExtent();
    pushConstants.screenWidth = extent.width;
    pushConstants.screenHeight = extent.height;
    totalPaths = extent.width * extent.height;
    pushConstants.totalPaths = static_cast<int>(totalPaths);

    // Allocate GPU buffers
    pathStateBuffer = std::make_unique<StorageBufferResource>(
        _d, sizeof(WFPathState) * totalPaths,
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT);

    rayBuffer = std::make_unique<StorageBufferResource>(
        _d, sizeof(WFRayData) * totalPaths,
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT);

    hitInfoBuffer = std::make_unique<StorageBufferResource>(
        _d, sizeof(WFHitInfo) * totalPaths,
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT);

    shadowRayBuffer = std::make_unique<StorageBufferResource>(
        _d, sizeof(WFShadowRay) * totalPaths,
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT);

    // counterBuffer: 8 uints = 32 bytes
    // [0-2] = active indirect dispatch args (groupX, 1, 1)
    // [3-5] = shadow indirect dispatch args (groupX, 1, 1)
    // [6]   = nextActiveCount (atomic)
    // [7]   = shadowCount (atomic)
    counterBuffer = std::make_unique<StorageBufferResource>(
        _d, sizeof(uint32_t) * 8,
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT);

    // Double-buffered queue: 2 * totalPaths
    queueBuffer = std::make_unique<StorageBufferResource>(
        _d, sizeof(uint32_t) * totalPaths * 2,
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT);
}

void WavefrontPTPass::init()
{
    VkShaderStageFlags cs = VK_SHADER_STAGE_COMPUTE_BIT;

    // All 6 pipelines share the same descriptor layout:
    // 0: pathState, 1: rays, 2: hitInfo, 3: shadowRays, 4: counters, 5: queue,
    // 6: colorImage, 7: uniform,
    // 8: TLAS, 9-14: vertices/indices/materials/normals/texcoords/lights, 15: instanceTransforms, 16: meshInfo
    // 17-19: gbuffer (normal, position, albedo)
    std::vector<DescriptorLayoutBinding> bindings = {
        {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, cs},  // 0: pathState
        {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, cs},  // 1: rays
        {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, cs},  // 2: hitInfo
        {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, cs},  // 3: shadowRays
        {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, cs},  // 4: counters
        {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, cs},  // 5: queue
        {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, cs},   // 6: colorImage
        {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, cs},  // 7: uniform
    };
    auto modelBindings = model.getDescriptorBindings(VK_SHADER_STAGE_COMPUTE_BIT, true);
    bindings.insert(bindings.end(), modelBindings.begin(), modelBindings.end());
    // G-buffer (bindings 17-19)
    bindings.push_back({VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, cs});
    bindings.push_back({VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, cs});
    bindings.push_back({VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, cs});

    auto createPipeline = [&](const std::string &spvPath) {
        return std::make_unique<ComputePipeline>(
            device, 1, bindings, spvPath, sizeof(WFPTPushConstants));
    };

    generatePipeline = createPipeline("build/shaders/wavefront_pt/wf_generate.slang.spv");
    prepareIndirectPipeline = createPipeline("build/shaders/wavefront_pt/wf_prepare_indirect.slang.spv");
    extendPipeline = createPipeline("build/shaders/wavefront_pt/wf_extend.slang.spv");
    shadePipeline = createPipeline("build/shaders/wavefront_pt/wf_shade.slang.spv");
    shadowPipeline = createPipeline("build/shaders/wavefront_pt/wf_shadow.slang.spv");
    accumulatePipeline = createPipeline("build/shaders/wavefront_pt/wf_accumulate.slang.spv");

    // Build descriptor infos
    std::vector<std::vector<DescriptorInfo>> infos = {
        {VkDescriptorBufferInfo{pathStateBuffer->getBuffer(), 0, VK_WHOLE_SIZE}},
        {VkDescriptorBufferInfo{rayBuffer->getBuffer(), 0, VK_WHOLE_SIZE}},
        {VkDescriptorBufferInfo{hitInfoBuffer->getBuffer(), 0, VK_WHOLE_SIZE}},
        {VkDescriptorBufferInfo{shadowRayBuffer->getBuffer(), 0, VK_WHOLE_SIZE}},
        {VkDescriptorBufferInfo{counterBuffer->getBuffer(), 0, VK_WHOLE_SIZE}},
        {VkDescriptorBufferInfo{queueBuffer->getBuffer(), 0, VK_WHOLE_SIZE}},
        {VkDescriptorImageInfo{colorImage.getSampler(), colorImage.getImageView(), VK_IMAGE_LAYOUT_GENERAL}},
        {VkDescriptorBufferInfo{uniformBuffer.getBuffer(), 0, sizeof(WFPTUniform)}},
    };
    auto modelInfos = model.getDescriptorInfos(true);
    infos.insert(infos.end(), modelInfos.begin(), modelInfos.end());
    // G-buffer
    infos.push_back({VkDescriptorImageInfo{gbuffer->getNormalSampler(), gbuffer->getNormalImageView(), VK_IMAGE_LAYOUT_GENERAL}});
    infos.push_back({VkDescriptorImageInfo{gbuffer->getPositionSampler(), gbuffer->getPositionImageView(), VK_IMAGE_LAYOUT_GENERAL}});
    infos.push_back({VkDescriptorImageInfo{gbuffer->getAlbedoSampler(), gbuffer->getAlbedoImageView(), VK_IMAGE_LAYOUT_GENERAL}});

    // Update all 6 pipelines with the same descriptor infos
    generatePipeline->updateDescriptorSets(infos);
    prepareIndirectPipeline->updateDescriptorSets(infos);
    extendPipeline->updateDescriptorSets(infos);
    shadePipeline->updateDescriptorSets(infos);
    shadowPipeline->updateDescriptorSets(infos);
    accumulatePipeline->updateDescriptorSets(infos);
}

void WavefrontPTPass::drawUI()
{
    ImGui::SliderInt("Max Depth", &pushConstants.maxDepth, 1, 32);
    ImGui::SliderInt("RR Depth", &pushConstants.rrDepth, 1, 16);
    ImGui::Text("Frame Index: %d", pushConstants.frameIndex);
    ImGui::Text("Total Paths: %d", pushConstants.totalPaths);

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

void WavefrontPTPass::update(uint32_t currentFrame, InputState &inputState)
{
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

void WavefrontPTPass::computeBarrier(VkCommandBuffer cmd)
{
    device.memoryBarrier(cmd,
                         VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT | VK_ACCESS_2_SHADER_WRITE_BIT,
                         VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_SHADER_READ_BIT |
                         VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                         VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                         VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT);
}

PassImageSlot WavefrontPTPass::recordCommand(VkCommandBuffer commandBuffer,
                                              const PassImageSlot &inputSlot,
                                              uint32_t currentFrame, uint32_t imageIndex)
{
    auto extent = colorImage.getExtent();

    // --- Image barriers ---
    VkImageLayout oldLayout = firstFrame ? VK_IMAGE_LAYOUT_UNDEFINED : VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    VkAccessFlags2 srcAccess = firstFrame ? (VkAccessFlags2)0 : VK_ACCESS_2_SHADER_READ_BIT;
    VkPipelineStageFlags2 srcStage = firstFrame ? VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT : VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;

    device.imageBarrier(commandBuffer, colorImage.getImage(),
                        oldLayout, VK_IMAGE_LAYOUT_GENERAL,
                        srcAccess, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                        srcStage, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, 1);

    // G-buffer barriers
    VkImageLayout gbOldLayout = firstFrame ? VK_IMAGE_LAYOUT_UNDEFINED : VK_IMAGE_LAYOUT_GENERAL;
    VkAccessFlags2 gbSrcAccess = firstFrame ? (VkAccessFlags2)0 : (VK_ACCESS_2_SHADER_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT);
    VkPipelineStageFlags2 gbSrcStage = firstFrame ? VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT : VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;

    device.imageBarrier(commandBuffer, gbuffer->getNormalImage(),
                        gbOldLayout, VK_IMAGE_LAYOUT_GENERAL,
                        gbSrcAccess, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                        gbSrcStage, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, 1);
    device.imageBarrier(commandBuffer, gbuffer->getPositionImage(),
                        gbOldLayout, VK_IMAGE_LAYOUT_GENERAL,
                        gbSrcAccess, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                        gbSrcStage, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, 1);
    device.imageBarrier(commandBuffer, gbuffer->getAlbedoImage(),
                        gbOldLayout, VK_IMAGE_LAYOUT_GENERAL,
                        gbSrcAccess, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                        gbSrcStage, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, 1);

    firstFrame = false;

    // --- Clear counterBuffer to 0 ---
    vkCmdFillBuffer(commandBuffer, counterBuffer->getBuffer(), 0, sizeof(uint32_t) * 8, 0);
    // Barrier: transfer write -> compute read/write
    device.memoryBarrier(commandBuffer,
                         VK_ACCESS_2_TRANSFER_WRITE_BIT,
                         VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                         VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                         VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT);

    // --- Generate: 2D dispatch, writes pathState + rayBuffer + queueA + counterBuffer ---
    pushConstants.bounceIndex = 0;
    pushConstants.queueOffset = 0; // read from first half (written by generate)
    generatePipeline->bindPipeline(commandBuffer);
    generatePipeline->bindDescriptorSets(commandBuffer, currentFrame);
    generatePipeline->pushConstants(commandBuffer, &pushConstants);
    vkCmdDispatch(commandBuffer, (extent.width + 15) / 16, (extent.height + 15) / 16, 1);

    computeBarrier(commandBuffer);

    // --- PrepareIndirect: compute dispatch args from Generate's activeCount ---
    prepareIndirectPipeline->bindPipeline(commandBuffer);
    prepareIndirectPipeline->bindDescriptorSets(commandBuffer, currentFrame);
    prepareIndirectPipeline->pushConstants(commandBuffer, &pushConstants);
    vkCmdDispatch(commandBuffer, 1, 1, 1);

    // Barrier: compute write -> indirect read + compute read
    device.memoryBarrier(commandBuffer,
                         VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                         VK_ACCESS_2_INDIRECT_COMMAND_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                         VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                         VK_PIPELINE_STAGE_2_DRAW_INDIRECT_BIT | VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT);

    // --- Bounce loop ---
    for (int bounce = 0; bounce < pushConstants.maxDepth; bounce++)
    {
        pushConstants.bounceIndex = bounce;
        // Alternate queue halves: even bounces read from first half, odd from second
        pushConstants.queueOffset = (bounce % 2 == 0) ? 0 : pushConstants.totalPaths;

        // Extend: indirect dispatch from counterBuffer[0..2]
        extendPipeline->bindPipeline(commandBuffer);
        extendPipeline->bindDescriptorSets(commandBuffer, currentFrame);
        extendPipeline->pushConstants(commandBuffer, &pushConstants);
        vkCmdDispatchIndirect(commandBuffer, counterBuffer->getBuffer(), 0);

        computeBarrier(commandBuffer);

        // Shade: indirect dispatch from counterBuffer[0..2]
        // Writes shadow rays + counters[7], surviving paths + counters[6]
        shadePipeline->bindPipeline(commandBuffer);
        shadePipeline->bindDescriptorSets(commandBuffer, currentFrame);
        shadePipeline->pushConstants(commandBuffer, &pushConstants);
        vkCmdDispatchIndirect(commandBuffer, counterBuffer->getBuffer(), 0);

        computeBarrier(commandBuffer);

        // PrepareIndirect: convert counters[6]->active dispatch, counters[7]->shadow dispatch
        // Must run AFTER Shade so shadow dispatch args reflect THIS bounce's shadow rays
        pushConstants.bounceIndex = bounce + 1;
        prepareIndirectPipeline->bindPipeline(commandBuffer);
        prepareIndirectPipeline->bindDescriptorSets(commandBuffer, currentFrame);
        prepareIndirectPipeline->pushConstants(commandBuffer, &pushConstants);
        vkCmdDispatch(commandBuffer, 1, 1, 1);

        // Barrier: PrepareIndirect writes -> Shadow indirect read + compute read
        device.memoryBarrier(commandBuffer,
                             VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                             VK_ACCESS_2_INDIRECT_COMMAND_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                             VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                             VK_PIPELINE_STAGE_2_DRAW_INDIRECT_BIT | VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT);

        // Shadow: indirect dispatch from counterBuffer[3..5] (12 bytes offset)
        shadowPipeline->bindPipeline(commandBuffer);
        shadowPipeline->bindDescriptorSets(commandBuffer, currentFrame);
        shadowPipeline->pushConstants(commandBuffer, &pushConstants);
        vkCmdDispatchIndirect(commandBuffer, counterBuffer->getBuffer(), 3 * sizeof(uint32_t));

        // Barrier for next iteration: shadow writes + PrepareIndirect writes -> next Extend indirect read
        device.memoryBarrier(commandBuffer,
                             VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                             VK_ACCESS_2_INDIRECT_COMMAND_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                             VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                             VK_PIPELINE_STAGE_2_DRAW_INDIRECT_BIT | VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT);
    }

    // --- Accumulate: 2D dispatch, read pathState.L -> write colorImage ---
    accumulatePipeline->bindPipeline(commandBuffer);
    accumulatePipeline->bindDescriptorSets(commandBuffer, currentFrame);
    accumulatePipeline->pushConstants(commandBuffer, &pushConstants);
    vkCmdDispatch(commandBuffer, (extent.width + 15) / 16, (extent.height + 15) / 16, 1);

    gbuffer->markWritten();

    return getOutputSlot();
}

PassImageSlot WavefrontPTPass::getOutputSlot() const
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
