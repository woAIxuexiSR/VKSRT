#include "branch_mc_pass.h"

#include "camera.h"
#include "gbuffer.h"
#include "imgui.h"

REGISTER_RENDER_PASS_CPP(BranchMCPass, "branch_mc");

uint32_t BranchMCPass::calcMaxVertices() const
{
    // Per-vertex memory: vertex(128) + ray(32) + hit(32) + shadow(48) = 240 bytes
    // Plus debias buffers: 2 * 16 = 32 bytes per entry
    // Total ~272 bytes per vertex
    const uint64_t bytesPerVertex = 272;
    uint64_t budgetBytes = static_cast<uint64_t>(vramBudgetMB) * 1024 * 1024;
    uint32_t budgetCap = static_cast<uint32_t>(std::min(budgetBytes / bytesPerVertex, (uint64_t)200000000));

    // Theoretical total = pixels * (1 + N0 + N0*N1 + N0*N1*N2 + ...)
    // With debiasing, each branching vertex may produce extra Geometric(r) children
    uint64_t total = totalPixels;
    uint64_t product = totalPixels;
    for (int d = 0; d < pushConstants.maxDepth; d++)
    {
        int n = getInnerSamples(d);
        if (pushConstants.useDebiasing)
            n += static_cast<int>(std::ceil(3.0f / pushConstants.debiasR));
        product *= n;
        total += product;
        if (total > budgetCap) break;
    }
    uint32_t result = static_cast<uint32_t>(std::min(total, (uint64_t)budgetCap));
    printf("[BMC] maxVertices=%u (theoretical=%llu, budgetCap=%u, pixels=%u)\n",
           result, (unsigned long long)total, budgetCap, totalPixels);
    return result;
}

int BranchMCPass::getInnerSamples(int depth) const
{
    if (depth < 4)
        return innerSamples[depth];
    return innerSamples[3]; // reuse last value
}

BranchMCPass::BranchMCPass(Device &_d, SwapChain &_sc, const json &params)
    : PassBase(_d, _sc),
      colorImage{_d, VK_FORMAT_R32G32B32A32_SFLOAT, _sc.getExtent(),
                 VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT},
      uniformBuffer{_d, sizeof(BMCUniform), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT},
      model{_d}
{
    if (params.contains("maxDepth"))
        pushConstants.maxDepth = params["maxDepth"].get<int>();
    if (params.contains("rrDepth"))
        pushConstants.rrDepth = params["rrDepth"].get<int>();
    if (params.contains("useDebiasing"))
        pushConstants.useDebiasing = params["useDebiasing"].get<int>();
    if (params.contains("debiasR"))
        pushConstants.debiasR = params["debiasR"].get<float>();
    if (params.contains("vramBudgetMB"))
        vramBudgetMB = params["vramBudgetMB"].get<uint32_t>();

    // Parse inner_samples array
    if (params.contains("innerSamples"))
    {
        auto &arr = params["innerSamples"];
        for (int i = 0; i < 4 && i < (int)arr.size(); i++)
            innerSamples[i] = arr[i].get<int>();
        // If fewer than 4 entries, pad with last value
        int lastIdx = std::min((int)arr.size() - 1, 3);
        for (int i = lastIdx + 1; i < 4; i++)
            innerSamples[i] = innerSamples[lastIdx];
    }
    pushConstants.innerSamples0 = innerSamples[0];
    pushConstants.innerSamples1 = innerSamples[1];
    pushConstants.innerSamples2 = innerSamples[2];
    pushConstants.innerSamples3 = innerSamples[3];

    SceneLoader::loadScene(params, model);
    model.buildAccelerationStructures();
    pushConstants.lightCount = model.getLightCount();

    auto extent = _sc.getExtent();
    pushConstants.screenWidth = extent.width;
    pushConstants.screenHeight = extent.height;
    totalPixels = extent.width * extent.height;

    maxVertices = calcMaxVertices();
    pushConstants.maxVertices = static_cast<int>(maxVertices);

    // Debias entries: only vertices with nInnerSamples > 1 need entries.
    // Upper bound is maxVertices, but cap to a fraction to save memory.
    maxDebiasEntries = std::min(maxVertices, totalPixels * 16);

    // Allocate GPU buffers
    vertexBuffer = std::make_unique<StorageBufferResource>(
        _d, sizeof(BMCVertex) * maxVertices,
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT);

    rayBuffer = std::make_unique<StorageBufferResource>(
        _d, sizeof(BMCRayData) * maxVertices,
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT);

    hitInfoBuffer = std::make_unique<StorageBufferResource>(
        _d, sizeof(BMCHitInfo) * maxVertices,
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT);

    shadowRayBuffer = std::make_unique<StorageBufferResource>(
        _d, sizeof(BMCShadowRay) * maxVertices,
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT);

    // counterBuffer: 8 uints = 32 bytes
    counterBuffer = std::make_unique<StorageBufferResource>(
        _d, sizeof(uint32_t) * 8,
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT);

    // depthRangeBuffer: 2 ints per depth level
    depthRangeBuffer = std::make_unique<StorageBufferResource>(
        _d, sizeof(DepthRange) * (pushConstants.maxDepth + 1),
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT);

    // Debias buffers: float3 (padded to 16 bytes) per entry
    debiasDirectBuffer = std::make_unique<StorageBufferResource>(
        _d, sizeof(float) * 4 * maxDebiasEntries,
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT);

    debiasIndirectBuffer = std::make_unique<StorageBufferResource>(
        _d, sizeof(float) * 4 * maxDebiasEntries,
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT);

    // Debias counter: 1 uint
    debiasCounterBuffer = std::make_unique<StorageBufferResource>(
        _d, sizeof(uint32_t),
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT);
}

void BranchMCPass::init()
{
    VkShaderStageFlags cs = VK_SHADER_STAGE_COMPUTE_BIT;

    // Descriptor layout:
    // 0: vertices, 1: rays, 2: hitInfo, 3: shadowRays, 4: counters, 5: depthRange,
    // 6: colorImage, 7: uniform,
    // 8: TLAS, 9-14: vertices/indices/materials/normals/texcoords/lights, 15: meshInfo
    // 16-18: gbuffer (normal, position, albedo)
    // 19: debiasDirectBuffer, 20: debiasIndirectBuffer, 21: debiasCounterBuffer
    std::vector<DescriptorLayoutBinding> bindings = {
        {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, cs},  // 0: vertices (BMCVertex[])
        {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, cs},  // 1: rays
        {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, cs},  // 2: hitInfo
        {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, cs},  // 3: shadowRays
        {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, cs},  // 4: counters
        {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, cs},  // 5: depthRange
        {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, cs},   // 6: colorImage
        {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, cs},  // 7: uniform
    };
    auto modelBindings = model.getDescriptorBindings(VK_SHADER_STAGE_COMPUTE_BIT, true);
    bindings.insert(bindings.end(), modelBindings.begin(), modelBindings.end());
    // G-buffer (bindings 16-18)
    bindings.push_back({VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, cs});
    bindings.push_back({VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, cs});
    bindings.push_back({VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, cs});
    // Debias buffers (bindings 19-21)
    bindings.push_back({VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, cs});  // 19: debiasDirectBuffer
    bindings.push_back({VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, cs});  // 20: debiasIndirectBuffer
    bindings.push_back({VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, cs});  // 21: debiasCounterBuffer

    auto createPipeline = [&](const std::string &spvPath) {
        return std::make_unique<ComputePipeline>(
            device, 1, bindings, spvPath, sizeof(BMCPushConstants));
    };

    initPipeline = createPipeline("../shaders/branch_mc/bmc_init.slang.spv");
    advancePipeline = createPipeline("../shaders/branch_mc/bmc_advance.slang.spv");
    extendPipeline = createPipeline("../shaders/branch_mc/bmc_extend.slang.spv");
    shadowPipeline = createPipeline("../shaders/branch_mc/bmc_shadow.slang.spv");
    propagatePipeline = createPipeline("../shaders/branch_mc/bmc_propagate.slang.spv");
    accumulatePipeline = createPipeline("../shaders/branch_mc/bmc_accumulate.slang.spv");
    prepareIndirectPipeline = createPipeline("../shaders/branch_mc/bmc_prepare_indirect.slang.spv");

    // Build descriptor infos
    std::vector<std::vector<DescriptorInfo>> infos = {
        {VkDescriptorBufferInfo{vertexBuffer->getBuffer(), 0, VK_WHOLE_SIZE}},
        {VkDescriptorBufferInfo{rayBuffer->getBuffer(), 0, VK_WHOLE_SIZE}},
        {VkDescriptorBufferInfo{hitInfoBuffer->getBuffer(), 0, VK_WHOLE_SIZE}},
        {VkDescriptorBufferInfo{shadowRayBuffer->getBuffer(), 0, VK_WHOLE_SIZE}},
        {VkDescriptorBufferInfo{counterBuffer->getBuffer(), 0, VK_WHOLE_SIZE}},
        {VkDescriptorBufferInfo{depthRangeBuffer->getBuffer(), 0, VK_WHOLE_SIZE}},
        {VkDescriptorImageInfo{colorImage.getSampler(), colorImage.getImageView(), VK_IMAGE_LAYOUT_GENERAL}},
        {VkDescriptorBufferInfo{uniformBuffer.getBuffer(), 0, sizeof(BMCUniform)}},
    };
    auto modelInfos = model.getDescriptorInfos(true);
    infos.insert(infos.end(), modelInfos.begin(), modelInfos.end());
    // G-buffer
    infos.push_back({VkDescriptorImageInfo{gbuffer->getNormalSampler(), gbuffer->getNormalImageView(), VK_IMAGE_LAYOUT_GENERAL}});
    infos.push_back({VkDescriptorImageInfo{gbuffer->getPositionSampler(), gbuffer->getPositionImageView(), VK_IMAGE_LAYOUT_GENERAL}});
    infos.push_back({VkDescriptorImageInfo{gbuffer->getAlbedoSampler(), gbuffer->getAlbedoImageView(), VK_IMAGE_LAYOUT_GENERAL}});
    // Debias buffers
    infos.push_back({VkDescriptorBufferInfo{debiasDirectBuffer->getBuffer(), 0, VK_WHOLE_SIZE}});
    infos.push_back({VkDescriptorBufferInfo{debiasIndirectBuffer->getBuffer(), 0, VK_WHOLE_SIZE}});
    infos.push_back({VkDescriptorBufferInfo{debiasCounterBuffer->getBuffer(), 0, VK_WHOLE_SIZE}});

    // Update all pipelines
    initPipeline->updateDescriptorSets(infos);
    advancePipeline->updateDescriptorSets(infos);
    extendPipeline->updateDescriptorSets(infos);
    shadowPipeline->updateDescriptorSets(infos);
    propagatePipeline->updateDescriptorSets(infos);
    accumulatePipeline->updateDescriptorSets(infos);
    prepareIndirectPipeline->updateDescriptorSets(infos);
}

void BranchMCPass::drawUI()
{
    ImGui::SliderInt("Max Depth", &pushConstants.maxDepth, 1, 16);
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

    ImGui::Separator();
    ImGui::Text("Branch MC (Tree)");
    ImGui::Text("Max Vertices: %d", pushConstants.maxVertices);
    ImGui::SliderInt("Inner N0", &pushConstants.innerSamples0, 1, 8);
    ImGui::SliderInt("Inner N1", &pushConstants.innerSamples1, 1, 8);
    ImGui::SliderInt("Inner N2", &pushConstants.innerSamples2, 1, 4);
    ImGui::SliderInt("Inner N3+", &pushConstants.innerSamples3, 1, 4);

    bool debias = pushConstants.useDebiasing != 0;
    if (ImGui::Checkbox("Use Debiasing", &debias))
        pushConstants.useDebiasing = debias ? 1 : 0;
    if (pushConstants.useDebiasing)
        ImGui::SliderFloat("Debias R", &pushConstants.debiasR, 0.1f, 0.9f);
}

void BranchMCPass::update(uint32_t currentFrame, InputState &inputState)
{
    static int lastNEE = pushConstants.useNEE;
    static int lastMIS = pushConstants.useMIS;
    static int lastMaxDepth = pushConstants.maxDepth;
    static int lastRRDepth = pushConstants.rrDepth;
    static int lastN0 = pushConstants.innerSamples0;
    static int lastN1 = pushConstants.innerSamples1;
    static int lastN2 = pushConstants.innerSamples2;
    static int lastN3 = pushConstants.innerSamples3;
    static int lastDebias = pushConstants.useDebiasing;
    static float lastDebiasR = pushConstants.debiasR;

    if (pushConstants.useNEE != lastNEE || pushConstants.useMIS != lastMIS ||
        pushConstants.maxDepth != lastMaxDepth || pushConstants.rrDepth != lastRRDepth ||
        pushConstants.innerSamples0 != lastN0 || pushConstants.innerSamples1 != lastN1 ||
        pushConstants.innerSamples2 != lastN2 || pushConstants.innerSamples3 != lastN3 ||
        pushConstants.useDebiasing != lastDebias || pushConstants.debiasR != lastDebiasR)
    {
        inputState.keyboardChanged = true;
        lastNEE = pushConstants.useNEE;
        lastMIS = pushConstants.useMIS;
        lastMaxDepth = pushConstants.maxDepth;
        lastRRDepth = pushConstants.rrDepth;
        lastN0 = pushConstants.innerSamples0;
        lastN1 = pushConstants.innerSamples1;
        lastN2 = pushConstants.innerSamples2;
        lastN3 = pushConstants.innerSamples3;
        lastDebias = pushConstants.useDebiasing;
        lastDebiasR = pushConstants.debiasR;
    }

    // Sync innerSamples array
    innerSamples[0] = pushConstants.innerSamples0;
    innerSamples[1] = pushConstants.innerSamples1;
    innerSamples[2] = pushConstants.innerSamples2;
    innerSamples[3] = pushConstants.innerSamples3;

    if (!inputState.isChanged())
        pushConstants.frameIndex++;
    else
        pushConstants.frameIndex = 0;

    ubo.viewInverse = camera->getInverseViewMatrix();
    ubo.projInverse = camera->getInverseProjectionMatrix();
    uniformBuffer.update(&ubo);
}

void BranchMCPass::computeBarrier(VkCommandBuffer cmd)
{
    device.memoryBarrier(cmd,
                         VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                         VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                         VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                         VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT);
}

PassImageSlot BranchMCPass::recordCommand(VkCommandBuffer commandBuffer,
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

    // --- Clear counterBuffer + debiasCounterBuffer ---
    vkCmdFillBuffer(commandBuffer, counterBuffer->getBuffer(), 0, sizeof(uint32_t) * 8, 0);
    vkCmdFillBuffer(commandBuffer, debiasCounterBuffer->getBuffer(), 0, sizeof(uint32_t), 0);
    device.memoryBarrier(commandBuffer,
                         VK_ACCESS_2_TRANSFER_WRITE_BIT,
                         VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                         VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                         VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT);

    // ================================================================
    // PHASE 1: bmc_init — create depth-0 vertices + trace primary rays
    // ================================================================
    pushConstants.currentDepth = 0;
    pushConstants.depthVertexStart = 0;
    pushConstants.depthVertexCount = static_cast<int>(totalPixels);

    initPipeline->bindPipeline(commandBuffer);
    initPipeline->bindDescriptorSets(commandBuffer, currentFrame);
    initPipeline->pushConstants(commandBuffer, &pushConstants);
    vkCmdDispatch(commandBuffer, (extent.width + 15) / 16, (extent.height + 15) / 16, 1);

    computeBarrier(commandBuffer);

    // ================================================================
    // PHASE 2: Forward pass — per depth: advance → extend → shadow
    // ================================================================
    // After init, depth-0 has totalPixels vertices starting at index 0
    int depthStart = 0;
    int depthCount = static_cast<int>(totalPixels);

    for (int depth = 0; depth < pushConstants.maxDepth; depth++)
    {
        pushConstants.currentDepth = depth;
        pushConstants.depthVertexStart = depthStart;
        pushConstants.depthVertexCount = depthCount;

        // Clamp to maxVertices — vertices beyond the buffer don't exist
        if (depthStart + depthCount > static_cast<int>(maxVertices))
            pushConstants.depthVertexCount = std::max(0, static_cast<int>(maxVertices) - depthStart);

        if (pushConstants.depthVertexCount <= 0) break;

        // --- Clear child vertex range to prevent stale alive flags ---
        // Compute next depth's upper bound for clearing
        int clearN = getInnerSamples(depth);
        if (pushConstants.useDebiasing)
            clearN += static_cast<int>(std::ceil(3.0f / pushConstants.debiasR));
        int clearStart = depthStart + depthCount;
        int clearCount = depthCount * clearN;
        if (clearStart + clearCount > static_cast<int>(maxVertices))
            clearCount = static_cast<int>(maxVertices) - clearStart;
        if (clearCount > 0)
        {
            VkDeviceSize clearOffset = static_cast<VkDeviceSize>(clearStart) * sizeof(BMCVertex);
            VkDeviceSize clearSize = static_cast<VkDeviceSize>(clearCount) * sizeof(BMCVertex);
            vkCmdFillBuffer(commandBuffer, vertexBuffer->getBuffer(), clearOffset, clearSize, 0);
        }

        // --- Advance: process vertices at current depth, spawn children ---
        // Also resets counters[6] for new vertex count, counters[7] for shadow count
        vkCmdFillBuffer(commandBuffer, counterBuffer->getBuffer(), 0, sizeof(uint32_t) * 8, 0);
        device.memoryBarrier(commandBuffer,
                             VK_ACCESS_2_TRANSFER_WRITE_BIT,
                             VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                             VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                             VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT);

        advancePipeline->bindPipeline(commandBuffer);
        advancePipeline->bindDescriptorSets(commandBuffer, currentFrame);
        advancePipeline->pushConstants(commandBuffer, &pushConstants);
        vkCmdDispatch(commandBuffer, (depthCount + 255) / 256, 1, 1);

        computeBarrier(commandBuffer);

        // --- PrepareIndirect: set dispatch args from counters[6] and counters[7] ---
        prepareIndirectPipeline->bindPipeline(commandBuffer);
        prepareIndirectPipeline->bindDescriptorSets(commandBuffer, currentFrame);
        prepareIndirectPipeline->pushConstants(commandBuffer, &pushConstants);
        vkCmdDispatch(commandBuffer, 1, 1, 1);

        device.memoryBarrier(commandBuffer,
                             VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                             VK_ACCESS_2_INDIRECT_COMMAND_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                             VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                             VK_PIPELINE_STAGE_2_DRAW_INDIRECT_BIT | VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT);

        // --- Extend: trace rays for new child vertices ---
        extendPipeline->bindPipeline(commandBuffer);
        extendPipeline->bindDescriptorSets(commandBuffer, currentFrame);
        extendPipeline->pushConstants(commandBuffer, &pushConstants);
        vkCmdDispatchIndirect(commandBuffer, counterBuffer->getBuffer(), 0);

        computeBarrier(commandBuffer);

        // --- Shadow: trace shadow rays ---
        shadowPipeline->bindPipeline(commandBuffer);
        shadowPipeline->bindDescriptorSets(commandBuffer, currentFrame);
        shadowPipeline->pushConstants(commandBuffer, &pushConstants);
        vkCmdDispatchIndirect(commandBuffer, counterBuffer->getBuffer(), 3 * sizeof(uint32_t));

        computeBarrier(commandBuffer);

        // Move to next depth: new children become the next depth's vertices
        // CPU-side upper bound for dispatch sizing.
        // With debiasing, each vertex may produce baseSamples + Geometric(r) children.
        // Use a multiplier to account for the extra debiasing samples.
        int nextN = getInnerSamples(depth);
        if (pushConstants.useDebiasing)
        {
            // Geometric(r) expected value = 1/r. With r=0.5, E[extra] = 2.
            // Use baseSamples + ceil(3/r) as a conservative upper bound.
            int extraBound = static_cast<int>(std::ceil(3.0f / pushConstants.debiasR));
            nextN += extraBound;
        }
        int nextStart = depthStart + depthCount;
        int nextCount = depthCount * nextN; // upper bound (some may be dead)
        // Clamp to buffer bounds
        if (nextStart + nextCount > static_cast<int>(maxVertices))
            nextCount = std::max(0, static_cast<int>(maxVertices) - nextStart);
        depthStart = nextStart;
        depthCount = nextCount;
    }

    // ================================================================
    // PHASE 3: Backward pass — per depth reversed: propagate
    // ================================================================
    // Reset depth tracking for backward pass
    depthStart = 0;
    depthCount = static_cast<int>(totalPixels);

    // Build depth ranges for backward pass
    struct DepthInfo { int start; int count; };
    std::vector<DepthInfo> depthInfos;
    {
        int s = 0;
        int c = static_cast<int>(totalPixels);
        for (int d = 0; d <= pushConstants.maxDepth; d++)
        {
            depthInfos.push_back({s, c});
            if (d < pushConstants.maxDepth)
            {
                int nextN = getInnerSamples(d);
                if (pushConstants.useDebiasing)
                {
                    int extraBound = static_cast<int>(std::ceil(3.0f / pushConstants.debiasR));
                    nextN += extraBound;
                }
                s = s + c;
                c = c * nextN;
            }
        }
    }

    // Propagate from deepest depth back to 0
    for (int depth = pushConstants.maxDepth - 1; depth >= 0; depth--)
    {
        pushConstants.currentDepth = depth;
        pushConstants.depthVertexStart = depthInfos[depth].start;
        pushConstants.depthVertexCount = depthInfos[depth].count;

        // Clamp to maxVertices
        if (depthInfos[depth].start + depthInfos[depth].count > static_cast<int>(maxVertices))
            pushConstants.depthVertexCount = std::max(0, static_cast<int>(maxVertices) - depthInfos[depth].start);

        if (pushConstants.depthVertexCount <= 0) break;

        propagatePipeline->bindPipeline(commandBuffer);
        propagatePipeline->bindDescriptorSets(commandBuffer, currentFrame);
        propagatePipeline->pushConstants(commandBuffer, &pushConstants);
        vkCmdDispatch(commandBuffer, (pushConstants.depthVertexCount + 255) / 256, 1, 1);

        computeBarrier(commandBuffer);
    }

    // ================================================================
    // PHASE 4: Accumulate — depth-0 vertex radiance → colorImage
    // ================================================================
    accumulatePipeline->bindPipeline(commandBuffer);
    accumulatePipeline->bindDescriptorSets(commandBuffer, currentFrame);
    accumulatePipeline->pushConstants(commandBuffer, &pushConstants);
    vkCmdDispatch(commandBuffer, (extent.width + 15) / 16, (extent.height + 15) / 16, 1);

    gbuffer->markWritten();

    return getOutputSlot();
}

PassImageSlot BranchMCPass::getOutputSlot() const
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
