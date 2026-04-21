#include "branch_pt_pass.h"

#include "camera.h"
#include "imgui.h"

REGISTER_RENDER_PASS_CPP(BranchPTPass, "branch_pt");

uint32_t BranchPTPass::calcVerticesPerPixel() const
{
    // Worst-case: every bounce hits a rough surface → full branching at each depth
    uint32_t total = 1; // depth-0 vertex
    uint32_t last = 1;
    for (int d = 0; d <= pushConstants.maxDepth; d++)
    {
        int n = getInnerSamples(d);
        if (pushConstants.useDebiasing)
        {
            // Geometric(r) extra samples: budget mean + 3*sigma upper bound
            float r = pushConstants.debiasR;
            float variance = 1.0f / r - 1.0f;
            float sigma = std::sqrt(variance);
            int extra = static_cast<int>(std::ceil((1.0f / r) * (1.0f + 3.0f * sigma)));
            n += extra;
        }
        total += last * n;
        last *= n;
        if (total > 100000000u) break; // safety cap
    }
    return total;
}

int BranchPTPass::getInnerSamples(int depth) const
{
    if (depth < 4)
        return innerSamples[depth];
    return innerSamples[3];
}

BranchPTPass::BranchPTPass(Device &_d, SwapChain &_sc, const json &params)
    : PassBase(_d, _sc),
      colorImage{_d, VK_FORMAT_R32G32B32A32_SFLOAT, _sc.getExtent(),
                 VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT},
      uniformBuffer{_d, sizeof(BrPTUniform), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT},
      model{_d}
{
    if (params.contains("maxDepth"))
        pushConstants.maxDepth = params["maxDepth"].get<int>();
    if (params.contains("useDebiasing"))
        pushConstants.useDebiasing = params["useDebiasing"].get<int>();
    if (params.contains("debiasR"))
        pushConstants.debiasR = params["debiasR"].get<float>();
    if (params.contains("vramBudgetMB"))
        vramBudgetMB = params["vramBudgetMB"].get<uint32_t>();
    if (params.contains("rrDepth"))
        pushConstants.rrDepth = params["rrDepth"].get<int>();

    // Parse inner_samples array
    if (params.contains("innerSamples"))
    {
        auto &arr = params["innerSamples"];
        for (int i = 0; i < 4 && i < (int)arr.size(); i++)
            innerSamples[i] = arr[i].get<int>();
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

    // Compute tiling parameters
    verticesPerPixel = calcVerticesPerPixel();

    uint64_t bytesPerVertex = sizeof(BrPTVertex); // 128
    if (pushConstants.useDebiasing)
        bytesPerVertex += 2 * 16; // debias direct + indirect buffers
    uint64_t budgetBytes = static_cast<uint64_t>(vramBudgetMB) * 1024ULL * 1024ULL;
    pixelsPerPass = static_cast<uint32_t>(
        std::min(budgetBytes / (verticesPerPixel * bytesPerVertex),
                 static_cast<uint64_t>(totalPixels)));
    pixelsPerPass = std::max(pixelsPerPass, 1u);

    maxVertices = verticesPerPixel * pixelsPerPass;

    // Clamp to Vulkan maxStorageBufferRange (typically ~1GB = 1073741820 bytes)
    // The largest single buffer is vertexBuffer = maxVertices * 128 bytes
    uint32_t maxStorageRange = _d.getPhysicalDeviceProperties().limits.maxStorageBufferRange;
    uint32_t maxVerticesByRange = maxStorageRange / sizeof(BrPTVertex);
    if (maxVertices > maxVerticesByRange)
    {
        maxVertices = maxVerticesByRange;
        pixelsPerPass = maxVertices / verticesPerPixel;
        pixelsPerPass = std::max(pixelsPerPass, 1u);
        maxVertices = verticesPerPixel * pixelsPerPass;
    }

    // Cap debias entries so each float4 buffer stays under maxStorageBufferRange
    uint32_t maxDebiasEntriesByRange = maxStorageRange / (sizeof(float) * 4);
    maxDebiasEntries = std::min(maxVertices, maxDebiasEntriesByRange);

    printf("[BrPT] verticesPerPixel=%u, pixelsPerPass=%u, maxVertices=%u, totalPixels=%u, tiles=%u\n",
           verticesPerPixel, pixelsPerPass, maxVertices, totalPixels,
           (totalPixels + pixelsPerPass - 1) / pixelsPerPass);

    pushConstants.maxVertices = static_cast<int>(maxVertices);

    // Allocate GPU buffers
    vertexBuffer = std::make_unique<StorageBufferResource>(
        _d, sizeof(BrPTVertex) * maxVertices,
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT);

    // counterBuffer: [vtxCounter, debiasCounter, dispatchX, Y, Z, perDepthDispatch[depth*3]...]
    uint32_t counterBufferSize = 5 + (pushConstants.maxDepth + 2) * 3;
    counterBuffer = std::make_unique<StorageBufferResource>(
        _d, sizeof(uint32_t) * counterBufferSize,
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT);

    // depthRangeBuffer: int2 per depth level (maxDepth + 1 entries)
    depthRangeBuffer = std::make_unique<StorageBufferResource>(
        _d, sizeof(DepthRange) * (pushConstants.maxDepth + 2),
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT);

    // Debias buffers
    debiasDirectBuffer = std::make_unique<StorageBufferResource>(
        _d, sizeof(float) * 4 * maxDebiasEntries,
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT);

    debiasIndirectBuffer = std::make_unique<StorageBufferResource>(
        _d, sizeof(float) * 4 * maxDebiasEntries,
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT);
}

void BranchPTPass::init()
{
    VkShaderStageFlags cs = VK_SHADER_STAGE_COMPUTE_BIT;

    // Descriptor layout:
    // 0: vertices, 1: counters, 2: depthRange, 3: colorImage, 4: uniform,
    // 5-13: model (TLAS, vertices, indices, materials, normals, texcoords, lights, instanceTransforms, meshInfo)
    // 14: debiasDirectBuffer, 15: debiasIndirectBuffer
    std::vector<DescriptorLayoutBinding> bindings = {
        {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, cs},  // 0: vertices
        {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, cs},  // 1: counters
        {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, cs},  // 2: depthRange
        {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, cs},   // 3: colorImage
        {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, cs},  // 4: uniform
    };
    auto modelBindings = model.getDescriptorBindings(VK_SHADER_STAGE_COMPUTE_BIT, true);
    bindings.insert(bindings.end(), modelBindings.begin(), modelBindings.end());
    // Debias buffers (bindings 14-15)
    bindings.push_back({VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, cs});
    bindings.push_back({VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, cs});

    auto createPipeline = [&](const std::string &spvPath) {
        return std::make_unique<ComputePipeline>(
            device, 1, bindings, spvPath, sizeof(BrPTPushConstants));
    };

    initPipeline = createPipeline(shaderPath("branch_pt/brpt_init.spv"));
    advancePipeline = createPipeline(shaderPath("branch_pt/brpt_advance.spv"));
    propagatePipeline = createPipeline(shaderPath("branch_pt/brpt_propagate.spv"));
    accumulatePipeline = createPipeline(shaderPath("branch_pt/brpt_accumulate.spv"));
    prepareIndirectPipeline = createPipeline(shaderPath("branch_pt/brpt_prepare_indirect.spv"));

    // Build descriptor infos
    std::vector<std::vector<DescriptorInfo>> infos = {
        {VkDescriptorBufferInfo{vertexBuffer->getBuffer(), 0, VK_WHOLE_SIZE}},
        {VkDescriptorBufferInfo{counterBuffer->getBuffer(), 0, VK_WHOLE_SIZE}},
        {VkDescriptorBufferInfo{depthRangeBuffer->getBuffer(), 0, VK_WHOLE_SIZE}},
        {VkDescriptorImageInfo{colorImage.getSampler(), colorImage.getImageView(), VK_IMAGE_LAYOUT_GENERAL}},
        {VkDescriptorBufferInfo{uniformBuffer.getBuffer(), 0, sizeof(BrPTUniform)}},
    };
    auto modelInfos = model.getDescriptorInfos(true);
    infos.insert(infos.end(), modelInfos.begin(), modelInfos.end());
    // Debias buffers
    infos.push_back({VkDescriptorBufferInfo{debiasDirectBuffer->getBuffer(), 0, VK_WHOLE_SIZE}});
    infos.push_back({VkDescriptorBufferInfo{debiasIndirectBuffer->getBuffer(), 0, VK_WHOLE_SIZE}});

    // Update all pipelines
    initPipeline->updateDescriptorSets(infos);
    advancePipeline->updateDescriptorSets(infos);
    propagatePipeline->updateDescriptorSets(infos);
    accumulatePipeline->updateDescriptorSets(infos);
    prepareIndirectPipeline->updateDescriptorSets(infos);
}

void BranchPTPass::drawUI()
{
    ImGui::Text("Max Depth: %d", pushConstants.maxDepth);
    ImGui::SliderInt("RR Depth", &pushConstants.rrDepth, 1, pushConstants.maxDepth);
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
    ImGui::Text("Branch PT (Tiling)");
    ImGui::Text("Vertices/Pixel: %d", verticesPerPixel);
    ImGui::Text("Pixels/Pass: %d", pixelsPerPass);
    ImGui::Text("Max Vertices: %d", pushConstants.maxVertices);
    ImGui::Text("Inner Samples: [%d, %d, %d, %d]",
                pushConstants.innerSamples0, pushConstants.innerSamples1,
                pushConstants.innerSamples2, pushConstants.innerSamples3);
    const char* debiasNames[] = {"Off", "Jackknife", "Telescoping"};
    int debiasMode = pushConstants.useDebiasing;
    if (debiasMode < 0 || debiasMode > 2) debiasMode = 0;
    ImGui::Text("Debiasing: %s (R=%.2f)", debiasNames[debiasMode], pushConstants.debiasR);
}

void BranchPTPass::update(uint32_t currentFrame, InputState &inputState)
{
    static int lastNEE = pushConstants.useNEE;
    static int lastMIS = pushConstants.useMIS;
    static int lastRRDepth = pushConstants.rrDepth;

    if (pushConstants.useNEE != lastNEE || pushConstants.useMIS != lastMIS ||
        pushConstants.rrDepth != lastRRDepth)
    {
        inputState.keyboardChanged = true;
        lastNEE = pushConstants.useNEE;
        lastMIS = pushConstants.useMIS;
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

void BranchPTPass::computeBarrier(VkCommandBuffer cmd)
{
    device.memoryBarrier(cmd,
                         VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT | VK_ACCESS_2_SHADER_WRITE_BIT,
                         VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_SHADER_READ_BIT |
                         VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT | VK_ACCESS_2_INDIRECT_COMMAND_READ_BIT,
                         VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                         VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_2_DRAW_INDIRECT_BIT);
}

PassImageSlot BranchPTPass::recordCommand(VkCommandBuffer commandBuffer,
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

    firstFrame = false;

    // ================================================================
    // TILING LOOP: process screen in tiles of pixelsPerPass pixels
    // ================================================================
    for (uint32_t tileStart = 0; tileStart < totalPixels; tileStart += pixelsPerPass)
    {
        uint32_t tileCount = std::min(pixelsPerPass, totalPixels - tileStart);

        pushConstants.tilePixelOffset = static_cast<int>(tileStart);
        pushConstants.tilePixelCount = static_cast<int>(tileCount);



        // --- Clear counters and depth ranges for this tile ---
        vkCmdFillBuffer(commandBuffer, counterBuffer->getBuffer(), 0, VK_WHOLE_SIZE, 0);
        vkCmdFillBuffer(commandBuffer, depthRangeBuffer->getBuffer(), 0,
                        sizeof(DepthRange) * (pushConstants.maxDepth + 2), 0);
        device.memoryBarrier(commandBuffer,
                             VK_ACCESS_2_TRANSFER_WRITE_BIT,
                             VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                             VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                             VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT);

        // ============================================================
        // PHASE 1: Init — create depth-0 vertices + trace primary rays
        // ============================================================
        pushConstants.currentDepth = 0;

        initPipeline->bindPipeline(commandBuffer);
        initPipeline->bindDescriptorSets(commandBuffer, currentFrame);
        initPipeline->pushConstants(commandBuffer, &pushConstants);
        vkCmdDispatch(commandBuffer, (tileCount + 255) / 256, 1, 1);

        computeBarrier(commandBuffer);

        // PrepareIndirect for depth 0: record depthRanges[0] + set indirect args
        prepareIndirectPipeline->bindPipeline(commandBuffer);
        prepareIndirectPipeline->bindDescriptorSets(commandBuffer, currentFrame);
        prepareIndirectPipeline->pushConstants(commandBuffer, &pushConstants);
        vkCmdDispatch(commandBuffer, 1, 1, 1);

        computeBarrier(commandBuffer);

        // ============================================================
        // PHASE 2: Forward pass — per depth: advance (megakernel) → prepareIndirect
        // ============================================================
        for (int depth = 0; depth <= pushConstants.maxDepth; depth++)
        {
            pushConstants.currentDepth = depth;

            // Advance: indirect dispatch from counters[2,3,4]
            advancePipeline->bindPipeline(commandBuffer);
            advancePipeline->bindDescriptorSets(commandBuffer, currentFrame);
            advancePipeline->pushConstants(commandBuffer, &pushConstants);
            vkCmdDispatchIndirect(commandBuffer, counterBuffer->getBuffer(), 2 * sizeof(uint32_t));

            computeBarrier(commandBuffer);

            // PrepareIndirect for depth+1: record depthRanges[depth+1] + set indirect args
            pushConstants.currentDepth = depth + 1;
            prepareIndirectPipeline->bindPipeline(commandBuffer);
            prepareIndirectPipeline->bindDescriptorSets(commandBuffer, currentFrame);
            prepareIndirectPipeline->pushConstants(commandBuffer, &pushConstants);
            vkCmdDispatch(commandBuffer, 1, 1, 1);

            computeBarrier(commandBuffer);
        }

        // ============================================================
        // PHASE 3: Backward pass — propagate from deepest to shallowest
        // Uses per-depth indirect args stored by forward prepareIndirect
        // ============================================================
        for (int depth = pushConstants.maxDepth - 1; depth >= 0; depth--)
        {
            pushConstants.currentDepth = depth;

            propagatePipeline->bindPipeline(commandBuffer);
            propagatePipeline->bindDescriptorSets(commandBuffer, currentFrame);
            propagatePipeline->pushConstants(commandBuffer, &pushConstants);
            VkDeviceSize indirectOffset = (5 + depth * 3) * sizeof(uint32_t);
            vkCmdDispatchIndirect(commandBuffer, counterBuffer->getBuffer(), indirectOffset);

            computeBarrier(commandBuffer);
        }

        // ============================================================
        // PHASE 4: Accumulate — depth-0 vertex radiance → colorImage
        // ============================================================
        pushConstants.currentDepth = 0;
        accumulatePipeline->bindPipeline(commandBuffer);
        accumulatePipeline->bindDescriptorSets(commandBuffer, currentFrame);
        accumulatePipeline->pushConstants(commandBuffer, &pushConstants);
        vkCmdDispatch(commandBuffer, (tileCount + 255) / 256, 1, 1);

        // Barrier between tiles
        computeBarrier(commandBuffer);
    }

    return getOutputSlot();
}

PassImageSlot BranchPTPass::getOutputSlot() const
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
