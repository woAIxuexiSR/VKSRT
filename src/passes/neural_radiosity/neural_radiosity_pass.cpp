#include "neural_radiosity_pass.h"

#include "camera.h"
#include "imgui.h"

#include <cassert>
#include <algorithm>
#include <cstdio>

REGISTER_RENDER_PASS_CPP(NeuralRadiosityPass, "neural_radiosity");

NeuralRadiosityPass::NeuralRadiosityPass(Device &_d, SwapChain &_sc, const json &params)
    : PassBase(_d, _sc),
      colorImage{_d, VK_FORMAT_R32G32B32A32_SFLOAT, _sc.getExtent(),
                 VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT |
                     VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT},
      uniformBuffer{_d, sizeof(NRUniform), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT}
{
    if (params.contains("batchSize"))
        batchSize = params["batchSize"].get<uint32_t>();
    if (params.contains("rhsSamples"))
        rhsSamples = std::max(1u, params["rhsSamples"].get<uint32_t>());

    auto extent = colorImage.getExtent();
    pixelCount = extent.width * extent.height;

    json netJson = params.contains("network") ? params["network"] : json::object();
    network = std::make_unique<NeuralNetwork>(device, netJson);
    network->initWeights(42);

    totalRawInputDim = network->getTotalRawInputDim();
    int mlpOutputSize = network->getOutputSize();
    assert(totalRawInputDim == 12 && "Neural Radiosity expects 12 raw input dims (pos+normal+dir+albedo)");
    assert(mlpOutputSize == 3 && "Neural Radiosity output must be 3 (RGB radiance)");

    // Buffers are allocated in init() once offlineMode is known — offline only needs
    // training buffers, online only needs inference buffers.
}

void NeuralRadiosityPass::init()
{
    VkBufferUsageFlags bdaFlags = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                                  VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
    const uint32_t rhsTotal = batchSize * rhsSamples;

    if (offlineMode)
    {
        // --- Training-only buffers ---
        lhsInputBuffer = std::make_unique<StorageBufferResource>(
            device, (VkDeviceSize)batchSize * totalRawInputDim * sizeof(float), bdaFlags);
        rhsInputBuffer = std::make_unique<StorageBufferResource>(
            device, (VkDeviceSize)rhsTotal * totalRawInputDim * sizeof(float), bdaFlags);
        rhsPredBuffer = std::make_unique<StorageBufferResource>(
            device, (VkDeviceSize)rhsTotal * 3 * sizeof(float), bdaFlags);
        auxBuffer = std::make_unique<StorageBufferResource>(
            device, (VkDeviceSize)rhsTotal * 12 * sizeof(float), bdaFlags);
        gtBuffer = std::make_unique<StorageBufferResource>(
            device, (VkDeviceSize)batchSize * 3 * sizeof(float), bdaFlags);

        lhsInputAddr = device.getBufferDeviceAddress(lhsInputBuffer->getBuffer());
        rhsInputAddr = device.getBufferDeviceAddress(rhsInputBuffer->getBuffer());
        rhsPredAddr  = device.getBufferDeviceAddress(rhsPredBuffer->getBuffer());
        auxAddr      = device.getBufferDeviceAddress(auxBuffer->getBuffer());
        gtAddr       = device.getBufferDeviceAddress(gtBuffer->getBuffer());

        // sampleGen: scene descriptors + surface sampler
        auto sgBindings = scene->getDescriptorBindings(VK_SHADER_STAGE_COMPUTE_BIT,
                                                       /*includeMeshInfo=*/true,
                                                       /*includeSurfaceSampler=*/true);
        sampleGenPipeline = std::make_unique<ComputePipeline>(
            device, 1, sgBindings,
            shaderPath("neural_radiosity/nr_sample_gen.spv"),
            sizeof(NRSampleGenPC));
        sampleGenPipeline->updateDescriptorSets(
            scene->getDescriptorInfos(/*includeMeshInfo=*/true, /*includeSurfaceSampler=*/true));

        // buildTarget: pure BDA. Dummy UBO binding to satisfy Pipeline's non-empty layout.
        buildTargetPipeline = std::make_unique<ComputePipeline>(
            device, 1,
            std::vector<DescriptorLayoutBinding>{
                {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT},
            },
            shaderPath("neural_radiosity/nr_build_target.spv"),
            sizeof(NRBuildTargetPC));
        buildTargetPipeline->updateDescriptorSets({
            {VkDescriptorBufferInfo{uniformBuffer.getBuffer(), 0, sizeof(NRUniform)}},
        });
    }
    else
    {
        // --- Inference-only buffers ---
        queryInputBuffer = std::make_unique<StorageBufferResource>(
            device, (VkDeviceSize)pixelCount * totalRawInputDim * sizeof(float), bdaFlags);
        queryOutputBuffer = std::make_unique<StorageBufferResource>(
            device, (VkDeviceSize)pixelCount * 3 * sizeof(float), bdaFlags);
        // primaryInfoBuffer: 2 vec4 per pixel.
        //   [0] = info:        rgb = direct color (emission / miss); a = validNet flag (0 or 1)
        //   [1] = throughput:  accumulated delta-bounce throughput before y;
        //                      when info.a==1, final color = throughput * network(queryInput)
        primaryInfoBuffer = std::make_unique<StorageBufferResource>(
            device, (VkDeviceSize)pixelCount * 8 * sizeof(float), bdaFlags);

        queryInputAddr   = device.getBufferDeviceAddress(queryInputBuffer->getBuffer());
        queryOutputAddr  = device.getBufferDeviceAddress(queryOutputBuffer->getBuffer());
        primaryInfoAddr  = device.getBufferDeviceAddress(primaryInfoBuffer->getBuffer());

        // queryGen: camera UBO + scene descriptors (no surface sampler needed for inference).
        std::vector<DescriptorLayoutBinding> qgBindings = {
            {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT},
        };
        auto sceneQgBindings = scene->getDescriptorBindings(VK_SHADER_STAGE_COMPUTE_BIT,
                                                             /*includeMeshInfo=*/true);
        qgBindings.insert(qgBindings.end(), sceneQgBindings.begin(), sceneQgBindings.end());
        queryGenPipeline = std::make_unique<ComputePipeline>(
            device, 1, qgBindings,
            shaderPath("neural_radiosity/nr_query_gen.spv"),
            sizeof(NRQueryGenPC));

        std::vector<std::vector<DescriptorInfo>> qgInfos = {
            {VkDescriptorBufferInfo{uniformBuffer.getBuffer(), 0, sizeof(NRUniform)}},
        };
        auto sceneQgInfos = scene->getDescriptorInfos(/*includeMeshInfo=*/true);
        qgInfos.insert(qgInfos.end(), sceneQgInfos.begin(), sceneQgInfos.end());
        queryGenPipeline->updateDescriptorSets(qgInfos);
    }

    // composite: always needed. Offline just writes black; online reads net+throughput.
    compositePipeline = std::make_unique<ComputePipeline>(
        device, 1,
        std::vector<DescriptorLayoutBinding>{
            {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, VK_SHADER_STAGE_COMPUTE_BIT},
        },
        shaderPath("neural_radiosity/nr_composite.spv"),
        sizeof(NRCompositePC));
    compositePipeline->updateDescriptorSets({
        {VkDescriptorImageInfo{colorImage.getSampler(), colorImage.getImageView(), VK_IMAGE_LAYOUT_GENERAL}},
    });

    network->createPipelines();

    printf("[NeuralRadiosity] mode=%s surfaces: %d triangles, total area %.3f\n",
           offlineMode ? "OFFLINE(train)" : "ONLINE(infer)",
           scene->getSurfaceTriangleCount(), scene->getTotalSurfaceArea());
}

void NeuralRadiosityPass::drawUI()
{
    auto extent = colorImage.getExtent();
    ImGui::Text("Resolution: %ux%u", extent.width, extent.height);
    ImGui::Text("Params: %d", network->getTotalParams());
    if (network->isEMAEnabled())
        ImGui::Text("EMA: enabled (shadow weights used for inference)");
    ImGui::Text("Surface Tris: %d  Area: %.3f",
                scene->getSurfaceTriangleCount(), scene->getTotalSurfaceArea());
}

void NeuralRadiosityPass::update(uint32_t /*currentFrame*/, InputState & /*inputState*/)
{
    // Camera UBO is only read by nr_query_gen (online inference).
    if (!offlineMode)
    {
        ubo.viewInverse = camera->getInverseViewMatrix();
        ubo.projInverse = camera->getInverseProjectionMatrix();
        uniformBuffer.update(&ubo);
    }

    if (offlineMode)
    {
        currentLoss = network->readLoss();
        lossAccum += currentLoss;
        lossCount++;
    }
    frameIndex++;

    constexpr uint32_t kLossReportInterval = 200;
    if (offlineMode && lossCount >= kLossReportInterval)
    {
        float avg = float(lossAccum / lossCount);
        printf("  [NeuralRadiosity] frame %u  avg loss over last %u frames: %.6f\n",
               frameIndex, lossCount, avg);
        lossAccum = 0.0;
        lossCount = 0;
    }
}

PassImageSlot NeuralRadiosityPass::recordCommand(VkCommandBuffer cmd,
                                                 const PassImageSlot & /*inputSlot*/,
                                                 uint32_t currentFrame, uint32_t /*imageIndex*/)
{
    auto extent = colorImage.getExtent();

    const uint32_t rhsTotal = batchSize * rhsSamples;
    // Only one of the two paths is active in each mode.
    uint32_t maxSamples = offlineMode ? rhsTotal : pixelCount;
    network->ensureBuffers(maxSamples);

    // --- Offline: train only. Online: skip training entirely. ---
    if (offlineMode)
    {
        NRSampleGenPC spc{};
        spc.lhsInputAddr = lhsInputAddr;
        spc.rhsInputAddr = rhsInputAddr;
        spc.auxAddr = auxAddr;
        spc.batchSize = batchSize;
        spc.rhsSamples = rhsSamples;
        spc.frameIndex = frameIndex;
        spc.surfaceTriCount = (uint32_t)scene->getSurfaceTriangleCount();
        spc.lightCount = scene->getLightCount();
        spc.totalSurfaceArea = scene->getTotalSurfaceArea();

        sampleGenPipeline->bindPipeline(cmd);
        sampleGenPipeline->bindDescriptorSets(cmd, currentFrame);
        sampleGenPipeline->pushConstants(cmd, &spc);
        // 2D dispatch: x = batch (16/group), y = RHS sub-sample (16/group)
        vkCmdDispatch(cmd, (batchSize + 15) / 16, (rhsSamples + 15) / 16, 1);

        device.memoryBarrier(cmd,
                             VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                             VK_ACCESS_2_SHADER_STORAGE_READ_BIT,
                             VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                             VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT);

        // EMA forward at all B*S RHS features at once (uses shadow weights when useEMA=true)
        network->recordForward(cmd, rhsInputBuffer->getBuffer(),
                               rhsPredBuffer->getBuffer(), rhsTotal);

        device.memoryBarrier(cmd,
                             VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                             VK_ACCESS_2_SHADER_STORAGE_READ_BIT,
                             VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                             VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT);

        NRBuildTargetPC bpc{};
        bpc.auxAddr = auxAddr;
        bpc.rhsPredAddr = rhsPredAddr;
        bpc.gtAddr = gtAddr;
        bpc.batchSize = batchSize;
        bpc.rhsSamples = rhsSamples;

        buildTargetPipeline->bindPipeline(cmd);
        buildTargetPipeline->bindDescriptorSets(cmd, currentFrame);
        buildTargetPipeline->pushConstants(cmd, &bpc);
        vkCmdDispatch(cmd, (batchSize + 255) / 256, 1, 1);

        device.memoryBarrier(cmd,
                             VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                             VK_ACCESS_2_SHADER_STORAGE_READ_BIT,
                             VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                             VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT);

        // Training step: forward/backward/adam/EMA on LHS (RHS is frozen in gtBuffer).
        network->recordTrain(cmd, lhsInputBuffer->getBuffer(),
                             gtBuffer->getBuffer(), batchSize);

        device.memoryBarrier(cmd,
                             VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                             VK_ACCESS_2_SHADER_STORAGE_READ_BIT,
                             VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                             VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT);
    }

    // --- Color image barrier: UNDEFINED -> GENERAL (pure-write every frame) ---
    device.imageBarrier(cmd, colorImage.getImage(),
                        VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL,
                        0, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                        VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT,
                        VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, 1);

    // --- Inference phase (skipped in offline mode) ---
    if (!offlineMode)
    {
        NRQueryGenPC qpc{};
        qpc.queryInputAddr = queryInputAddr;
        qpc.primaryInfoAddr = primaryInfoAddr;
        qpc.width = extent.width;
        qpc.height = extent.height;
        qpc.frameIndex = frameIndex;

        queryGenPipeline->bindPipeline(cmd);
        queryGenPipeline->bindDescriptorSets(cmd, currentFrame);
        queryGenPipeline->pushConstants(cmd, &qpc);
        vkCmdDispatch(cmd, (extent.width + 15) / 16, (extent.height + 15) / 16, 1);

        device.memoryBarrier(cmd,
                             VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                             VK_ACCESS_2_SHADER_STORAGE_READ_BIT,
                             VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                             VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT);

        network->recordForward(cmd, queryInputBuffer->getBuffer(),
                               queryOutputBuffer->getBuffer(), pixelCount);

        device.memoryBarrier(cmd,
                             VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                             VK_ACCESS_2_SHADER_STORAGE_READ_BIT,
                             VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                             VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT);
    }

    // --- Composite ---
    NRCompositePC cpc{};
    cpc.queryOutputAddr = queryOutputAddr;
    cpc.primaryInfoAddr = primaryInfoAddr;
    cpc.width = extent.width;
    cpc.height = extent.height;
    cpc.showBlack = offlineMode ? 1u : 0u;

    compositePipeline->bindPipeline(cmd);
    compositePipeline->bindDescriptorSets(cmd, currentFrame);
    compositePipeline->pushConstants(cmd, &cpc);
    vkCmdDispatch(cmd, (extent.width + 15) / 16, (extent.height + 15) / 16, 1);

    return getOutputSlot();
}

PassImageSlot NeuralRadiosityPass::getOutputSlot() const
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
