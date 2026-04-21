#include "nrc_pass.h"

#include "camera.h"
#include "gbuffer.h"
#include "imgui.h"

#include <cassert>
#include <algorithm>

REGISTER_RENDER_PASS_CPP(NRCPass, "nrc");

static VkExtent2D getNrcExtent(const json &params, const SwapChain &sc)
{
    uint32_t w = 128, h = 128;
    if (params.contains("width")) w = params["width"].get<uint32_t>();
    if (params.contains("height")) h = params["height"].get<uint32_t>();
    return {w, h};
}

NRCPass::NRCPass(Device &_d, SwapChain &_sc, const json &params)
    : PassBase(_d, _sc),
      colorImage{_d, VK_FORMAT_R32G32B32A32_SFLOAT, getNrcExtent(params, _sc),
                 VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT |
                     VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT},
      shortPathImage{_d, VK_FORMAT_R32G32B32A32_SFLOAT, getNrcExtent(params, _sc),
                     VK_IMAGE_USAGE_STORAGE_BIT},
      throughputImage{_d, VK_FORMAT_R32G32B32A32_SFLOAT, getNrcExtent(params, _sc),
                      VK_IMAGE_USAGE_STORAGE_BIT},
      uniformBuffer{_d, sizeof(NRCUniform), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT},
      model{_d}
{
    vkGetBufferDeviceAddressKHR = device.loadDeviceFunction<PFN_vkGetBufferDeviceAddressKHR>(
        "vkGetBufferDeviceAddressKHR");

    if (params.contains("maxDepth"))
        pushConstants.maxDepth = params["maxDepth"].get<int>();
    if (params.contains("rrDepth"))
        pushConstants.rrDepth = params["rrDepth"].get<int>();
    if (params.contains("nrcQueryDepth"))
        pushConstants.nrcQueryDepth = params["nrcQueryDepth"].get<int>();
    if (params.contains("trainFraction"))
        pushConstants.trainFraction = params["trainFraction"].get<int>();

    SceneLoader::loadScene(params, model);
    model.buildAccelerationStructures();
    pushConstants.lightCount = model.getLightCount();

    auto extent = colorImage.getExtent();
    pushConstants.screenWidth = extent.width;
    pushConstants.screenHeight = extent.height;
    pixelCount = extent.width * extent.height;
    trainBatchSize = pixelCount / pushConstants.trainFraction;

    // Build neural network from config
    int hiddenSize = 64;
    int hiddenLayers = 2;
    int mlpOutputSize = 3;
    int mlpInputSize = 0;

    NeuralNetwork::Config netCfg{};

    if (params.contains("network"))
    {
        const auto &net = params["network"];

        if (net.contains("encoding") && net["encoding"].is_array())
        {
            totalRawInputDim = 0;
            for (auto &enc : net["encoding"])
            {
                NeuralNetwork::EncodingConfig ec;
                ec.type = enc["type"].get<std::string>();
                ec.inputDim = enc["inputDim"].get<int>();
                ec.params = enc;
                netCfg.encodings.push_back(ec);
                totalRawInputDim += ec.inputDim;
            }

            for (auto &ec : netCfg.encodings)
            {
                if (ec.type == "identity")
                    mlpInputSize += ec.inputDim;
                else if (ec.type == "hashgrid")
                {
                    int nl = 16, fpl = 2;
                    if (ec.params.contains("numLevels")) nl = ec.params["numLevels"].get<int>();
                    if (ec.params.contains("featuresPerLevel")) fpl = ec.params["featuresPerLevel"].get<int>();
                    mlpInputSize += nl * fpl;
                }
                else if (ec.type == "sh")
                {
                    int deg = 4;
                    if (ec.params.contains("degree")) deg = ec.params["degree"].get<int>();
                    mlpInputSize += (deg + 1) * (deg + 1);
                }
                else if (ec.type == "frequency")
                {
                    int nf = 6;
                    if (ec.params.contains("numFreqs")) nf = ec.params["numFreqs"].get<int>();
                    mlpInputSize += ec.inputDim * nf * 2;
                }
                else if (ec.type == "oneblob")
                {
                    int nb = 16;
                    if (ec.params.contains("numBins")) nb = ec.params["numBins"].get<int>();
                    mlpInputSize += ec.inputDim * nb;
                }
            }
        }

        if (net.contains("mlp"))
        {
            const auto &mlp = net["mlp"];
            if (mlp.contains("outputSize")) mlpOutputSize = mlp["outputSize"].get<int>();
            if (mlp.contains("hiddenSize")) hiddenSize = mlp["hiddenSize"].get<int>();
            if (mlp.contains("hiddenLayers")) hiddenLayers = mlp["hiddenLayers"].get<int>();
        }

        if (net.contains("useEMA"))
            netCfg.useEMA = net["useEMA"].get<bool>();
        if (net.contains("emaAlpha"))
            netCfg.emaAlpha = net["emaAlpha"].get<float>();
    }

    assert(mlpOutputSize == 3 && "NRC output must be 3 (RGB radiance)");
    assert(hiddenSize == 64 && "hiddenSize must be 64");
    assert(totalRawInputDim == 12 && "NRC expects 12 raw input dims (pos+normal+dir+albedo)");

    netCfg.layers.push_back({mlpInputSize, hiddenSize});
    for (int i = 0; i < hiddenLayers; i++)
        netCfg.layers.push_back({hiddenSize, hiddenSize});
    netCfg.layers.push_back({hiddenSize, mlpOutputSize});

    network = std::make_unique<NeuralNetwork>(device, netCfg);
    network->initWeights(42);

    VkBufferUsageFlags bdaFlags = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                                  VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;

    trainInputBuffer = std::make_unique<StorageBufferResource>(
        device, (VkDeviceSize)trainBatchSize * totalRawInputDim * sizeof(float), bdaFlags);
    trainGTBuffer = std::make_unique<StorageBufferResource>(
        device, (VkDeviceSize)trainBatchSize * mlpOutputSize * sizeof(float), bdaFlags);
    trainCounterBuffer = std::make_unique<StorageBufferResource>(
        device, sizeof(uint32_t), bdaFlags | VK_BUFFER_USAGE_TRANSFER_DST_BIT);
    queryInputBuffer = std::make_unique<StorageBufferResource>(
        device, (VkDeviceSize)pixelCount * totalRawInputDim * sizeof(float), bdaFlags);
    queryOutputBuffer = std::make_unique<StorageBufferResource>(
        device, (VkDeviceSize)pixelCount * mlpOutputSize * sizeof(float), bdaFlags);

    pushConstants.trainInputAddr = getBufferDeviceAddress(trainInputBuffer->getBuffer());
    pushConstants.trainGTAddr = getBufferDeviceAddress(trainGTBuffer->getBuffer());
    pushConstants.trainCounterAddr = getBufferDeviceAddress(trainCounterBuffer->getBuffer());
    pushConstants.queryInputAddr = getBufferDeviceAddress(queryInputBuffer->getBuffer());
}

uint64_t NRCPass::getBufferDeviceAddress(VkBuffer buffer)
{
    VkBufferDeviceAddressInfoKHR info{};
    info.sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO_KHR;
    info.buffer = buffer;
    return vkGetBufferDeviceAddressKHR(device.getDevice(), &info);
}

void NRCPass::init()
{
    // RT pipeline descriptors (12 bindings)
    std::vector<DescriptorLayoutBinding> rtBindings = {
        {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, VK_SHADER_STAGE_RAYGEN_BIT_KHR}, // 0: colorImage
        {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, VK_SHADER_STAGE_RAYGEN_BIT_KHR}, // 1: camera UBO
    };
    VkShaderStageFlags hitStages = VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR;
    auto modelBindings = model.getDescriptorBindings(hitStages);
    rtBindings.insert(rtBindings.end(), modelBindings.begin(), modelBindings.end()); // 2-9
    // NRC intermediate images (10-11)
    rtBindings.push_back({VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, VK_SHADER_STAGE_RAYGEN_BIT_KHR});
    rtBindings.push_back({VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, VK_SHADER_STAGE_RAYGEN_BIT_KHR});

    rtPipeline = std::make_unique<RayTracingPipeline>(
        device, 1, rtBindings,
        shaderPath("nrc/nrc.spv"),
        model.getHitSBTRecords(),
        "raygenMain", "missMain", "closestHitMain",
        sizeof(NRCPushConstants));

    std::vector<std::vector<DescriptorInfo>> rtInfos = {
        {VkDescriptorImageInfo{colorImage.getSampler(), colorImage.getImageView(), VK_IMAGE_LAYOUT_GENERAL}},
        {VkDescriptorBufferInfo{uniformBuffer.getBuffer(), 0, sizeof(NRCUniform)}},
    };
    auto modelInfos = model.getDescriptorInfos();
    rtInfos.insert(rtInfos.end(), modelInfos.begin(), modelInfos.end());
    rtInfos.push_back({VkDescriptorImageInfo{shortPathImage.getSampler(), shortPathImage.getImageView(), VK_IMAGE_LAYOUT_GENERAL}});
    rtInfos.push_back({VkDescriptorImageInfo{throughputImage.getSampler(), throughputImage.getImageView(), VK_IMAGE_LAYOUT_GENERAL}});
    rtPipeline->updateDescriptorSets(rtInfos);

    // Composite compute pipeline (3 bindings)
    compositePipeline = std::make_unique<ComputePipeline>(
        device, 1,
        std::vector<DescriptorLayoutBinding>{
            {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, VK_SHADER_STAGE_COMPUTE_BIT},
            {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, VK_SHADER_STAGE_COMPUTE_BIT},
            {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, VK_SHADER_STAGE_COMPUTE_BIT},
        },
        shaderPath("nrc/nrc_composite.spv"),
        sizeof(NRCCompositePushConstants));

    compositePipeline->updateDescriptorSets({
        {VkDescriptorImageInfo{shortPathImage.getSampler(), shortPathImage.getImageView(), VK_IMAGE_LAYOUT_GENERAL}},
        {VkDescriptorImageInfo{throughputImage.getSampler(), throughputImage.getImageView(), VK_IMAGE_LAYOUT_GENERAL}},
        {VkDescriptorImageInfo{colorImage.getSampler(), colorImage.getImageView(), VK_IMAGE_LAYOUT_GENERAL}},
    });

    network->createPipelines();
}

void NRCPass::drawUI()
{
    ImGui::SliderInt("Max Depth (train)", &pushConstants.maxDepth, 1, 32);
    ImGui::SliderInt("NRC Query Depth", &pushConstants.nrcQueryDepth, 1, 16);
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
    bool nrc = pushConstants.useNRC != 0;
    if (ImGui::Checkbox("Enable NRC", &nrc))
        pushConstants.useNRC = nrc ? 1 : 0;
    if (pushConstants.useNRC)
    {
        ImGui::Text("  Train Fraction: 1/%d", pushConstants.trainFraction);
        ImGui::Text("  Train Batch: %u", trainBatchSize);
        ImGui::Checkbox("Train NRC", &trainEnabled);
        ImGui::Text("  Loss: %.6f", currentLoss);
        ImGui::Text("  Params: %d", network->getTotalParams());
        if (network->isEMAEnabled())
            ImGui::Text("  EMA: enabled");
    }
}

void NRCPass::update(uint32_t currentFrame, InputState &inputState)
{
    static int lastNEE = pushConstants.useNEE;
    static int lastMIS = pushConstants.useMIS;
    static int lastMaxDepth = pushConstants.maxDepth;
    static int lastRRDepth = pushConstants.rrDepth;
    static int lastQueryDepth = pushConstants.nrcQueryDepth;
    static int lastUseNRC = pushConstants.useNRC;
    if (pushConstants.useNEE != lastNEE || pushConstants.useMIS != lastMIS ||
        pushConstants.maxDepth != lastMaxDepth || pushConstants.rrDepth != lastRRDepth ||
        pushConstants.nrcQueryDepth != lastQueryDepth || pushConstants.useNRC != lastUseNRC)
    {
        inputState.keyboardChanged = true;
        lastNEE = pushConstants.useNEE;
        lastMIS = pushConstants.useMIS;
        lastMaxDepth = pushConstants.maxDepth;
        lastRRDepth = pushConstants.rrDepth;
        lastQueryDepth = pushConstants.nrcQueryDepth;
        lastUseNRC = pushConstants.useNRC;
    }

    if (!inputState.isChanged())
        pushConstants.frameIndex++;
    else
        pushConstants.frameIndex = 0;

    ubo.viewInverse = camera->getInverseViewMatrix();
    ubo.projInverse = camera->getInverseProjectionMatrix();
    uniformBuffer.update(&ubo);

    currentLoss = network->readLoss();

    if (pushConstants.frameIndex > 0 && pushConstants.frameIndex % 100 == 0)
        printf("  [NRC] frame %d  loss: %.6f\n", pushConstants.frameIndex, currentLoss);
}

PassImageSlot NRCPass::recordCommand(VkCommandBuffer cmd,
                                      const PassImageSlot &inputSlot,
                                      uint32_t currentFrame, uint32_t imageIndex)
{
    auto extent = colorImage.getExtent();

    // === Phase 0: Barriers ===

    VkImageLayout oldLayout = firstFrame ? VK_IMAGE_LAYOUT_UNDEFINED : VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    VkAccessFlags2 srcAccess = firstFrame ? (VkAccessFlags2)0 : VK_ACCESS_2_SHADER_READ_BIT;
    VkPipelineStageFlags2 srcStage = firstFrame ? VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT : VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;

    device.imageBarrier(cmd, colorImage.getImage(),
                        oldLayout, VK_IMAGE_LAYOUT_GENERAL,
                        srcAccess, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                        srcStage, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, 1);

    VkImageLayout nrcOld = firstFrame ? VK_IMAGE_LAYOUT_UNDEFINED : VK_IMAGE_LAYOUT_GENERAL;
    VkAccessFlags2 nrcSrc = firstFrame ? (VkAccessFlags2)0 : VK_ACCESS_2_SHADER_STORAGE_READ_BIT;
    VkPipelineStageFlags2 nrcSrcStage = firstFrame ? VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT : VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;

    device.imageBarrier(cmd, shortPathImage.getImage(),
                        nrcOld, VK_IMAGE_LAYOUT_GENERAL,
                        nrcSrc, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                        nrcSrcStage, VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR, 1);
    device.imageBarrier(cmd, throughputImage.getImage(),
                        nrcOld, VK_IMAGE_LAYOUT_GENERAL,
                        nrcSrc, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                        nrcSrcStage, VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR, 1);

    // Zero training counter
    vkCmdFillBuffer(cmd, trainCounterBuffer->getBuffer(), 0, sizeof(uint32_t), 0);
    device.memoryBarrier(cmd,
                         VK_ACCESS_2_TRANSFER_WRITE_BIT,
                         VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                         VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                         VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR);

    firstFrame = false;

    // === Phase 1: RT Dispatch ===

    rtPipeline->bindPipeline(cmd);
    rtPipeline->bindDescriptorSets(cmd, currentFrame);
    rtPipeline->pushConstants(cmd, &pushConstants);
    rtPipeline->traceRays(cmd, {extent.width, extent.height, 1});

    device.memoryBarrier(cmd,
                         VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                         VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_TRANSFER_WRITE_BIT,
                         VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR,
                         VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_2_TRANSFER_BIT);

    // === Phase 2 & 3: Training + Inference (only when NRC enabled) ===

    if (pushConstants.useNRC)
    {
        uint32_t maxSamples = std::max(trainBatchSize, pixelCount);
        network->ensureBuffers(maxSamples);

        if (trainEnabled)
        {
            network->recordTrain(cmd, trainInputBuffer->getBuffer(),
                                 trainGTBuffer->getBuffer(), trainBatchSize);

            device.memoryBarrier(cmd,
                                 VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT | VK_ACCESS_2_TRANSFER_WRITE_BIT,
                                 VK_ACCESS_2_SHADER_STORAGE_READ_BIT,
                                 VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                                 VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT);
        }

        network->recordForward(cmd, queryInputBuffer->getBuffer(),
                               queryOutputBuffer->getBuffer(), pixelCount);

        device.memoryBarrier(cmd,
                             VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                             VK_ACCESS_2_SHADER_STORAGE_READ_BIT,
                             VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                             VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT);
    }

    // === Phase 4: Composite ===

    NRCCompositePushConstants compositePc{};
    compositePc.queryOutputAddr = getBufferDeviceAddress(queryOutputBuffer->getBuffer());
    compositePc.width = extent.width;
    compositePc.height = extent.height;

    compositePipeline->bindPipeline(cmd);
    compositePipeline->bindDescriptorSets(cmd, currentFrame);
    compositePipeline->pushConstants(cmd, &compositePc);
    vkCmdDispatch(cmd, (extent.width + 15) / 16, (extent.height + 15) / 16, 1);

    return getOutputSlot();
}

PassImageSlot NRCPass::getOutputSlot() const
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
