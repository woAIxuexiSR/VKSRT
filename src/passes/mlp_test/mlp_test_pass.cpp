#include "mlp_test_pass.h"

#include "imgui.h"

#include <cassert>
#include <cmath>
#include <vector>

REGISTER_RENDER_PASS_CPP(MlpTestPass, "mlp_test");

struct DataGenPushConstants
{
    uint64_t inputBuffer;
    uint64_t gtBuffer;
    uint32_t sampleCount;
    uint32_t seed;
};

struct WriteImagePushConstants
{
    uint64_t outputBuffer;
    uint32_t width;
    uint32_t height;
    uint32_t outputSize;
    uint32_t showGT;
};

MlpTestPass::MlpTestPass(Device &_d, SwapChain &_sc, const json &params)
    : PassBase(_d, _sc),
      outputImage{_d, VK_FORMAT_R8G8B8A8_UNORM, VkExtent2D{256, 256},
                  VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT}
{
    vkGetBufferDeviceAddressKHR = device.loadDeviceFunction<PFN_vkGetBufferDeviceAddressKHR>(
        "vkGetBufferDeviceAddressKHR");

    if (params.contains("batchSize"))
        batchSize = params["batchSize"].get<int>();
    if (params.contains("showGT"))
        showGT = params["showGT"].get<bool>();

    int hiddenSize = 64;
    int hiddenLayers = 2;
    int mlpInputSize = 2;
    int mlpOutputSize = 3;

    NeuralNetwork::Config netCfg{};

    if (params.contains("network"))
    {
        const auto &net = params["network"];

        if (net.contains("encoding"))
        {
            const auto &enc = net["encoding"];
            useEncoding = true;
            netCfg.useEncoding = true;
            if (enc.contains("numLevels"))          netCfg.encoding.numLevels = enc["numLevels"].get<int>();
            if (enc.contains("featuresPerLevel"))   netCfg.encoding.featuresPerLevel = enc["featuresPerLevel"].get<int>();
            if (enc.contains("tableSize"))          netCfg.encoding.tableSize = enc["tableSize"].get<int>();
            if (enc.contains("coarsestResolution")) netCfg.encoding.coarsestResolution = enc["coarsestResolution"].get<int>();
            if (enc.contains("finestResolution"))   netCfg.encoding.finestResolution = enc["finestResolution"].get<int>();
            if (enc.contains("inputDim"))           netCfg.encoding.inputDim = enc["inputDim"].get<int>();
            mlpInputSize = netCfg.encoding.numLevels * netCfg.encoding.featuresPerLevel;
        }

        if (net.contains("mlp"))
        {
            const auto &mlp = net["mlp"];
            if (!useEncoding && mlp.contains("inputSize")) mlpInputSize = mlp["inputSize"].get<int>();
            if (mlp.contains("outputSize"))                mlpOutputSize = mlp["outputSize"].get<int>();
            if (mlp.contains("hiddenSize"))                hiddenSize = mlp["hiddenSize"].get<int>();
            if (mlp.contains("hiddenLayers"))              hiddenLayers = mlp["hiddenLayers"].get<int>();
        }
        else
        {
            if (!useEncoding && net.contains("inputSize"))    mlpInputSize = net["inputSize"].get<int>();
            if (net.contains("outputSize"))                   mlpOutputSize = net["outputSize"].get<int>();
            if (net.contains("hiddenSize"))                   hiddenSize = net["hiddenSize"].get<int>();
            if (net.contains("hiddenLayers"))                 hiddenLayers = net["hiddenLayers"].get<int>();
        }
    }

    assert(mlpOutputSize == 3 && "mlp_test requires outputSize == 3");
    if (useEncoding)
        assert(netCfg.encoding.inputDim == 2 && "mlp_test hash encoding requires inputDim == 2");
    else
        assert(mlpInputSize == 2 && "mlp_test (no encoding) requires inputSize == 2");
    assert(hiddenSize == 64 && "hiddenSize must be 64");

    netCfg.layers.push_back({mlpInputSize, hiddenSize});
    for (int i = 0; i < hiddenLayers; i++)
        netCfg.layers.push_back({hiddenSize, hiddenSize});
    netCfg.layers.push_back({hiddenSize, mlpOutputSize});

    network = std::make_unique<NeuralNetwork>(device, netCfg);
    network->initWeights(42);

    int rawInputDim = 2;
    int outDim = mlpOutputSize;

    inputBuffer = std::make_unique<StorageBufferResource>(
        device, (VkDeviceSize)maxBatchSize * rawInputDim * sizeof(float),
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
            VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT);

    gtBuffer = std::make_unique<StorageBufferResource>(
        device, (VkDeviceSize)maxBatchSize * outDim * sizeof(float),
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
            VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT);

    auto extent = outputImage.getExtent();
    uint32_t pixelCount = extent.width * extent.height;

    inferenceInputBuffer = std::make_unique<StorageBufferResource>(
        device, (VkDeviceSize)pixelCount * rawInputDim * sizeof(float),
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
            VK_BUFFER_USAGE_TRANSFER_DST_BIT |
            VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT);

    std::vector<float> gridCoords(pixelCount * rawInputDim);
    for (uint32_t y = 0; y < extent.height; y++)
    {
        for (uint32_t x = 0; x < extent.width; x++)
        {
            uint32_t idx = (y * extent.width + x) * rawInputDim;
            gridCoords[idx + 0] = ((float)x + 0.5f) / (float)extent.width;
            gridCoords[idx + 1] = ((float)y + 0.5f) / (float)extent.height;
        }
    }
    inferenceInputBuffer->update(gridCoords.data());

    inferenceOutputBuffer = std::make_unique<StorageBufferResource>(
        device, (VkDeviceSize)pixelCount * outDim * sizeof(float),
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
            VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT);
}

uint64_t MlpTestPass::getBufferDeviceAddress(VkBuffer buffer)
{
    VkBufferDeviceAddressInfoKHR info{};
    info.sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO_KHR;
    info.buffer = buffer;
    return vkGetBufferDeviceAddressKHR(device.getDevice(), &info);
}

void MlpTestPass::init()
{
    std::vector<DescriptorLayoutBinding> emptyBindings{};

    dataGenPipeline = std::make_unique<ComputePipeline>(
        device, 1, emptyBindings,
        "build/shaders/mlp_test/mlp_data_gen.slang.spv",
        sizeof(DataGenPushConstants));

    writeImagePipeline = std::make_unique<ComputePipeline>(
        device, 1,
        std::vector<DescriptorLayoutBinding>{
            {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, VK_SHADER_STAGE_COMPUTE_BIT},
        },
        "build/shaders/mlp_test/write_image.slang.spv",
        sizeof(WriteImagePushConstants));

    writeImagePipeline->updateDescriptorSets({
        {VkDescriptorImageInfo{outputImage.getSampler(), outputImage.getImageView(), VK_IMAGE_LAYOUT_GENERAL}},
    });

    network->createPipelines();
}

void MlpTestPass::update(uint32_t currentFrame, InputState &inputState)
{
    if (!enabled)
        return;
    currentLoss = network->readLoss();
    frameIndex++;
}

void MlpTestPass::drawUI()
{
    ImGui::Checkbox("Train", &trainEnabled);
    ImGui::Text("Batch Size: %d", batchSize);
    {
        int logVal = 0;
        for (int v = batchSize; v > 1; v >>= 1) logVal++;
        if (ImGui::SliderInt("##batch", &logVal, 6, 12))
            batchSize = 1 << logVal;
    }
    ImGui::Checkbox("Show GT", &showGT);
    ImGui::Text("Loss: %.6f", currentLoss);
    ImGui::Text("Frame: %u", frameIndex);
    ImGui::Text("Params: %d", network->getTotalParams());
    ImGui::Text("Encoding: %s", useEncoding ? "HashGrid" : "None");
}

PassImageSlot MlpTestPass::recordCommand(VkCommandBuffer cmd,
                                          const PassImageSlot &inputSlot,
                                          uint32_t currentFrame, uint32_t imageIndex)
{
    if (!enabled)
        return inputSlot;

    auto extent = outputImage.getExtent();
    uint32_t pixelCount = extent.width * extent.height;

    uint32_t maxSamples = pixelCount;
    if (trainEnabled)
        maxSamples = std::max(maxSamples, (uint32_t)batchSize);
    network->ensureBuffers(maxSamples);

    if (trainEnabled)
    {
        {
            DataGenPushConstants pc{};
            pc.inputBuffer = getBufferDeviceAddress(inputBuffer->getBuffer());
            pc.gtBuffer = getBufferDeviceAddress(gtBuffer->getBuffer());
            pc.sampleCount = (uint32_t)batchSize;
            pc.seed = frameIndex * maxBatchSize;

            dataGenPipeline->bindPipeline(cmd);
            dataGenPipeline->pushConstants(cmd, &pc);
            vkCmdDispatch(cmd, ((uint32_t)batchSize + 255) / 256, 1, 1);
        }

        device.memoryBarrier(cmd,
                             VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                             VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_TRANSFER_WRITE_BIT,
                             VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                             VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_2_TRANSFER_BIT);

        network->recordTrain(cmd, inputBuffer->getBuffer(), gtBuffer->getBuffer(), (uint32_t)batchSize);

        device.memoryBarrier(cmd,
                             VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT | VK_ACCESS_2_TRANSFER_WRITE_BIT,
                             VK_ACCESS_2_SHADER_STORAGE_READ_BIT,
                             VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                             VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT);
    }

    device.imageBarrier(cmd, outputImage.getImage(),
                        VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL,
                        0, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                        VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, 1);

    if (!showGT)
    {
        network->recordForward(cmd, inferenceInputBuffer->getBuffer(),
                               inferenceOutputBuffer->getBuffer(), pixelCount);

        device.memoryBarrier(cmd,
                             VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                             VK_ACCESS_2_SHADER_STORAGE_READ_BIT,
                             VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                             VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT);
    }

    {
        WriteImagePushConstants pc{};
        pc.outputBuffer = getBufferDeviceAddress(inferenceOutputBuffer->getBuffer());
        pc.width = extent.width;
        pc.height = extent.height;
        pc.outputSize = (uint32_t)network->getOutputSize();
        pc.showGT = showGT ? 1u : 0u;

        writeImagePipeline->bindPipeline(cmd);
        writeImagePipeline->bindDescriptorSets(cmd, currentFrame);
        writeImagePipeline->pushConstants(cmd, &pc);
        vkCmdDispatch(cmd, (extent.width + 15) / 16, (extent.height + 15) / 16, 1);
    }

    return getOutputSlot();
}

PassImageSlot MlpTestPass::getOutputSlot() const
{
    return {
        outputImage.getImage(),
        outputImage.getImageView(),
        outputImage.getSampler(),
        VK_FORMAT_R8G8B8A8_UNORM,
        outputImage.getExtent(),
        VK_IMAGE_LAYOUT_GENERAL,
    };
}
