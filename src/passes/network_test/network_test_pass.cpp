#include "network_test_pass.h"

#include "imgui.h"

#include <cassert>
#include <cmath>
#include <vector>

REGISTER_RENDER_PASS_CPP(NetworkTestPass, "network_test");

struct DataGenPushConstants
{
    uint64_t inputBuffer;
    uint64_t gtBuffer;
    uint32_t sampleCount;
    uint32_t seed;
    uint32_t inputDim;
};

struct WriteImagePushConstants
{
    uint64_t outputBuffer;
    uint32_t width;
    uint32_t height;
    uint32_t outputSize;
    uint32_t showGT;
};

NetworkTestPass::NetworkTestPass(Device &_d, SwapChain &_sc, const json &params)
    : PassBase(_d, _sc),
      outputImage{_d, VK_FORMAT_R8G8B8A8_UNORM, VkExtent2D{256, 256},
                  VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT}
{
    if (params.contains("batchSize"))
        batchSize = params["batchSize"].get<int>();
    if (params.contains("showGT"))
        showGT = params["showGT"].get<bool>();

    json netJson = params.contains("network") ? params["network"] : json::object();
    network = std::make_unique<NeuralNetwork>(device, netJson);
    network->initWeights(42);

    totalRawInputDim = network->getTotalRawInputDim();
    int outDim = network->getOutputSize();

    assert(outDim == 3 && "network_test requires outputSize == 3");

    inputBuffer = std::make_unique<StorageBufferResource>(
        device, (VkDeviceSize)maxBatchSize * totalRawInputDim * sizeof(float),
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
            VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT);

    gtBuffer = std::make_unique<StorageBufferResource>(
        device, (VkDeviceSize)maxBatchSize * outDim * sizeof(float),
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
            VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT);

    auto extent = outputImage.getExtent();
    uint32_t pixelCount = extent.width * extent.height;

    inferenceInputBuffer = std::make_unique<StorageBufferResource>(
        device, (VkDeviceSize)pixelCount * totalRawInputDim * sizeof(float),
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
            VK_BUFFER_USAGE_TRANSFER_DST_BIT |
            VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT);

    std::vector<float> gridCoords(pixelCount * totalRawInputDim, 0.0f);
    for (uint32_t y = 0; y < extent.height; y++)
    {
        for (uint32_t x = 0; x < extent.width; x++)
        {
            uint32_t idx = (y * extent.width + x) * totalRawInputDim;
            float u = ((float)x + 0.5f) / (float)extent.width;
            float v = ((float)y + 0.5f) / (float)extent.height;
            for (int f = 0; f < totalRawInputDim; f += 2)
            {
                gridCoords[idx + f + 0] = u;
                if (f + 1 < totalRawInputDim)
                    gridCoords[idx + f + 1] = v;
            }
        }
    }
    inferenceInputBuffer->update(gridCoords.data());

    inferenceOutputBuffer = std::make_unique<StorageBufferResource>(
        device, (VkDeviceSize)pixelCount * outDim * sizeof(float),
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
            VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT);
}

void NetworkTestPass::init()
{
    std::vector<DescriptorLayoutBinding> emptyBindings{};

    dataGenPipeline = std::make_unique<ComputePipeline>(
        device, 1, emptyBindings,
        shaderPath("network_test/mlp_data_gen.spv"),
        sizeof(DataGenPushConstants));

    writeImagePipeline = std::make_unique<ComputePipeline>(
        device, 1,
        std::vector<DescriptorLayoutBinding>{
            {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, VK_SHADER_STAGE_COMPUTE_BIT},
        },
        shaderPath("network_test/write_image.spv"),
        sizeof(WriteImagePushConstants));

    writeImagePipeline->updateDescriptorSets({
        {VkDescriptorImageInfo{outputImage.getSampler(), outputImage.getImageView(), VK_IMAGE_LAYOUT_GENERAL}},
    });

    network->createPipelines();
}

void NetworkTestPass::update(uint32_t currentFrame, InputState &inputState)
{
    if (!enabled)
        return;
    currentLoss = network->readLoss();
    frameIndex++;
}

void NetworkTestPass::drawUI()
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
    if (network->hasEncodings())
        ImGui::Text("Encodings: %d", (int)network->getTotalEncodedDim());
}

PassImageSlot NetworkTestPass::recordCommand(VkCommandBuffer cmd,
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
            pc.inputBuffer = device.getBufferDeviceAddress(inputBuffer->getBuffer());
            pc.gtBuffer = device.getBufferDeviceAddress(gtBuffer->getBuffer());
            pc.sampleCount = (uint32_t)batchSize;
            pc.seed = frameIndex * maxBatchSize;
            pc.inputDim = (uint32_t)totalRawInputDim;

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
        pc.outputBuffer = device.getBufferDeviceAddress(inferenceOutputBuffer->getBuffer());
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

PassImageSlot NetworkTestPass::getOutputSlot() const
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
