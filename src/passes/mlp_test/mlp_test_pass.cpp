#include "mlp_test_pass.h"

#include "imgui.h"

REGISTER_RENDER_PASS_CPP(MlpTestPass, "mlp_test");

struct DataGenPushConstants
{
    uint64_t inputBuffer;
    uint64_t gtBuffer;
    uint32_t sampleCount;
    uint32_t seed;
};

struct InferencePushConstants
{
    uint64_t layerAddressBuffer;
    uint32_t width;
    uint32_t height;
    uint32_t hiddenLayerCount;
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

    // Network: input(2->64) + 1 hidden(64->64) + output(64->3)
    std::vector<NeuralNetwork::LayerConfig> layers = {
        {2, 64}, {64, 64}, {64, 64}, {64, 3}};
    network = std::make_unique<NeuralNetwork>(device, layers);
    network->initWeights(42);

    // Allocate input/gt buffers for max batch size
    inputBuffer = std::make_unique<StorageBufferResource>(
        device, (VkDeviceSize)maxBatchSize * 2 * sizeof(float),
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
            VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT);

    gtBuffer = std::make_unique<StorageBufferResource>(
        device, (VkDeviceSize)maxBatchSize * 3 * sizeof(float),
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

    inferencePipeline = std::make_unique<ComputePipeline>(
        device, 1,
        std::vector<DescriptorLayoutBinding>{
            {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, VK_SHADER_STAGE_COMPUTE_BIT},
        },
        "build/shaders/mlp_test/mlp_inference.slang.spv",
        sizeof(InferencePushConstants));

    inferencePipeline->updateDescriptorSets({
        {VkDescriptorImageInfo{outputImage.getSampler(), outputImage.getImageView(), VK_IMAGE_LAYOUT_GENERAL}},
    });
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
}

PassImageSlot MlpTestPass::recordCommand(VkCommandBuffer cmd,
                                          const PassImageSlot &inputSlot,
                                          uint32_t currentFrame, uint32_t imageIndex)
{
    if (!enabled)
        return inputSlot;

    auto extent = outputImage.getExtent();

    if (trainEnabled)
    {
        // 1. Data generation
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

        // Barrier: data gen writes -> train reads
        device.memoryBarrier(cmd,
                             VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                             VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_TRANSFER_WRITE_BIT,
                             VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                             VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_2_TRANSFER_BIT);

        // 2. Train
        network->recordTrain(cmd, inputBuffer->getBuffer(), gtBuffer->getBuffer(), (uint32_t)batchSize);

        // Barrier: train writes -> inference reads
        device.memoryBarrier(cmd,
                             VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT | VK_ACCESS_2_TRANSFER_WRITE_BIT,
                             VK_ACCESS_2_SHADER_STORAGE_READ_BIT,
                             VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                             VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT);
    }

    // 3. Transition output image to GENERAL for compute write
    device.imageBarrier(cmd, outputImage.getImage(),
                        VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL,
                        0, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                        VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, 1);

    // 4. Inference
    {
        InferencePushConstants pc{};
        pc.layerAddressBuffer = getBufferDeviceAddress(
            network->getLayerAddressBuffer());
        pc.width = extent.width;
        pc.height = extent.height;
        pc.hiddenLayerCount = (uint32_t)(network->getLayerCount() - 2);
        pc.showGT = showGT ? 1u : 0u;

        inferencePipeline->bindPipeline(cmd);
        inferencePipeline->bindDescriptorSets(cmd, currentFrame);
        inferencePipeline->pushConstants(cmd, &pc);
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
