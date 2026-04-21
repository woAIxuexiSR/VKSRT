#include "bilateral_pass.h"

#include "gbuffer.h"
#include "imgui.h"

REGISTER_RENDER_PASS_CPP(BilateralPass, "bilateral");

BilateralPass::BilateralPass(Device &_d, SwapChain &_sc, const json &params)
    : PassBase(_d, _sc),
      outputImage{_d, VK_FORMAT_R32G32B32A32_SFLOAT, _sc.getExtent(),
                  VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT}
{
    if (params.contains("sigmaS"))
        pushConstants.sigmaS = params["sigmaS"].get<float>();
    if (params.contains("sigmaN"))
        pushConstants.sigmaN = params["sigmaN"].get<float>();
    if (params.contains("sigmaP"))
        pushConstants.sigmaP = params["sigmaP"].get<float>();
    if (params.contains("kernelRadius"))
        pushConstants.kernelRadius = params["kernelRadius"].get<int>();
}

void BilateralPass::init()
{
    // Bindings: 0=input color (sampled), 1=output (storage), 2=gbuffer normal (sampled), 3=gbuffer position (sampled)
    filterPipeline = std::make_unique<ComputePipeline>(
        device, 1,
        std::vector<DescriptorLayoutBinding>{
            {VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, VK_SHADER_STAGE_COMPUTE_BIT},
            {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, VK_SHADER_STAGE_COMPUTE_BIT},
            {VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, VK_SHADER_STAGE_COMPUTE_BIT},
            {VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, VK_SHADER_STAGE_COMPUTE_BIT},
        },
        shaderPath("bilateral/bilateral.spv"),
        sizeof(BilateralPushConstants));

    filterPipeline->updateDescriptorSets({
        {VkDescriptorImageInfo{initInputSlot.sampler, initInputSlot.imageView, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL}},
        {VkDescriptorImageInfo{outputImage.getSampler(), outputImage.getImageView(), VK_IMAGE_LAYOUT_GENERAL}},
        {VkDescriptorImageInfo{gbuffer->getNormalSampler(), gbuffer->getNormalImageView(), VK_IMAGE_LAYOUT_GENERAL}},
        {VkDescriptorImageInfo{gbuffer->getPositionSampler(), gbuffer->getPositionImageView(), VK_IMAGE_LAYOUT_GENERAL}},
    });
    lastBoundInputView = initInputSlot.imageView;
    lastBoundInputSampler = initInputSlot.sampler;
}

void BilateralPass::drawUI()
{
    ImGui::SliderFloat("Sigma Spatial", &pushConstants.sigmaS, 0.5f, 16.0f);
    ImGui::SliderFloat("Sigma Normal", &pushConstants.sigmaN, 0.01f, 1.0f);
    ImGui::SliderFloat("Sigma Position", &pushConstants.sigmaP, 0.01f, 1.0f);
    ImGui::SliderInt("Kernel Radius", &pushConstants.kernelRadius, 1, 16);
}

PassImageSlot BilateralPass::recordCommand(VkCommandBuffer commandBuffer,
                                            const PassImageSlot &inputSlot,
                                            uint32_t currentFrame, uint32_t imageIndex)
{
    if (!enabled || !gbuffer || !gbuffer->isWritten())
        return inputSlot;

    if (inputSlot.imageView != lastBoundInputView || inputSlot.sampler != lastBoundInputSampler)
    {
        vkDeviceWaitIdle(device.getDevice());
        filterPipeline->updateDescriptorSets({
            {VkDescriptorImageInfo{inputSlot.sampler, inputSlot.imageView, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL}},
            {VkDescriptorImageInfo{outputImage.getSampler(), outputImage.getImageView(), VK_IMAGE_LAYOUT_GENERAL}},
            {VkDescriptorImageInfo{gbuffer->getNormalSampler(), gbuffer->getNormalImageView(), VK_IMAGE_LAYOUT_GENERAL}},
            {VkDescriptorImageInfo{gbuffer->getPositionSampler(), gbuffer->getPositionImageView(), VK_IMAGE_LAYOUT_GENERAL}},
        });
        lastBoundInputView = inputSlot.imageView;
        lastBoundInputSampler = inputSlot.sampler;
    }

    auto extent = outputImage.getExtent();

    // Transition input color -> SHADER_READ_ONLY for compute read
    device.imageBarrier(commandBuffer, inputSlot.image,
                        inputSlot.layout, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                        VK_ACCESS_2_MEMORY_WRITE_BIT, VK_ACCESS_2_SHADER_READ_BIT,
                        VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, 1);

    // G-buffer memory barrier: wait for RT write to finish (keep GENERAL layout)
    device.imageBarrier(commandBuffer, gbuffer->getNormalImage(),
                        VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_GENERAL,
                        VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT, VK_ACCESS_2_SHADER_READ_BIT,
                        VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, 1);
    device.imageBarrier(commandBuffer, gbuffer->getPositionImage(),
                        VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_GENERAL,
                        VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT, VK_ACCESS_2_SHADER_READ_BIT,
                        VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, 1);

    // Transition output to GENERAL for compute write
    device.imageBarrier(commandBuffer, outputImage.getImage(),
                        VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL,
                        0, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                        VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, 1);

    // Dispatch bilateral filter compute shader
    filterPipeline->bindPipeline(commandBuffer);
    filterPipeline->bindDescriptorSets(commandBuffer, currentFrame);
    filterPipeline->pushConstants(commandBuffer, &pushConstants);
    uint32_t groupsX = (extent.width + 15) / 16;
    uint32_t groupsY = (extent.height + 15) / 16;
    vkCmdDispatch(commandBuffer, groupsX, groupsY, 1);

    return getOutputSlot();
}

PassImageSlot BilateralPass::getOutputSlot() const
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
