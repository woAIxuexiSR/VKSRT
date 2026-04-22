#include "taa_pass.h"

#include "camera.h"
#include "gbuffer.h"
#include "imgui.h"

REGISTER_RENDER_PASS_CPP(TAAPass, "taa");

TAAPass::TAAPass(Device &_d, SwapChain &_sc, const json &params)
    : PassBase(_d, _sc),
      historyImage{_d, VK_FORMAT_R32G32B32A32_SFLOAT, _sc.getExtent(),
                   VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT},
      outputImage{_d, VK_FORMAT_R32G32B32A32_SFLOAT, _sc.getExtent(),
                  VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT},
      uniformBuffer{_d, sizeof(TAAUniform), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT}
{
    if (params.contains("blendFactor"))
        pushConstants.blendFactor = params["blendFactor"].get<float>();
}

void TAAPass::init()
{
    taaPipeline = std::make_unique<ComputePipeline>(
        device, 1,
        std::vector<DescriptorLayoutBinding>{
            {VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, VK_SHADER_STAGE_COMPUTE_BIT},          // 0: currentFrame
            {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, VK_SHADER_STAGE_COMPUTE_BIT},          // 1: outputImage
            {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_COMPUTE_BIT}, // 2: historyBuffer
            {VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, VK_SHADER_STAGE_COMPUTE_BIT},          // 3: gbufferPosition
            {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT},         // 4: taaParams
        },
        shaderPath("taa/taa.spv"),
        sizeof(TAAPushConstants));

    taaPipeline->updateDescriptorSets({
        {VkDescriptorImageInfo{initInputSlot.sampler, initInputSlot.imageView, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL}},
        {VkDescriptorImageInfo{outputImage.getSampler(), outputImage.getImageView(), VK_IMAGE_LAYOUT_GENERAL}},
        {VkDescriptorImageInfo{historyImage.getSampler(), historyImage.getImageView(), VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL}},
        {VkDescriptorImageInfo{gbuffer->getPositionSampler(), gbuffer->getPositionImageView(), VK_IMAGE_LAYOUT_GENERAL}},
        {VkDescriptorBufferInfo{uniformBuffer.getBuffer(), 0, sizeof(TAAUniform)}},
    });
    markInputSlotBound(initInputSlot);
}

void TAAPass::drawUI()
{
    ImGui::SliderFloat("Blend Factor", &pushConstants.blendFactor, 0.01f, 0.5f);
    ImGui::Text("Mode: %s", pushConstants.useAccumMode ? "Accumulate" : "TAA");
    ImGui::Text("Frame Index: %d", pushConstants.frameIndex);
}

void TAAPass::update(uint32_t currentFrame, InputState &inputState)
{
    if (!enabled)
    {
        state = State::Disabled;
        pushConstants.frameIndex = 0;
        return;
    }

    // Re-enable from Disabled: history contents are stale, treat as Fresh.
    if (state == State::Disabled)
        state = State::Fresh;

    bool isInteracting = inputState.isChanged();
    bool gbufferAvailable = gbuffer && gbuffer->isWritten();
    pendingMode = (isInteracting && gbufferAvailable) ? State::TAA : State::Accumulate;

    // Reset accumulation on Fresh (first frame after init/re-enable) or on
    // TAA <-> Accumulate transition. Fresh compares unequal to TAA/Accumulate,
    // so the single inequality check covers both cases.
    if (state != pendingMode)
        pushConstants.frameIndex = 0;
    else
        pushConstants.frameIndex++;

    pushConstants.useAccumMode = (pendingMode == State::Accumulate) ? 1 : 0;

    // Update uniform buffer
    taaUniform.prevViewProj = camera->getPrevViewProjectionMatrix();
    auto extent = outputImage.getExtent();
    taaUniform.screenWidth = extent.width;
    taaUniform.screenHeight = extent.height;
    uniformBuffer.update(&taaUniform);

    // state is advanced to pendingMode by recordCommand after it consumes Fresh.
}

PassImageSlot TAAPass::recordCommand(VkCommandBuffer commandBuffer,
                                      const PassImageSlot &inputSlot,
                                      uint32_t currentFrame, uint32_t imageIndex)
{
    if (!enabled)
        return inputSlot;

    // Dynamic rebind if upstream pass changed
    if (inputSlotChanged(inputSlot))
    {
        vkDeviceWaitIdle(device.getDevice());
        taaPipeline->updateDescriptorSets({
            {VkDescriptorImageInfo{inputSlot.sampler, inputSlot.imageView, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL}},
            {VkDescriptorImageInfo{outputImage.getSampler(), outputImage.getImageView(), VK_IMAGE_LAYOUT_GENERAL}},
            {VkDescriptorImageInfo{historyImage.getSampler(), historyImage.getImageView(), VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL}},
            {VkDescriptorImageInfo{gbuffer->getPositionSampler(), gbuffer->getPositionImageView(), VK_IMAGE_LAYOUT_GENERAL}},
            {VkDescriptorBufferInfo{uniformBuffer.getBuffer(), 0, sizeof(TAAUniform)}},
        });
    }

    auto extent = outputImage.getExtent();

    // 1. Transition input -> SHADER_READ_ONLY
    device.imageBarrier(commandBuffer, inputSlot.image,
                        inputSlot.layout, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                        VK_ACCESS_2_MEMORY_WRITE_BIT, VK_ACCESS_2_SHADER_READ_BIT,
                        VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, 1);

    // 2. Transition history buffer for reading.
    // Fresh: historyImage has never been written, use UNDEFINED to discard stale contents.
    bool isFresh = (state == State::Fresh);
    VkImageLayout histOldLayout = isFresh ? VK_IMAGE_LAYOUT_UNDEFINED : VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    VkAccessFlags2 histSrcAccess = isFresh ? (VkAccessFlags2)0 : VK_ACCESS_2_TRANSFER_WRITE_BIT;
    VkPipelineStageFlags2 histSrcStage = isFresh ? VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT : VK_PIPELINE_STAGE_2_TRANSFER_BIT;

    device.imageBarrier(commandBuffer, historyImage.getImage(),
                        histOldLayout, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                        histSrcAccess, VK_ACCESS_2_SHADER_READ_BIT,
                        histSrcStage, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, 1);

    // 3. G-buffer position: ensure GENERAL layout for descriptor binding
    if (gbuffer)
    {
        VkImageLayout gbufOldLayout = isFresh ? VK_IMAGE_LAYOUT_UNDEFINED : VK_IMAGE_LAYOUT_GENERAL;
        device.imageBarrier(commandBuffer, gbuffer->getPositionImage(),
                            gbufOldLayout, VK_IMAGE_LAYOUT_GENERAL,
                            VK_ACCESS_2_MEMORY_WRITE_BIT, VK_ACCESS_2_SHADER_READ_BIT,
                            VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, 1);
    }

    // 4. Transition output to GENERAL for compute write
    device.imageBarrier(commandBuffer, outputImage.getImage(),
                        VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL,
                        0, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                        VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, 1);

    // Dispatch TAA compute shader
    taaPipeline->bindPipeline(commandBuffer);
    taaPipeline->bindDescriptorSets(commandBuffer, currentFrame);
    taaPipeline->pushConstants(commandBuffer, &pushConstants);
    uint32_t groupsX = (extent.width + 15) / 16;
    uint32_t groupsY = (extent.height + 15) / 16;
    vkCmdDispatch(commandBuffer, groupsX, groupsY, 1);

    // 5. Copy output -> history for next frame
    // Transition output: GENERAL -> TRANSFER_SRC
    device.imageBarrier(commandBuffer, outputImage.getImage(),
                        VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                        VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT, VK_ACCESS_2_TRANSFER_READ_BIT,
                        VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_2_TRANSFER_BIT, 1);

    // Transition history: SHADER_READ_ONLY -> TRANSFER_DST
    device.imageBarrier(commandBuffer, historyImage.getImage(),
                        VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                        VK_ACCESS_2_SHADER_READ_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT,
                        VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_2_TRANSFER_BIT, 1);

    // Copy
    VkImageCopy region{};
    region.srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    region.dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    region.extent = {extent.width, extent.height, 1};
    vkCmdCopyImage(commandBuffer,
                   outputImage.getImage(), VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                   historyImage.getImage(), VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                   1, &region);

    // 6. Transition output back to GENERAL for downstream passes
    device.imageBarrier(commandBuffer, outputImage.getImage(),
                        VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_IMAGE_LAYOUT_GENERAL,
                        VK_ACCESS_2_TRANSFER_READ_BIT, VK_ACCESS_2_SHADER_READ_BIT,
                        VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, 1);

    state = pendingMode;
    return getOutputSlot();
}

PassImageSlot TAAPass::getOutputSlot() const
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
