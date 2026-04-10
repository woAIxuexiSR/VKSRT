#include "path_tracing_pass.h"

#include "camera.h"
#include "imgui.h"

REGISTER_RENDER_PASS_CPP(PathTracingPass, "path_tracing");

PathTracingPass::PathTracingPass(Device &_d, SwapChain &_sc, const json &params)
    : PassBase(_d, _sc),
      colorImage{_d, VK_FORMAT_R32G32B32A32_SFLOAT, _sc.getExtent(),
                 VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT},
      uniformBuffer{_d, sizeof(PTUniform), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT},
      model{_d}
{
    if (params.contains("maxDepth"))
        pushConstants.maxDepth = params["maxDepth"].get<int>();
    if (params.contains("rrDepth"))
        pushConstants.rrDepth = params["rrDepth"].get<int>();

    // Load scene from params
    if (params.contains("scene"))
    {
        auto &scene = params["scene"];
        std::string type = scene.value("type", "cornell_box");
        if (type == "model")
        {
            std::string path = scene.at("path");
            float scale = scene.value("scale", 1.0f);
            glm::vec3 offset = {0, 0, 0};
            if (scene.contains("offset"))
            {
                auto &o = scene["offset"];
                offset = {o[0].get<float>(), o[1].get<float>(), o[2].get<float>()};
            }
            SceneLoader::loadModel(path, model, scale, offset);
        }
        else
            SceneLoader::buildCornellBox(model);
    }
    else
        SceneLoader::buildCornellBox(model);

    model.buildAccelerationStructures();
    pushConstants.lightCount = model.getLightCount();
}

void PathTracingPass::init()
{
    std::vector<DescriptorLayoutBinding> bindings = {
        {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, VK_SHADER_STAGE_RAYGEN_BIT_KHR},
        {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, VK_SHADER_STAGE_RAYGEN_BIT_KHR},
    };
    auto modelBindings = model.getDescriptorLayoutBindings();
    bindings.insert(bindings.end(), modelBindings.begin(), modelBindings.end());

    rtPipeline = std::make_unique<RayTracingPipeline>(
        device, 1, bindings,
        "../shaders/path_tracing/path_tracing.spv",
        model.getHitSBTRecords(),
        "raygenMain", "missMain", "closestHitMain",
        sizeof(PTPushConstants));

    std::vector<std::vector<DescriptorInfo>> infos = {
        {VkDescriptorImageInfo{colorImage.getSampler(), colorImage.getImageView(), VK_IMAGE_LAYOUT_GENERAL}},
        {VkDescriptorBufferInfo{uniformBuffer.getBuffer(), 0, sizeof(PTUniform)}},
    };
    auto modelInfos = model.getDescriptorInfos();
    infos.insert(infos.end(), modelInfos.begin(), modelInfos.end());
    rtPipeline->updateDescriptorSets(infos);
}

void PathTracingPass::drawUI()
{
    ImGui::SliderInt("Max Depth", &pushConstants.maxDepth, 1, 32);
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
}

void PathTracingPass::update(uint32_t currentFrame, InputState &inputState)
{
    // Detect NEE/MIS parameter changes to trigger accumulate reset
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

PassImageSlot PathTracingPass::recordCommand(VkCommandBuffer commandBuffer,
                                              const PassImageSlot &inputSlot,
                                              uint32_t currentFrame, uint32_t imageIndex)
{
    auto colorExtent = colorImage.getExtent();

    VkImageLayout oldLayout = firstFrame ? VK_IMAGE_LAYOUT_UNDEFINED : VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    VkAccessFlags2 srcAccess = firstFrame ? (VkAccessFlags2)0 : VK_ACCESS_2_SHADER_READ_BIT;
    VkPipelineStageFlags2 srcStage = firstFrame ? VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT : VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;

    device.imageBarrier(commandBuffer, colorImage.getImage(),
                        oldLayout, VK_IMAGE_LAYOUT_GENERAL,
                        srcAccess, VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                        srcStage, VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR, 1);

    firstFrame = false;

    rtPipeline->bindPipeline(commandBuffer);
    rtPipeline->bindDescriptorSets(commandBuffer, currentFrame);
    rtPipeline->pushConstants(commandBuffer, &pushConstants);
    rtPipeline->traceRays(commandBuffer, {colorExtent.width, colorExtent.height, 1});

    return getOutputSlot();
}

PassImageSlot PathTracingPass::getOutputSlot() const
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
