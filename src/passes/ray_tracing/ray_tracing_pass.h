#pragma once

#include "pass_base.h"
#include "pipeline.h"
#include "resource.h"
#include "ray_tracing_model.h"
#include "scene_loader.h"
#include "camera.h"

#include <memory>

struct RTUniform
{
    alignas(16) glm::mat4 viewInverse;
    alignas(16) glm::mat4 projInverse;
};

class RayTracingPass : public PassBase
{
    REGISTER_RENDER_PASS(RayTracingPass);

private:
    std::unique_ptr<RayTracingPipeline> rtPipeline;

    ImageResource colorImage;
    RTUniform ubo;
    UniformBufferResource uniformBuffer;

    RayTracingModel model;
    Camera camera;
    float lastTime{0.0f};
    bool firstFrame{true};

public:
    RayTracingPass(Device &_d, SwapChain &_sc, const json &params);

    std::string getName() const override { return "RayTracing"; }
    bool canDisable() const override { return false; }
    PassImageSlot getOutputSlot() const override;

    void init() override;
    void drawUI() override;
    void update(uint32_t currentFrame, InputState &inputState) override;
    PassImageSlot recordCommand(VkCommandBuffer commandBuffer,
                                const PassImageSlot &inputSlot,
                                uint32_t currentFrame, uint32_t imageIndex) override;
};
