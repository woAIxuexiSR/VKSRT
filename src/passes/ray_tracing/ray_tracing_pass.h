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
    alignas(4) int frameIndex{0};
    alignas(4) int _pad0{0};
    alignas(4) int _pad1{0};
    alignas(4) int _pad2{0};
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

    void init() override;
    void drawUI() override;
    void update(uint32_t currentFrame, InputState &inputState) override;
    void recordCommand(VkCommandBuffer commandBuffer,
                       uint32_t currentFrame, uint32_t imageIndex) override;
};
