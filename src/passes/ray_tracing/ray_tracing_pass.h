#pragma once

#include "pass_base.h"
#include "pipeline.h"
#include "resource.h"
#include "ray_tracing_model.h"

#include <memory>

struct RTUniform
{
    alignas(16) glm::mat4 viewInverse;
    alignas(16) glm::mat4 projInverse;
};

struct RTPushConstants
{
    int shadingMode{0}; // 0=Material, 1=Position, 2=Normal, 3=UV
};

class RayTracingPass : public PassBase
{
    REGISTER_RENDER_PASS(RayTracingPass);

private:
    std::unique_ptr<RayTracingPipeline> rtPipeline;

    ImageResource colorImage;
    RTUniform ubo;
    UniformBufferResource uniformBuffer;
    RTPushConstants pushConstants;

    Camera *camera{nullptr};
    RayTracingModel *scene{nullptr};
    bool firstFrame{true};

public:
    RayTracingPass(Device &_d, SwapChain &_sc, const json &params);

    std::string getName() const override { return "RayTracing"; }
    bool canDisable() const override { return false; }
    PassImageSlot getOutputSlot() const override;

    void setCamera(Camera *c) override { camera = c; }
    void setScene(RayTracingModel *s) override { scene = s; }

    void init() override;
    void drawUI() override;
    void update(uint32_t currentFrame, InputState &inputState) override;
    PassImageSlot recordCommand(VkCommandBuffer commandBuffer,
                                const PassImageSlot &inputSlot,
                                uint32_t currentFrame, uint32_t imageIndex) override;
};
