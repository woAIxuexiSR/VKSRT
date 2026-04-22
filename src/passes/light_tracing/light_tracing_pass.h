#pragma once

#include "pass_base.h"
#include "pipeline.h"
#include "resource.h"
#include "ray_tracing_model.h"

#include <memory>

struct LTUniform
{
    alignas(16) glm::mat4 viewProj;
    alignas(16) glm::mat4 viewInverse;
    alignas(4) int screenWidth;
    alignas(4) int screenHeight;
    alignas(4) float tanHalfFov;
    alignas(4) float _pad0;
};

struct LTPushConstants
{
    int frameIndex{0};
    int maxDepth{8};
    int rrDepth{3};
    int lightCount{0};
    int photonSize{256};
};

class LightTracingPass : public PassBase
{
    REGISTER_RENDER_PASS(LightTracingPass);

private:
    std::unique_ptr<RayTracingPipeline> rtPipeline;
    std::unique_ptr<ComputePipeline> composePipeline;

    // Float atomic splat buffers (R32_SFLOAT x 3)
    ImageResource splatR, splatG, splatB;
    // Final output (R32G32B32A32_SFLOAT)
    ImageResource colorImage;

    LTUniform ubo;
    UniformBufferResource uniformBuffer;
    LTPushConstants pushConstants;

    Camera *camera{nullptr};
    RayTracingModel *scene{nullptr};
    bool firstFrame{true};

public:
    LightTracingPass(Device &_d, SwapChain &_sc, const json &params);

    std::string getName() const override { return "LightTracing"; }
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
