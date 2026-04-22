#pragma once

#include "pass_base.h"
#include "pipeline.h"
#include "resource.h"
#include "ray_tracing_model.h"

#include <memory>

struct PTUniform
{
    alignas(16) glm::mat4 viewInverse;
    alignas(16) glm::mat4 projInverse;
};

struct PTPushConstants
{
    int frameIndex{0};
    int maxDepth{8};
    int rrDepth{3};
    int lightCount{0};
    int useNEE{1};
    int useMIS{1};
};

class PathTracingPass : public PassBase
{
    REGISTER_RENDER_PASS(PathTracingPass);

private:
    std::unique_ptr<RayTracingPipeline> rtPipeline;

    ImageResource colorImage;
    PTUniform ubo;
    UniformBufferResource uniformBuffer;
    PTPushConstants pushConstants;

    Camera *camera{nullptr};
    GBuffer *gbuffer{nullptr};
    RayTracingModel *scene{nullptr};
    bool firstFrame{true};

public:
    PathTracingPass(Device &_d, SwapChain &_sc, const json &params);

    std::string getName() const override { return "PathTracing"; }
    bool canDisable() const override { return false; }
    PassImageSlot getOutputSlot() const override;

    void setCamera(Camera *c) override { camera = c; }
    void setGBuffer(GBuffer *gb) override { gbuffer = gb; }
    void setScene(RayTracingModel *s) override { scene = s; }

    void init() override;
    void drawUI() override;
    void update(uint32_t currentFrame, InputState &inputState) override;
    PassImageSlot recordCommand(VkCommandBuffer commandBuffer,
                                const PassImageSlot &inputSlot,
                                uint32_t currentFrame, uint32_t imageIndex) override;
};
