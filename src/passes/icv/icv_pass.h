#pragma once

#include "pass_base.h"
#include "pipeline.h"
#include "resource.h"
#include "ray_tracing_model.h"

#include <memory>

struct ICVUniform
{
    alignas(16) glm::mat4 viewInverse;
    alignas(16) glm::mat4 projInverse;
};

struct ICVPushConstants
{
    int frameIndex{0};
    int maxDepth{8};
    int rrDepth{3};
    int lightCount{0};
    int useNEE{1};
    int useMIS{1};
    float a{1.0f};
    float gaussianSigma{1.0f};
    int screenWidth{0};
    int screenHeight{0};
    int _pad0{0};
    int _pad1{0};
};
static_assert(sizeof(ICVPushConstants) == 48, "ICVPushConstants must be 48 bytes");

class ICVPass : public PassBase
{
    REGISTER_RENDER_PASS(ICVPass);

private:
    std::unique_ptr<RayTracingPipeline> anchorPipeline;
    std::unique_ptr<RayTracingPipeline> edgePipeline;
    std::unique_ptr<ComputePipeline> composePipeline;

    ImageResource anchorImage;
    ImageResource hEdgeLeft;
    ImageResource hEdgeRight;
    ImageResource vEdgeUp;
    ImageResource vEdgeDown;
    ImageResource outputImage;

    ICVUniform ubo;
    UniformBufferResource uniformBuffer;
    ICVPushConstants pushConstants;

    Camera *camera{nullptr};
    GBuffer *gbuffer{nullptr};
    RayTracingModel *scene{nullptr};
    bool firstFrame{true};

public:
    ICVPass(Device &_d, SwapChain &_sc, const json &params);

    std::string getName() const override { return "Image Space Control Variates"; }
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
