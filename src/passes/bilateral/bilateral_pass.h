#pragma once

#include "pass_base.h"
#include "pipeline.h"
#include "resource.h"

#include <memory>

struct BilateralPushConstants
{
    float sigmaS{3.0f};  // spatial sigma (pixels)
    float sigmaN{0.1f};  // normal threshold
    float sigmaP{0.1f};  // position threshold
    int kernelRadius{5};  // filter radius in pixels
};

class BilateralPass : public PassBase
{
    REGISTER_RENDER_PASS(BilateralPass);

private:
    std::unique_ptr<ComputePipeline> filterPipeline;
    ImageResource outputImage;
    BilateralPushConstants pushConstants;

    GBuffer *gbuffer{nullptr};

public:
    BilateralPass(Device &_d, SwapChain &_sc, const json &params);

    std::string getName() const override { return "Bilateral"; }
    PassImageSlot getOutputSlot() const override;

    void setGBuffer(GBuffer *gb) override { gbuffer = gb; }

    void init() override;
    void drawUI() override;
    PassImageSlot recordCommand(VkCommandBuffer commandBuffer,
                                const PassImageSlot &inputSlot,
                                uint32_t currentFrame, uint32_t imageIndex) override;
};
