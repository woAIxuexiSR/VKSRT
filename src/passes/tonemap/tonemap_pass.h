#pragma once

#include "pass_base.h"
#include "pipeline.h"
#include "resource.h"

#include <memory>

struct TonemapPushConstants
{
    float exposure{1.0f};
};

class TonemapPass : public PassBase
{
    REGISTER_RENDER_PASS(TonemapPass);

private:
    std::unique_ptr<ComputePipeline> tonemapPipeline;
    ImageResource ldrImage;
    TonemapPushConstants pushConstants;

public:
    TonemapPass(Device &_d, SwapChain &_sc, const json &params);

    std::string getName() const override { return "Tonemap"; }

    void init() override;
    void drawUI() override;
    void recordCommand(VkCommandBuffer commandBuffer,
                       uint32_t currentFrame, uint32_t imageIndex) override;
};
