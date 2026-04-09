#pragma once

#include "pass_base.h"
#include "pipeline.h"

#include <memory>

class BlitPass : public PassBase
{
    REGISTER_RENDER_PASS(BlitPass);

private:
    std::unique_ptr<GraphicsPipeline> blitPipeline;

public:
    BlitPass(Device &_d, SwapChain &_sc, const json &params);

    std::string getName() const override { return "Blit"; }

    void init() override;
    void recordCommand(VkCommandBuffer commandBuffer,
                       uint32_t currentFrame, uint32_t imageIndex) override;
};
