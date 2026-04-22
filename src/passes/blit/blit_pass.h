#pragma once

#include "pass_base.h"
#include "pipeline.h"
#include "resource.h"

#include <memory>

class BlitPass : public PassBase
{
    REGISTER_RENDER_PASS(BlitPass);

private:
    std::unique_ptr<GraphicsPipeline> blitPipeline;

    // Offline mode: render to this image instead of swapchain
    std::unique_ptr<ImageResource> offlineImage;
    bool offlineMode{false};

public:
    BlitPass(Device &_d, SwapChain &_sc, const json &params);

    std::string getName() const override { return "Blit"; }
    bool canDisable() const override { return false; }
    PassImageSlot getOutputSlot() const override;

    void setOfflineMode(bool offline);
    void init() override;
    void drawUI() override;
    PassImageSlot recordCommand(VkCommandBuffer commandBuffer,
                                const PassImageSlot &inputSlot,
                                uint32_t currentFrame, uint32_t imageIndex) override;
};
