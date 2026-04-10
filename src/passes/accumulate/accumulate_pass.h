#pragma once

#include "pass_base.h"
#include "pipeline.h"
#include "resource.h"

#include <memory>

struct AccumulatePushConstants
{
    int frameIndex{0};
};

class AccumulatePass : public PassBase
{
    REGISTER_RENDER_PASS(AccumulatePass);

private:
    std::unique_ptr<ComputePipeline> accumulatePipeline;
    ImageResource accumImage;
    AccumulatePushConstants pushConstants;
    bool firstFrame{true};
    bool wasEnabled{true};
    VkImageView lastBoundInputView{VK_NULL_HANDLE};
    VkSampler lastBoundInputSampler{VK_NULL_HANDLE};

public:
    AccumulatePass(Device &_d, SwapChain &_sc, const json &params);

    std::string getName() const override { return "Accumulate"; }
    PassImageSlot getOutputSlot() const override;

    void init() override;
    void drawUI() override;
    void update(uint32_t currentFrame, InputState &inputState) override;
    PassImageSlot recordCommand(VkCommandBuffer commandBuffer,
                                const PassImageSlot &inputSlot,
                                uint32_t currentFrame, uint32_t imageIndex) override;
};
