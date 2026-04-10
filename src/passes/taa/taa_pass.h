#pragma once

#include "pass_base.h"
#include "pipeline.h"
#include "resource.h"

#include <glm/glm.hpp>
#include <memory>

struct TAAPushConstants
{
    float blendFactor{0.1f};
    int frameIndex{0};
    int useAccumMode{0}; // 0=TAA, 1=accumulate (static camera fallback)
    int _pad0{0};
};

struct TAAUniform
{
    alignas(16) glm::mat4 prevViewProj;
    alignas(4) int screenWidth;
    alignas(4) int screenHeight;
};

class TAAPass : public PassBase
{
    REGISTER_RENDER_PASS(TAAPass);

private:
    std::unique_ptr<ComputePipeline> taaPipeline;
    ImageResource historyImage;
    ImageResource outputImage;
    TAAPushConstants pushConstants;
    TAAUniform taaUniform;
    UniformBufferResource uniformBuffer;
    bool firstFrame{true};
    bool wasInteracting{false};
    bool wasEnabled{false};
    VkImageView lastBoundInputView{VK_NULL_HANDLE};
    VkSampler lastBoundInputSampler{VK_NULL_HANDLE};

public:
    TAAPass(Device &_d, SwapChain &_sc, const json &params);

    std::string getName() const override { return "TAA"; }
    PassImageSlot getOutputSlot() const override;

    void init() override;
    void drawUI() override;
    void update(uint32_t currentFrame, InputState &inputState) override;
    PassImageSlot recordCommand(VkCommandBuffer commandBuffer,
                                const PassImageSlot &inputSlot,
                                uint32_t currentFrame, uint32_t imageIndex) override;
};
