#pragma once

#include "device.h"
#include "swap_chain.h"

#include <string>

// Base class for all render passes in the chain.
// Each pass receives a VkImage as input and writes to a VkImage as output.
// The chain manages image transitions between passes.
class PassBase
{
protected:
    Device &device;

public:
    PassBase(Device &_d) : device(_d) {}
    virtual ~PassBase() = default;

    PassBase(const PassBase &) = delete;
    PassBase &operator=(const PassBase &) = delete;

    virtual std::string getName() const = 0;

    // Called once per frame before recordCommand
    virtual void update(uint32_t currentFrame, struct InputState &inputState) {}

    // Record rendering commands. The pass writes to the swapchain image at imageIndex.
    virtual void recordCommand(VkCommandBuffer commandBuffer, SwapChain &swapChain,
                               uint32_t currentFrame, uint32_t imageIndex) = 0;

    // Called after frame submission
    virtual void endFrame() {}

    // Draw ImGui UI for this pass
    virtual void drawUI() {}
};
