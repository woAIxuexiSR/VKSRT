#pragma once

#include "device.h"
#include "window.h"
#include "swap_chain.h"

#define IMGUI_DEFINE_MATH_OPERATORS
#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_vulkan.h"

class ImGUIRenderer
{
private:
    uint32_t framesInFlight;
    Device &device;
    Window &window;

    VkDescriptorPool descriptorPool;
    std::vector<VkCommandBuffer> commandBuffers;

    void createDescriptorPool();
    void initImGUI(SwapChain &swapChain);

public:
    ImGUIRenderer(Device &_d, Window &_w, SwapChain &swapChain, uint32_t f);
    ~ImGUIRenderer();

    ImGUIRenderer(const ImGUIRenderer &) = delete;
    ImGUIRenderer &operator=(const ImGUIRenderer &) = delete;

    VkCommandBuffer prepareCommandBuffer(SwapChain &swapChain, int currentFrame, int imageIndex);
};
