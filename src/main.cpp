#include <iostream>
#include <memory>
#include <vector>

#include "window.h"
#include "device.h"
#include "resource.h"
#include "swap_chain.h"
#include "pipeline.h"
#include "imgui_renderer.h"
#include "pass_base.h"
#include "ray_tracing/ray_tracing_pass.h"

static constexpr int MAX_FRAMES_IN_FLIGHT = 2;
static constexpr int WIDTH = 1600, HEIGHT = 1200;

class Application
{
public:
    Window window{WIDTH, HEIGHT, "VKSRT"};
    Device device{window};

    std::unique_ptr<SwapChain> swapChain{std::make_unique<SwapChain>(device, window.getExtent(), MAX_FRAMES_IN_FLIGHT)};
    ImGUIRenderer imguiRenderer{device, window, *swapChain, MAX_FRAMES_IN_FLIGHT};

    std::vector<VkCommandBuffer> commandBuffers;
    std::unique_ptr<PassBase> activePass;
    uint32_t currentFrame{0};

public:
    Application()
    {
        commandBuffers.resize(MAX_FRAMES_IN_FLIGHT);

        VkCommandBufferAllocateInfo allocInfo{};
        allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        allocInfo.commandPool = device.getCommandPool();
        allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        allocInfo.commandBufferCount = (uint32_t)commandBuffers.size();

        if (vkAllocateCommandBuffers(device.getDevice(), &allocInfo, commandBuffers.data()) != VK_SUCCESS)
            throw std::runtime_error("failed to allocate command buffers!");

        activePass = std::make_unique<RayTracingPass>(device, *swapChain);
    }

    void run()
    {
        while (!window.shouldClose())
        {
            window.pollEvents();
            drawFrame();
        }
        vkDeviceWaitIdle(device.getDevice());
    }

private:
    void recreateSwapChain()
    {
        window.checkIdle();
        vkDeviceWaitIdle(device.getDevice());

        auto extent = window.getExtent();
        if (swapChain == nullptr)
            swapChain = std::make_unique<SwapChain>(device, extent, MAX_FRAMES_IN_FLIGHT);
        else
            swapChain = std::make_unique<SwapChain>(device, extent, MAX_FRAMES_IN_FLIGHT, swapChain->getSwapChain());
    }

    void drawFrame()
    {
        uint32_t imageIndex;
        VkResult result = swapChain->acquireNextImage(currentFrame, &imageIndex);

        if (result == VK_ERROR_OUT_OF_DATE_KHR)
        {
            recreateSwapChain();
            return;
        }
        else if (result != VK_SUCCESS && result != VK_SUBOPTIMAL_KHR)
            throw std::runtime_error("failed to acquire swap chain image!");

        auto &inputState = window.getInputState();
        activePass->update(currentFrame, inputState);

        // ImGui
        ImGui_ImplVulkan_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();

        ImGui::SetNextWindowPos(ImVec2(50, 50), ImGuiCond_Once);
        ImGui::SetNextWindowSize(ImVec2(300, 200), ImGuiCond_Once);
        ImGui::Begin("VKSRT");
        ImGui::Text("FPS: %.1f", ImGui::GetIO().Framerate);
        ImGui::Text("Pass: %s", activePass->getName().c_str());
        activePass->drawUI();
        ImGui::End();

        ImGui::Render();

        // Record pass command
        vkResetCommandBuffer(commandBuffers[currentFrame], 0);

        VkCommandBufferBeginInfo beginInfo{};
        beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        if (vkBeginCommandBuffer(commandBuffers[currentFrame], &beginInfo) != VK_SUCCESS)
            throw std::runtime_error("failed to begin recording command buffer!");

        activePass->recordCommand(commandBuffers[currentFrame], *swapChain, currentFrame, imageIndex);

        if (vkEndCommandBuffer(commandBuffers[currentFrame]) != VK_SUCCESS)
            throw std::runtime_error("failed to record command buffer!");

        // Submit
        result = swapChain->submitCommandBuffer(
            {commandBuffers[currentFrame],
             imguiRenderer.prepareCommandBuffer(*swapChain, currentFrame, imageIndex)},
            currentFrame, imageIndex);

        if (result == VK_ERROR_OUT_OF_DATE_KHR || result == VK_SUBOPTIMAL_KHR || inputState.framebufferResized)
        {
            window.resetInputState();
            recreateSwapChain();
        }
        else if (result != VK_SUCCESS)
            throw std::runtime_error("failed to present swap chain image!");

        activePass->endFrame();
        window.resetInputState();
        currentFrame = (currentFrame + 1) % MAX_FRAMES_IN_FLIGHT;
    }
};

int main()
{
    try
    {
        Application app;
        app.run();
    }
    catch (const std::runtime_error &e)
    {
        std::cerr << e.what() << std::endl;
        return 1;
    }

    return 0;
}
