#include <iostream>
#include <fstream>
#include <memory>
#include <vector>

#include "window.h"
#include "device.h"
#include "resource.h"
#include "swap_chain.h"
#include "pipeline.h"
#include "imgui_renderer.h"
#include "pass_base.h"

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
    std::vector<std::shared_ptr<PassBase>> passes;
    std::unordered_map<std::string, std::shared_ptr<PassBase>> passMap;
    uint32_t currentFrame{0};

public:
    Application(const std::string &configPath)
    {
        commandBuffers.resize(MAX_FRAMES_IN_FLIGHT);

        VkCommandBufferAllocateInfo allocInfo{};
        allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        allocInfo.commandPool = device.getCommandPool();
        allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        allocInfo.commandBufferCount = (uint32_t)commandBuffers.size();

        if (vkAllocateCommandBuffers(device.getDevice(), &allocInfo, commandBuffers.data()) != VK_SUCCESS)
            throw std::runtime_error("failed to allocate command buffers!");

        RenderPassFactory::printRegistered();
        loadConfig(configPath);
    }

    void loadConfig(const std::string &configPath)
    {
        std::ifstream file(configPath);
        if (!file.is_open())
            throw std::runtime_error("failed to open config: " + configPath);

        json config = json::parse(file);

        passes.clear();
        passMap.clear();

        // Phase 1: Create all passes
        for (auto &passConfig : config["passes"])
        {
            std::string type = passConfig.at("type");
            std::string name = passConfig.value("name", type);
            json params = passConfig.value("params", json::object());

            if (passMap.count(name))
                throw std::runtime_error("duplicate pass name: '" + name + "'");

            auto pass = RenderPassFactory::createPass(type, device, *swapChain, params);
            passes.push_back(pass);
            passMap[name] = pass;
        }

        // Phase 2: Wire inputs
        for (auto &passConfig : config["passes"])
        {
            if (!passConfig.contains("inputs"))
                continue;

            std::string name = passConfig.value("name", passConfig.at("type").get<std::string>());
            auto &pass = passMap.at(name);

            for (auto &[inputSlot, sourceRef] : passConfig["inputs"].items())
            {
                std::string src = sourceRef.get<std::string>();
                auto dotPos = src.find('.');
                if (dotPos == std::string::npos)
                    throw std::runtime_error("invalid input reference '" + src + "', expected 'passName.slotName'");

                std::string srcPass = src.substr(0, dotPos);
                std::string srcSlot = src.substr(dotPos + 1);

                auto it = passMap.find(srcPass);
                if (it == passMap.end())
                    throw std::runtime_error("input references unknown pass '" + srcPass + "'");

                pass->setInput(inputSlot, it->second->getOutput(srcSlot));
            }
        }

        // Phase 3: Init all passes (creates pipelines after wiring)
        for (auto &pass : passes)
            pass->init();
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

        // Update all passes
        for (auto &pass : passes)
            if (pass->isEnabled())
                pass->update(currentFrame, inputState);

        // ImGui
        ImGui_ImplVulkan_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();

        ImGui::SetNextWindowPos(ImVec2(50, 50), ImGuiCond_Once);
        ImGui::SetNextWindowSize(ImVec2(300, 300), ImGuiCond_Once);
        ImGui::Begin("VKSRT");
        ImGui::Text("FPS: %.1f", ImGui::GetIO().Framerate);
        ImGui::Separator();
        for (auto &pass : passes)
        {
            if (ImGui::CollapsingHeader(pass->getName().c_str(), ImGuiTreeNodeFlags_DefaultOpen))
            {
                ImGui::PushID(pass->getName().c_str());
                if (pass->isEnabled())
                    pass->drawUI();
                ImGui::PopID();
            }
        }
        ImGui::End();

        ImGui::Render();

        // Record pass commands
        vkResetCommandBuffer(commandBuffers[currentFrame], 0);

        VkCommandBufferBeginInfo beginInfo{};
        beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        if (vkBeginCommandBuffer(commandBuffers[currentFrame], &beginInfo) != VK_SUCCESS)
            throw std::runtime_error("failed to begin recording command buffer!");

        for (auto &pass : passes)
            if (pass->isEnabled())
                pass->recordCommand(commandBuffers[currentFrame], currentFrame, imageIndex);

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

        for (auto &pass : passes)
            if (pass->isEnabled())
                pass->endFrame();

        window.resetInputState();
        currentFrame = (currentFrame + 1) % MAX_FRAMES_IN_FLIGHT;
    }
};

int main(int argc, char *argv[])
{
    std::string configPath = "../config.json";
    for (int i = 1; i < argc; i++)
    {
        std::string arg = argv[i];
        if (arg == "--config" && i + 1 < argc)
            configPath = argv[++i];
        else
            configPath = arg;
    }

    try
    {
        Application app(configPath);
        app.run();
    }
    catch (const std::exception &e)
    {
        std::cerr << e.what() << std::endl;
        return 1;
    }

    return 0;
}
