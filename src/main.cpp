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
#include "camera.h"
#include "gbuffer.h"

#include <chrono>
#include <cstring>
#include <algorithm>

#include "stb_image_write.h"

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
    Camera camera;
    std::unique_ptr<GBuffer> gbuffer;
    float lastTime{0.0f};
    uint32_t currentFrame{0};
    bool saveRequested{false};
    char saveFilename[128] = "screenshot";

public:
    Application(const std::string &configPath)
        : camera({0.5f, -2.0f, 0.5f}, {0.5f, 0.0f, 0.5f},
                 (float)WIDTH / HEIGHT, 45.0f, 0.1f, 100.0f)
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

        // Phase 1: Create all passes
        for (auto &passConfig : config["passes"])
        {
            std::string type = passConfig.at("type");
            json params = passConfig.value("params", json::object());

            auto pass = RenderPassFactory::createPass(type, device, *swapChain, params);
            passes.push_back(pass);
        }

        // Create G-buffer (App-managed, shared across passes)
        gbuffer = std::make_unique<GBuffer>(device, swapChain->getExtent());

        // Phase 2: Wire input slots, G-buffer, inject camera, and init passes (chain order)
        PassImageSlot prevSlot{};
        for (auto &pass : passes)
        {
            pass->setCamera(&camera);
            pass->setGBuffer(gbuffer.get());
            pass->setInputSlot(prevSlot);
            pass->init();
            prevSlot = pass->getOutputSlot();
        }
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

        // Update camera
        auto now = std::chrono::high_resolution_clock::now();
        static auto startTime = now;
        float time = std::chrono::duration<float>(now - startTime).count();
        camera.processInput(inputState, time - lastTime);
        lastTime = time;

        // Update all passes (always called, even when disabled, for state tracking)
        for (auto &pass : passes)
            pass->update(currentFrame, inputState);

        // ImGui
        ImGui_ImplVulkan_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();

        ImGui::SetNextWindowPos(ImVec2(50, 50), ImGuiCond_Once);
        ImGui::SetNextWindowSize(ImVec2(450, 450), ImGuiCond_Once);
        ImGui::Begin("VKSRT");
        ImGui::Text("FPS: %.1f", ImGui::GetIO().Framerate);
        if (ImGui::Button("Save Image"))
            saveRequested = true;
        ImGui::SameLine();
        ImGui::PushItemWidth(150);
        ImGui::InputText("##filename", saveFilename, sizeof(saveFilename));
        ImGui::PopItemWidth();
        ImGui::Separator();
        if (ImGui::CollapsingHeader("Camera", ImGuiTreeNodeFlags_DefaultOpen))
        {
            ImGui::Text("Pos: (%.2f, %.2f, %.2f)", camera.pos.x, camera.pos.y, camera.pos.z);
            ImGui::Text("Yaw: %.1f  Pitch: %.1f", camera.yaw, camera.pitch);
            ImGui::Text("FOV: %.1f", camera.fov);
        }
        ImGui::Separator();
        for (auto &pass : passes)
        {
            if (ImGui::CollapsingHeader(pass->getName().c_str(), ImGuiTreeNodeFlags_DefaultOpen))
            {
                ImGui::PushID(pass->getName().c_str());
                pass->renderUI();
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

        PassImageSlot slot{};
        for (auto &pass : passes)
            slot = pass->recordCommand(commandBuffers[currentFrame], slot, currentFrame, imageIndex);

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

        if (saveRequested && result == VK_SUCCESS)
        {
            saveImage(imageIndex);
            saveRequested = false;
        }

        camera.updatePrevMatrices();

        for (auto &pass : passes)
            if (pass->isEnabled())
                pass->endFrame();

        window.resetInputState();
        currentFrame = (currentFrame + 1) % MAX_FRAMES_IN_FLIGHT;
    }

    void saveImage(uint32_t imageIndex)
    {
        vkDeviceWaitIdle(device.getDevice());

        auto extent = swapChain->getExtent();
        uint32_t width = extent.width;
        uint32_t height = extent.height;
        uint32_t bytesPerPixel = 4;
        VkDeviceSize imageSize = width * height * bytesPerPixel;

        VkBuffer stagingBuffer;
        VkDeviceMemory stagingBufferMemory;
        device.createBuffer(imageSize, VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                            stagingBuffer, stagingBufferMemory);

        VkCommandBuffer cmd = device.beginSingleTimeCommands();

        device.imageBarrier(cmd, swapChain->getImage(imageIndex),
                            VK_IMAGE_LAYOUT_PRESENT_SRC_KHR, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                            0, VK_ACCESS_2_TRANSFER_READ_BIT,
                            VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_2_TRANSFER_BIT, 1);

        device.copyImageToBuffer(cmd, swapChain->getImage(imageIndex), stagingBuffer, width, height, bytesPerPixel);

        device.endSingleTimeCommands(cmd);

        void *mapped;
        std::vector<uint8_t> pixels(imageSize);
        vkMapMemory(device.getDevice(), stagingBufferMemory, 0, imageSize, 0, &mapped);
        memcpy(pixels.data(), mapped, static_cast<size_t>(imageSize));
        vkUnmapMemory(device.getDevice(), stagingBufferMemory);

        // Swapchain is B8G8R8A8: swap B and R channels
        VkFormat format = swapChain->getImageFormat();
        if (format == VK_FORMAT_B8G8R8A8_SRGB || format == VK_FORMAT_B8G8R8A8_UNORM)
        {
            for (uint32_t i = 0; i < width * height; i++)
                std::swap(pixels[i * 4 + 0], pixels[i * 4 + 2]);
        }

        std::string filename = std::string(saveFilename) + ".png";
        stbi_write_png(filename.c_str(), width, height, bytesPerPixel, pixels.data(), bytesPerPixel * width);
        std::cout << "Saved: " << filename << std::endl;

        vkDestroyBuffer(device.getDevice(), stagingBuffer, nullptr);
        vkFreeMemory(device.getDevice(), stagingBufferMemory, nullptr);
    }
};

int main(int argc, char *argv[])
{
    std::string configPath = "../../config.json";
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
