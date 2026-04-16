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
#include "blit_pass.h"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <algorithm>

#include "stb_image_write.h"

static constexpr int MAX_FRAMES_IN_FLIGHT = 2;
static constexpr int WIDTH = 1600, HEIGHT = 1200;

struct OfflineConfig
{
    bool enabled{false};
    int samples{0};
    std::string outputFilename{"output"};
};

class Application
{
public:
    OfflineConfig offlineConfig;

    Window window;
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
    Application(const std::string &configPath, const OfflineConfig &offline = {})
        : offlineConfig(offline),
          window(WIDTH, HEIGHT, "VKSRT", !offline.enabled),
          camera({0.5f, -2.0f, 0.5f}, {0.5f, 0.0f, 0.5f},
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

        // Pipeline::loadPipelineCache(device, "../../build/pipeline_cache.bin");

        RenderPassFactory::printRegistered();
        loadConfig(configPath);
    }

    void loadConfig(const std::string &configPath)
    {
        std::ifstream file(configPath);
        if (!file.is_open())
            throw std::runtime_error("failed to open config: " + configPath);

        json config = json::parse(file);

        // Parse optional camera config
        if (config.contains("camera"))
        {
            auto &cam = config["camera"];
            if (cam.contains("pos"))
            {
                auto &p = cam["pos"];
                camera.pos = {p[0].get<float>(), p[1].get<float>(), p[2].get<float>()};
            }
            if (cam.contains("target"))
            {
                auto &t = cam["target"];
                camera.target = {t[0].get<float>(), t[1].get<float>(), t[2].get<float>()};
            }
            if (cam.contains("fov"))
                camera.fov = cam["fov"].get<float>();

            // Recompute direction vectors from updated pos/target
            camera.front = glm::normalize(camera.target - camera.pos);
            camera.right = glm::normalize(glm::cross(camera.front, Camera::worldUp));
            camera.up = glm::normalize(glm::cross(camera.right, camera.front));
        }

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
        if (offlineConfig.enabled)
        {
            runOffline();
            return;
        }
        while (!window.shouldClose())
        {
            window.pollEvents();
            drawFrame();
        }
        vkDeviceWaitIdle(device.getDevice());
    }

    void runOffline()
    {
        // Set blit pass to offline mode (renders to dedicated image with TRANSFER_SRC)
        for (auto &pass : passes)
            if (auto *blit = dynamic_cast<BlitPass *>(pass.get()))
                blit->setOfflineMode(true);

        // Create a fence for CPU-GPU sync (no swapchain semaphores)
        VkFenceCreateInfo fenceInfo{};
        fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
        VkFence fence;
        if (vkCreateFence(device.getDevice(), &fenceInfo, nullptr, &fence) != VK_SUCCESS)
            throw std::runtime_error("failed to create offline fence!");

        int totalSamples = offlineConfig.samples;
        InputState dummyInput{}; // all zeros, isChanged() always false

        auto startTime = std::chrono::high_resolution_clock::now();
        PassImageSlot lastSlot{};

        for (int sample = 0; sample < totalSamples; sample++)
        {
            // Update passes with dummy input (no camera movement)
            for (auto &pass : passes)
                pass->update(currentFrame, dummyInput);

            // Record pass commands (no ImGui, no swapchain)
            vkResetCommandBuffer(commandBuffers[currentFrame], 0);

            VkCommandBufferBeginInfo beginInfo{};
            beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
            beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
            if (vkBeginCommandBuffer(commandBuffers[currentFrame], &beginInfo) != VK_SUCCESS)
                throw std::runtime_error("failed to begin recording command buffer!");

            lastSlot = {};
            for (auto &pass : passes)
                lastSlot = pass->recordCommand(commandBuffers[currentFrame], lastSlot, currentFrame, 0);

            if (vkEndCommandBuffer(commandBuffers[currentFrame]) != VK_SUCCESS)
                throw std::runtime_error("failed to record command buffer!");

            // Submit with fence only — no semaphores, no present
            VkSubmitInfo submitInfo{};
            submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
            submitInfo.commandBufferCount = 1;
            submitInfo.pCommandBuffers = &commandBuffers[currentFrame];

            vkResetFences(device.getDevice(), 1, &fence);
            if (vkQueueSubmit(device.getGraphicsQueue(), 1, &submitInfo, fence) != VK_SUCCESS)
                throw std::runtime_error("failed to submit offline command buffer!");
            vkWaitForFences(device.getDevice(), 1, &fence, VK_TRUE, UINT64_MAX);

            camera.updatePrevMatrices();
            for (auto &pass : passes)
                pass->endFrame();

            currentFrame = (currentFrame + 1) % MAX_FRAMES_IN_FLIGHT;

            // Progress output
            auto now = std::chrono::high_resolution_clock::now();
            float elapsed = std::chrono::duration<float>(now - startTime).count();
            float perSample = elapsed / (sample + 1);
            float eta = perSample * (totalSamples - sample - 1);
            int etaMin = (int)(eta / 60.0f);
            int etaSec = (int)eta % 60;
            printf("\rRendering: %d/%d samples (%.1f%%) ETA: %dm%02ds   ",
                   sample + 1, totalSamples,
                   100.0f * (sample + 1) / totalSamples,
                   etaMin, etaSec);
            fflush(stdout);
        }
        auto endTime = std::chrono::high_resolution_clock::now();
        float totalTime = std::chrono::duration<float>(endTime - startTime).count();
        printf("\nRendering done: %d samples in %.2fs (%.2f samples/s)\n",
               totalSamples, totalTime, totalSamples / totalTime);

        if (lastSlot.image == VK_NULL_HANDLE)
            throw std::runtime_error("no valid pass output for offline save!");

        saveOfflineImage(lastSlot, offlineConfig.outputFilename);

        vkDeviceWaitIdle(device.getDevice());
        vkDestroyFence(device.getDevice(), fence, nullptr);
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

    void saveOfflineImage(const PassImageSlot &slot, const std::string &outputName)
    {
        vkDeviceWaitIdle(device.getDevice());

        uint32_t width = slot.extent.width;
        uint32_t height = slot.extent.height;
        bool isFloat = (slot.format == VK_FORMAT_R32G32B32A32_SFLOAT);
        uint32_t srcBytesPerPixel = isFloat ? 16 : 4;
        VkDeviceSize bufferSize = (VkDeviceSize)width * height * srcBytesPerPixel;

        VkBuffer stagingBuffer;
        VkDeviceMemory stagingBufferMemory;
        device.createBuffer(bufferSize, VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                            stagingBuffer, stagingBufferMemory);

        // Image is already in TRANSFER_SRC_OPTIMAL (transitioned in last frame's cmd buffer)
        VkCommandBuffer cmd = device.beginSingleTimeCommands();
        device.copyImageToBuffer(cmd, slot.image, stagingBuffer, width, height, srcBytesPerPixel);
        device.endSingleTimeCommands(cmd);

        void *mapped;
        vkMapMemory(device.getDevice(), stagingBufferMemory, 0, bufferSize, 0, &mapped);

        std::vector<uint8_t> pixels(width * height * 4);

        if (isFloat)
        {
            // RGBA32F -> 8-bit sRGB PNG
            const float *src = static_cast<const float *>(mapped);
            for (uint32_t i = 0; i < width * height; i++)
            {
                for (int c = 0; c < 3; c++)
                {
                    float v = std::max(0.0f, std::min(1.0f, src[i * 4 + c]));
                    // Linear to sRGB
                    if (v <= 0.0031308f)
                        v = 12.92f * v;
                    else
                        v = 1.055f * std::pow(v, 1.0f / 2.4f) - 0.055f;
                    pixels[i * 4 + c] = (uint8_t)(v * 255.0f + 0.5f);
                }
                pixels[i * 4 + 3] = 255;
            }
        }
        else
        {
            memcpy(pixels.data(), mapped, width * height * 4);
            // Handle BGRA formats
            if (slot.format == VK_FORMAT_B8G8R8A8_SRGB || slot.format == VK_FORMAT_B8G8R8A8_UNORM)
            {
                for (uint32_t i = 0; i < width * height; i++)
                    std::swap(pixels[i * 4 + 0], pixels[i * 4 + 2]);
            }
        }

        vkUnmapMemory(device.getDevice(), stagingBufferMemory);

        std::string filename = outputName + ".png";
        stbi_write_png(filename.c_str(), width, height, 4, pixels.data(), 4 * width);
        std::cout << "Saved: " << filename << " (" << width << "x" << height << ")" << std::endl;

        vkDestroyBuffer(device.getDevice(), stagingBuffer, nullptr);
        vkFreeMemory(device.getDevice(), stagingBufferMemory, nullptr);
    }
};

int main(int argc, char *argv[])
{
    std::string configPath = "../../config.json";
    OfflineConfig offline;

    for (int i = 1; i < argc; i++)
    {
        std::string arg = argv[i];
        if (arg == "--config" && i + 1 < argc)
            configPath = argv[++i];
        else if (arg == "--offline" && i + 1 < argc)
        {
            offline.enabled = true;
            offline.samples = std::atoi(argv[++i]);
        }
        else if (arg == "--output" && i + 1 < argc)
            offline.outputFilename = argv[++i];
        else
            configPath = arg;
    }

    try
    {
        Application app(configPath, offline);
        app.run();
    }
    catch (const std::exception &e)
    {
        std::cerr << e.what() << std::endl;
        return 1;
    }

    return 0;
}
