#pragma once

#include "device.h"
#include "resource.h"

#include <algorithm>
#include <string>
#include <vector>
#include <memory>

class SwapChain
{
private:
    Device &device;
    VkExtent2D windowExtent;

    VkSwapchainKHR swapChain;
    VkSwapchainKHR oldSwapChain; // non-null only briefly during recreate(); passed as oldSwapchain to createSwapChain()

    VkFormat swapChainImageFormat;
    VkExtent2D swapChainExtent;
    std::vector<VkImage> swapChainImages;
    std::vector<VkImageView> swapChainImageViews;

    std::unique_ptr<ImageResource> depthResource;

    std::vector<VkSemaphore> imageAvailableSemaphores;
    std::vector<VkSemaphore> renderFinishedSemaphores;
    std::vector<VkFence> inFlightFences;

    uint32_t imageCount;
    uint32_t framesInFlight;

    void createSwapChain();
    void createImageViews();
    void createSyncObjects();

    VkSurfaceFormatKHR chooseSwapSurfaceFormat(const std::vector<VkSurfaceFormatKHR> &availableFormats) const;
    VkPresentModeKHR chooseSwapPresentMode(const std::vector<VkPresentModeKHR> &availablePresentModes) const;
    VkExtent2D chooseSwapExtent(const VkSurfaceCapabilitiesKHR &capabilities) const;

public:
    SwapChain(Device &_d, VkExtent2D _we, uint32_t _f);
    ~SwapChain();

    SwapChain(const SwapChain &) = delete;
    SwapChain &operator=(const SwapChain &) = delete;

    void recreate(VkExtent2D newExtent);

    VkSwapchainKHR getSwapChain() const { return swapChain; }
    VkFormat getImageFormat() const { return swapChainImageFormat; }
    VkExtent2D getExtent() const { return swapChainExtent; }
    VkImage getImage(int index) const { return swapChainImages[index]; }
    VkImageView getImageView(int index) const { return swapChainImageViews[index]; }
    VkFormat getDepthFormat() const { return depthResource->getFormat(); }
    VkImageView getDepthImageView() const { return depthResource->getImageView(); }
    size_t getImageCount() const { return imageCount; }

    VkResult acquireNextImage(uint32_t currentFrame, uint32_t *imageIndex);
    VkResult submitCommandBuffer(std::vector<VkCommandBuffer> commandBuffer, uint32_t currentFrame, uint32_t imageIndex);
};
