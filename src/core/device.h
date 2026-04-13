#pragma once

#include "window.h"

#include <iostream>
#include <string>
#include <vector>
#include <optional>
#include <set>

#ifdef NDEBUG
const bool enableValidationLayers = false;
#else
const bool enableValidationLayers = true;
#endif

const std::vector<const char *> validationLayers = {
    "VK_LAYER_KHRONOS_validation",
};

const std::vector<const char *> deviceExtensions = {
    VK_KHR_SWAPCHAIN_EXTENSION_NAME,
    VK_KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME,
    VK_KHR_BUFFER_DEVICE_ADDRESS_EXTENSION_NAME,
    VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME,
    VK_KHR_RAY_TRACING_PIPELINE_EXTENSION_NAME,
    VK_EXT_SHADER_ATOMIC_FLOAT_EXTENSION_NAME,
};

struct QueueFamilyIndices
{
    std::optional<uint32_t> graphicsFamily;
    std::optional<uint32_t> presentFamily;

    bool isComplete()
    {
        return graphicsFamily.has_value() && presentFamily.has_value();
    }
};

struct SwapChainSupportDetails
{
    VkSurfaceCapabilitiesKHR capabilities;
    std::vector<VkSurfaceFormatKHR> formats;
    std::vector<VkPresentModeKHR> presentModes;
};

class Device
{
private:
    uint32_t apiVersion;
    VkInstance instance;
    VkDebugUtilsMessengerEXT debugMessenger;
    VkSurfaceKHR surface;

    VkPhysicalDevice physicalDevice = VK_NULL_HANDLE;
    VkDevice device;
    VkPhysicalDeviceProperties physicalDeviceProperties;
    VkPhysicalDeviceRayTracingPipelinePropertiesKHR physicalDeviceRTPipelineProperties;
    VkPhysicalDeviceAccelerationStructurePropertiesKHR physicalDeviceASProperties;
    VkPhysicalDeviceAccelerationStructureFeaturesKHR physicalDeviceASFeatures;

    QueueFamilyIndices queueFamilyIndices;
    VkQueue graphicsQueue;
    VkQueue presentQueue;

    VkCommandPool commandPool;

    void createInstance();
    void setupDebugMessenger();
    void pickPhysicalDevice();
    void createLogicalDevice();
    void createCommandPool();

    bool checkValidationLayerSupport() const;
    std::vector<const char *> getRequiredExtensions() const;
    void populateDebugMessengerCreateInfo(VkDebugUtilsMessengerCreateInfoEXT &createInfo) const;
    bool isDeviceSuitable(VkPhysicalDevice pd) const;
    QueueFamilyIndices findQueueFamilies(VkPhysicalDevice pd) const;
    bool checkDeviceExtensionSupport(VkPhysicalDevice pd) const;
    SwapChainSupportDetails querySwapChainSupport(VkPhysicalDevice pd) const;

public:
    Device(Window &window);
    ~Device();

    Device(const Device &) = delete;
    Device &operator=(const Device &) = delete;

    VkInstance getInstance() const { return instance; }
    uint32_t getApiVersion() const { return apiVersion; }
    VkPhysicalDevice getPhysicalDevice() const { return physicalDevice; }
    QueueFamilyIndices getQueueFamilyIndices() const { return queueFamilyIndices; }

    VkDevice getDevice() const { return device; }
    VkSurfaceKHR getSurface() const { return surface; }
    VkCommandPool getCommandPool() const { return commandPool; }
    VkQueue getGraphicsQueue() const { return graphicsQueue; }
    VkQueue getPresentQueue() const { return presentQueue; }
    VkPhysicalDeviceProperties getPhysicalDeviceProperties() const { return physicalDeviceProperties; }
    VkPhysicalDeviceRayTracingPipelinePropertiesKHR getPhysicalDeviceRTPipelineProperties() const { return physicalDeviceRTPipelineProperties; }
    VkPhysicalDeviceAccelerationStructurePropertiesKHR getPhysicalDeviceASProperties() const { return physicalDeviceASProperties; }
    VkPhysicalDeviceAccelerationStructureFeaturesKHR getPhysicalDeviceASFeatures() const { return physicalDeviceASFeatures; }

    SwapChainSupportDetails getSwapChainSupport() const { return querySwapChainSupport(physicalDevice); }
    QueueFamilyIndices findPhysicalQueueFamilies() const { return findQueueFamilies(physicalDevice); }

    void createBuffer(
        VkDeviceSize size,
        VkBufferUsageFlags usage,
        VkMemoryPropertyFlags properties,
        VkBuffer &buffer,
        VkDeviceMemory &bufferMemory);
    void copyBuffer(VkCommandBuffer commandBuffer, VkBuffer srcBuffer, VkBuffer dstBuffer, VkDeviceSize size);
    void createImage(
        uint32_t width, uint32_t height, uint32_t mipLevels, uint32_t arrayLayers,
        VkSampleCountFlagBits numSamples, VkFormat format,
        VkImageTiling tiling, VkImageUsageFlags usage,
        VkMemoryPropertyFlags properties,
        VkImage &image,
        VkDeviceMemory &imageMemory);
    VkImageView createImageView(VkImage image, VkImageViewType viewType, VkFormat format, VkImageAspectFlags aspectFlags, uint32_t mipLevels, uint32_t layerCount = 1);
    VkSampler createSampler(VkFilter filter, VkSamplerAddressMode addressMode, VkBool32 compareEnable = VK_FALSE, VkCompareOp compareOp = VK_COMPARE_OP_NEVER);

    void copyImageToBuffer(VkCommandBuffer commandBuffer, VkImage image, VkBuffer buffer, uint32_t width, uint32_t height, uint32_t bytesPerPixel);
    void imageBarrier(VkCommandBuffer commandBuffer, VkImage image, VkImageLayout oldLayout, VkImageLayout newLayout,
                      VkAccessFlags2 srcAccessMask, VkAccessFlags2 dstAccessMask,
                      VkPipelineStageFlags2 sourceStage, VkPipelineStageFlags2 destinationStage,
                      uint32_t mipLevels, uint32_t layerCount = 1, VkImageAspectFlags aspectMask = VK_IMAGE_ASPECT_COLOR_BIT);
    void bindViewport(VkCommandBuffer commandBuffer, VkExtent2D extent);

    uint32_t findMemoryType(uint32_t typeFilter, VkMemoryPropertyFlags properties) const;
    VkFormat findSupportedFormat(const std::vector<VkFormat> &candidates, VkImageTiling tiling, VkFormatFeatureFlags features) const;
    VkFormat findDepthFormat() const;
    VkCommandBuffer beginSingleTimeCommands();
    void endSingleTimeCommands(VkCommandBuffer commandBuffer);

    template <typename T>
    T loadInstanceFunction(const std::string &name) const
    {
        auto func = reinterpret_cast<T>(vkGetInstanceProcAddr(instance, name.c_str()));
        if (func == nullptr)
            throw std::runtime_error("failed to load function " + name);
        return func;
    }
    template <typename T>
    T loadDeviceFunction(const std::string &name) const
    {
        auto func = reinterpret_cast<T>(vkGetDeviceProcAddr(device, name.c_str()));
        if (func == nullptr)
            throw std::runtime_error("failed to load function " + name);
        return func;
    }
};
