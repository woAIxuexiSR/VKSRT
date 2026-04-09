#include "resource.h"

ImageResource::ImageResource(Device &_d, VkFormat _f, VkExtent2D _e, VkImageUsageFlags usage,
                             VkImageAspectFlags _aspectMask, VkBool32 compareEnable, VkCompareOp compareOp)
    : device(_d), format(_f), extent(_e), aspectMask(_aspectMask)
{
    device.createImage(extent.width, extent.height, 1, 1, VK_SAMPLE_COUNT_1_BIT,
                       format, VK_IMAGE_TILING_OPTIMAL, usage,
                       VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, image, imageMemory);

    imageView = device.createImageView(image, VK_IMAGE_VIEW_TYPE_2D, format, aspectMask, 1);
    sampler = device.createSampler(VK_FILTER_LINEAR, VK_SAMPLER_ADDRESS_MODE_REPEAT, compareEnable, compareOp);
}

ImageResource::~ImageResource()
{
    vkDestroySampler(device.getDevice(), sampler, nullptr);
    vkDestroyImageView(device.getDevice(), imageView, nullptr);
    vkDestroyImage(device.getDevice(), image, nullptr);
    vkFreeMemory(device.getDevice(), imageMemory, nullptr);
}

StorageBufferResource::StorageBufferResource(Device &_d, VkDeviceSize _s, VkBufferUsageFlags usage)
    : device(_d), size(_s)
{
    device.createBuffer(size, usage, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, buffer, bufferMemory);
}

StorageBufferResource::~StorageBufferResource()
{
    vkDestroyBuffer(device.getDevice(), buffer, nullptr);
    vkFreeMemory(device.getDevice(), bufferMemory, nullptr);
}

void StorageBufferResource::update(void *data)
{
    VkBuffer stagingBuffer;
    VkDeviceMemory stagingBufferMemory;
    device.createBuffer(size, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, stagingBuffer, stagingBufferMemory);

    void *dst;
    vkMapMemory(device.getDevice(), stagingBufferMemory, 0, size, 0, &dst);
    memcpy(dst, data, (size_t)size);
    vkUnmapMemory(device.getDevice(), stagingBufferMemory);

    VkCommandBuffer cmd = device.beginSingleTimeCommands();
    device.copyBuffer(cmd, stagingBuffer, buffer, size);
    device.endSingleTimeCommands(cmd);

    vkDestroyBuffer(device.getDevice(), stagingBuffer, nullptr);
    vkFreeMemory(device.getDevice(), stagingBufferMemory, nullptr);
}

UniformBufferResource::UniformBufferResource(Device &_d, VkDeviceSize _s, VkBufferUsageFlags usage)
    : device(_d), size(_s)
{
    device.createBuffer(size, usage, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, buffer, bufferMemory);
    vkMapMemory(device.getDevice(), bufferMemory, 0, size, 0, &mappedMemory);
}

UniformBufferResource::~UniformBufferResource()
{
    vkUnmapMemory(device.getDevice(), bufferMemory);
    vkDestroyBuffer(device.getDevice(), buffer, nullptr);
    vkFreeMemory(device.getDevice(), bufferMemory, nullptr);
}
