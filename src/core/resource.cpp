#include "resource.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

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
    update(data, size, 0);
}

void StorageBufferResource::update(const void *data, VkDeviceSize srcSize, VkDeviceSize dstOffset)
{
    VkBuffer stagingBuffer;
    VkDeviceMemory stagingBufferMemory;
    device.createBuffer(srcSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                        stagingBuffer, stagingBufferMemory);

    void *dst;
    vkMapMemory(device.getDevice(), stagingBufferMemory, 0, srcSize, 0, &dst);
    memcpy(dst, data, (size_t)srcSize);
    vkUnmapMemory(device.getDevice(), stagingBufferMemory);

    VkCommandBuffer cmd = device.beginSingleTimeCommands();
    VkBufferCopy region{};
    region.srcOffset = 0;
    region.dstOffset = dstOffset;
    region.size = srcSize;
    vkCmdCopyBuffer(cmd, stagingBuffer, buffer, 1, &region);
    device.endSingleTimeCommands(cmd);

    vkDestroyBuffer(device.getDevice(), stagingBuffer, nullptr);
    vkFreeMemory(device.getDevice(), stagingBufferMemory, nullptr);
}

void StorageBufferResource::download(void *data) const
{
    download(data, size, 0);
}

void StorageBufferResource::download(void *data, VkDeviceSize dstSize, VkDeviceSize srcOffset) const
{
    VkBuffer stagingBuffer;
    VkDeviceMemory stagingBufferMemory;
    device.createBuffer(dstSize, VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                        stagingBuffer, stagingBufferMemory);

    VkCommandBuffer cmd = device.beginSingleTimeCommands();
    VkBufferCopy region{};
    region.srcOffset = srcOffset;
    region.dstOffset = 0;
    region.size = dstSize;
    vkCmdCopyBuffer(cmd, buffer, stagingBuffer, 1, &region);
    device.endSingleTimeCommands(cmd);

    void *src;
    vkMapMemory(device.getDevice(), stagingBufferMemory, 0, dstSize, 0, &src);
    memcpy(data, src, (size_t)dstSize);
    vkUnmapMemory(device.getDevice(), stagingBufferMemory);

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
