#pragma once

#include "device.h"

class ImageResource
{
private:
    Device &device;
    VkFormat format;
    VkExtent2D extent;
    VkImageAspectFlags aspectMask;

    VkImage image;
    VkDeviceMemory imageMemory;
    VkImageView imageView;
    VkSampler sampler;

public:
    ImageResource(Device &_d, VkFormat _f, VkExtent2D _e, VkImageUsageFlags usage,
                  VkImageAspectFlags _aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
                  VkBool32 compareEnable = VK_FALSE, VkCompareOp compareOp = VK_COMPARE_OP_NEVER);
    ~ImageResource();

    ImageResource(const ImageResource &) = delete;
    ImageResource &operator=(const ImageResource &) = delete;

    VkExtent2D getExtent() const { return extent; }
    VkFormat getFormat() const { return format; }
    VkImage getImage() const { return image; }
    VkImageView getImageView() const { return imageView; }
    VkSampler getSampler() const { return sampler; }
};

class StorageBufferResource
{
private:
    Device &device;
    VkDeviceSize size;
    VkBuffer buffer;
    VkDeviceMemory bufferMemory;

public:
    StorageBufferResource(Device &_d, VkDeviceSize _s, VkBufferUsageFlags usage);
    ~StorageBufferResource();

    StorageBufferResource(const StorageBufferResource &) = delete;
    StorageBufferResource &operator=(const StorageBufferResource &) = delete;

    void update(void *data);
    void update(const void *data, VkDeviceSize srcSize, VkDeviceSize dstOffset = 0);
    void download(void *data) const;
    void download(void *data, VkDeviceSize dstSize, VkDeviceSize srcOffset = 0) const;
    VkDeviceSize getSize() const { return size; }
    VkBuffer getBuffer() const { return buffer; }
};

class UniformBufferResource
{
private:
    Device &device;
    VkDeviceSize size;
    VkBuffer buffer;
    VkDeviceMemory bufferMemory;
    void *mappedMemory;

public:
    UniformBufferResource(Device &_d, VkDeviceSize _s, VkBufferUsageFlags usage);
    ~UniformBufferResource();

    UniformBufferResource(const UniformBufferResource &) = delete;
    UniformBufferResource &operator=(const UniformBufferResource &) = delete;

    void update(void *data) { memcpy(mappedMemory, data, (size_t)size); }
    VkBuffer getBuffer() const { return buffer; }
};
