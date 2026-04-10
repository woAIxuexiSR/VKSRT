#pragma once

#include "resource.h"

class GBuffer
{
private:
    ImageResource positionImage;
    ImageResource normalImage;
    ImageResource albedoImage;
    bool written{false};

public:
    GBuffer(Device &device, VkExtent2D extent)
        : positionImage{device, VK_FORMAT_R32G32B32A32_SFLOAT, extent,
                        VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT},
          normalImage{device, VK_FORMAT_R32G32B32A32_SFLOAT, extent,
                      VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT},
          albedoImage{device, VK_FORMAT_R32G32B32A32_SFLOAT, extent,
                      VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT}
    {
    }

    VkImage getPositionImage() const { return positionImage.getImage(); }
    VkImageView getPositionImageView() const { return positionImage.getImageView(); }
    VkSampler getPositionSampler() const { return positionImage.getSampler(); }

    VkImage getNormalImage() const { return normalImage.getImage(); }
    VkImageView getNormalImageView() const { return normalImage.getImageView(); }
    VkSampler getNormalSampler() const { return normalImage.getSampler(); }

    VkImage getAlbedoImage() const { return albedoImage.getImage(); }
    VkImageView getAlbedoImageView() const { return albedoImage.getImageView(); }
    VkSampler getAlbedoSampler() const { return albedoImage.getSampler(); }

    VkExtent2D getExtent() const { return positionImage.getExtent(); }

    bool isWritten() const { return written; }
    void markWritten() { written = true; }
};
