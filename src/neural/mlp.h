#pragma once

#include "device.h"
#include "resource.h"

#include <vector>
#include <memory>
#include <cstdint>

// Fixed-hidden-width MLP (hidden width = 64).
// Owns packed param/grad storage and a BDA layer-address table.
class MLP
{
public:
    struct LayerConfig
    {
        int inputSize;
        int outputSize;
    };

    MLP(Device &device, const std::vector<LayerConfig> &layers);
    ~MLP() = default;

    MLP(const MLP &) = delete;
    MLP &operator=(const MLP &) = delete;

    void initWeights(unsigned int seed = 42);

    int getTotalParams() const { return totalParamCount; }
    int getLayerCount() const { return (int)layers.size(); }
    int getHiddenLayerCount() const { return hiddenLayerCount; }
    int getInputSize() const { return layers.front().inputSize; }
    int getOutputSize() const { return layers.back().outputSize; }
    const std::vector<LayerConfig> &getLayers() const { return layers; }

    // Params and gradients are packed in a single buffer:
    //   [0, gradientOffset)       = params  (weights + biases per layer)
    //   [gradientOffset, total)   = grads   (same layout as params)
    VkBuffer getParamBuffer() const { return paramBuffer->getBuffer(); }
    VkDeviceSize getParamBufferSize() const { return totalBufferSize; }
    VkDeviceSize getGradientOffset() const { return gradientOffset; }
    uint64_t getParamBufferAddress() const { return paramBufferAddr; }

    // BDA table consumed by train/inference shaders: uint64_t[4 * layerCount]
    // Layout per layer: {weights, weightsGrad, biases, biasesGrad}
    VkBuffer getLayerAddressBuffer() const { return layerAddressBuffer->getBuffer(); }

private:
    Device &device;
    std::vector<LayerConfig> layers;
    int hiddenLayerCount{0};

    struct LayerAllocation
    {
        size_t weightsOffset, weightsSize;
        size_t biasOffset, biasSize;
        size_t weightsGradOffset, biasGradOffset;
    };
    std::vector<LayerAllocation> allocations;

    size_t totalBufferSize{0};
    size_t gradientOffset{0};
    int totalParamCount{0};

    std::unique_ptr<StorageBufferResource> paramBuffer;
    std::unique_ptr<StorageBufferResource> layerAddressBuffer;
    uint64_t paramBufferAddr{0};

    PFN_vkGetBufferDeviceAddressKHR vkGetBufferDeviceAddressKHR{nullptr};
    uint64_t getBufferDeviceAddress(VkBuffer buffer);

    void allocateStorage();
    void buildLayerAddressBuffer();
};
