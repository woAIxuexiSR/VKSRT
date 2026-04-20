#pragma once

#include "device.h"
#include "resource.h"
#include "pipeline.h"

#include <vector>
#include <memory>
#include <cstdint>

// Fixed-hidden-width MLP (hidden width = 64).
// Owns packed param/grad storage, BDA layer-address table,
// forward/backward compute pipelines, and Adam optimizer state.
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
    void createPipelines();

    // Record forward pass: input → output + activations.
    // Caller must provide activations buffer of size sampleCount * actStride * sizeof(float).
    void recordForward(VkCommandBuffer cmd, VkBuffer input, VkBuffer output,
                       VkBuffer activations, uint32_t sampleCount);

    // Record backward pass: activations + output + gt → dInput + weight gradients + loss accumulation.
    void recordBackward(VkCommandBuffer cmd, VkBuffer activations,
                        VkBuffer output, VkBuffer gt, VkBuffer dInput,
                        VkBuffer loss, uint32_t sampleCount);

    // Record zero-fill of gradient region.
    void recordZeroGrads(VkCommandBuffer cmd);

    // Record Adam optimizer step on MLP parameters.
    void recordAdam(VkCommandBuffer cmd);

    // Reset Adam state (call once after initWeights).
    void resetAdamState();

    int getTotalParams() const { return totalParamCount; }
    int getLayerCount() const { return (int)layers.size(); }
    int getHiddenLayerCount() const { return hiddenLayerCount; }
    int getInputSize() const { return layers.front().inputSize; }
    int getOutputSize() const { return layers.back().outputSize; }
    int getActStride() const { return (hiddenLayerCount + 2) * 64; }
    const std::vector<LayerConfig> &getLayers() const { return layers; }

    VkBuffer getParamBuffer() const { return paramBuffer->getBuffer(); }
    VkDeviceSize getParamBufferSize() const { return totalBufferSize; }
    VkDeviceSize getGradientOffset() const { return gradientOffset; }
    uint64_t getParamBufferAddress() const { return paramBufferAddr; }

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

    // Adam optimizer state
    std::unique_ptr<StorageBufferResource> adamState;

    // Compute pipelines
    std::unique_ptr<ComputePipeline> forwardPipeline;
    std::unique_ptr<ComputePipeline> backwardPipeline;
    std::unique_ptr<ComputePipeline> adamPipeline;

    PFN_vkGetBufferDeviceAddressKHR vkGetBufferDeviceAddressKHR{nullptr};
    uint64_t getBufferDeviceAddress(VkBuffer buffer);

    void allocateStorage();
    void buildLayerAddressBuffer();
};
