#pragma once

#include "device.h"
#include "resource.h"
#include "pipeline.h"

#include <vector>
#include <memory>
#include <cstdint>

class NeuralNetwork
{
public:
    struct LayerConfig
    {
        int inputSize;
        int outputSize;
    };

    NeuralNetwork(Device &device, const std::vector<LayerConfig> &layers);
    ~NeuralNetwork();

    NeuralNetwork(const NeuralNetwork &) = delete;
    NeuralNetwork &operator=(const NeuralNetwork &) = delete;

    int getTotalParams() const { return totalParamCount; }
    int getLayerCount() const { return (int)layers.size(); }
    const std::vector<LayerConfig> &getLayers() const { return layers; }

    void initWeights(unsigned int seed = 42);

    // Full training step: zero grad + loss → train dispatch → adam → copy loss to readback
    void recordTrain(VkCommandBuffer cmd, VkBuffer inputBuffer, VkBuffer gtBuffer, uint32_t sampleCount);

    // Read loss from previous frame (1-frame latency, no stall)
    float readLoss() const;

    // For passes that need to bind the network for inference
    VkBuffer getNetworkConstantBuffer() const { return networkConstantBuffer->getBuffer(); }
    VkDeviceSize getNetworkConstantBufferSize() const { return networkConstantBufferSize; }
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
    uint32_t lastSampleCount{1};
    VkDeviceSize networkConstantBufferSize{0};

    std::unique_ptr<StorageBufferResource> layerAddressBuffer;
    std::unique_ptr<StorageBufferResource> paramBuffer;
    std::unique_ptr<StorageBufferResource> adamStateBuffer;
    std::unique_ptr<UniformBufferResource> networkConstantBuffer;

    // Loss tracking
    std::unique_ptr<StorageBufferResource> lossGpuBuffer;
    VkBuffer lossReadbackBuffer{VK_NULL_HANDLE};
    VkDeviceMemory lossReadbackMemory{VK_NULL_HANDLE};
    void *lossReadbackMapped{nullptr};

    std::unique_ptr<ComputePipeline> trainPipeline;
    std::unique_ptr<ComputePipeline> adamPipeline;

    PFN_vkGetBufferDeviceAddressKHR vkGetBufferDeviceAddressKHR{nullptr};
    uint64_t getBufferDeviceAddress(VkBuffer buffer);

    void allocateStorage();
    void buildConstantBuffer();
    void buildLayerAddressBuffer();
    void createPipelines();
    void computeBarrier(VkCommandBuffer cmd);
};
