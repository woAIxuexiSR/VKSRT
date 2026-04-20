#pragma once

#include "device.h"
#include "resource.h"
#include "pipeline.h"
#include "mlp.h"
#include "hash_grid_encoding.h"

#include <vector>
#include <memory>
#include <cstdint>

// Orchestrator: owns an MLP, optional HashGridEncoding, training pipelines,
// Adam optimizer state, and loss readback.
class NeuralNetwork
{
public:
    using LayerConfig = MLP::LayerConfig;

    struct Config
    {
        std::vector<LayerConfig> layers;
        bool useEncoding{false};
        HashGridEncoding::Config encoding{};
    };

    // Simple constructor: MLP only, no encoding.
    NeuralNetwork(Device &device, const std::vector<LayerConfig> &layers);

    // Full constructor: MLP + optional hash grid encoding.
    NeuralNetwork(Device &device, const Config &config);

    ~NeuralNetwork();

    NeuralNetwork(const NeuralNetwork &) = delete;
    NeuralNetwork &operator=(const NeuralNetwork &) = delete;

    void initWeights(unsigned int seed = 42);

    // Full training step: zero grads + loss → train dispatch → Adam (MLP + hash table) → copy loss
    void recordTrain(VkCommandBuffer cmd, VkBuffer inputBuffer, VkBuffer gtBuffer, uint32_t sampleCount);

    // Read loss from previous frame (1-frame latency, no stall)
    float readLoss() const;

    // Passthrough accessors (MLP owns the actual buffers/layout)
    int getTotalParams() const { return mlp->getTotalParams(); }
    int getLayerCount() const { return mlp->getLayerCount(); }
    const std::vector<LayerConfig> &getLayers() const { return mlp->getLayers(); }
    VkBuffer getLayerAddressBuffer() const { return mlp->getLayerAddressBuffer(); }

    // HashGrid accessors (null if encoding disabled)
    bool hasEncoding() const { return hashGrid != nullptr; }
    const HashGridEncoding *getEncoding() const { return hashGrid.get(); }

private:
    Device &device;
    std::unique_ptr<MLP> mlp;
    std::unique_ptr<HashGridEncoding> hashGrid;

    uint32_t lastSampleCount{1};

    // Adam state: separate buffers per parameter block (MLP params and hash table
    // are independent, so they each need their own (m, v, t) state).
    std::unique_ptr<StorageBufferResource> adamStateMlp;
    std::unique_ptr<StorageBufferResource> adamStateHash;

    // Loss tracking
    std::unique_ptr<StorageBufferResource> lossGpuBuffer;
    VkBuffer lossReadbackBuffer{VK_NULL_HANDLE};
    VkDeviceMemory lossReadbackMemory{VK_NULL_HANDLE};
    void *lossReadbackMapped{nullptr};

    std::unique_ptr<ComputePipeline> trainPipeline;
    std::unique_ptr<ComputePipeline> adamPipeline;

    PFN_vkGetBufferDeviceAddressKHR vkGetBufferDeviceAddressKHR{nullptr};
    uint64_t getBufferDeviceAddress(VkBuffer buffer);

    void allocateTrainingBuffers();
    void createPipelines();
    void computeBarrier(VkCommandBuffer cmd);
};
