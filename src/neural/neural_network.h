#pragma once

#include "device.h"
#include "resource.h"
#include "mlp.h"
#include "hash_grid_encoding.h"

#include <vector>
#include <memory>
#include <cstdint>

// Orchestrator: owns an MLP + optional HashGridEncoding, manages intermediate
// buffers, and records forward/train dispatch chains.
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

    NeuralNetwork(Device &device, const std::vector<LayerConfig> &layers);
    NeuralNetwork(Device &device, const Config &config);
    ~NeuralNetwork();

    NeuralNetwork(const NeuralNetwork &) = delete;
    NeuralNetwork &operator=(const NeuralNetwork &) = delete;

    void initWeights(unsigned int seed = 42);
    void createPipelines();

    // Pre-allocate intermediate buffers for at least sampleCount samples.
    // Call before recordForward/recordTrain if multiple calls per frame use different counts.
    void ensureBuffers(uint32_t sampleCount);

    // Record full forward chain: [hashgrid_forward →] mlp_forward.
    // Writes MLP output to outputBuffer.
    void recordForward(VkCommandBuffer cmd, VkBuffer inputBuffer,
                       VkBuffer outputBuffer, uint32_t sampleCount);

    // Record full training step: forward → loss → backward → adam → copy loss.
    void recordTrain(VkCommandBuffer cmd, VkBuffer inputBuffer,
                     VkBuffer gtBuffer, uint32_t sampleCount);

    float readLoss() const;

    int getTotalParams() const;
    int getInputSize() const;
    int getOutputSize() const;
    bool hasEncoding() const { return hashGrid != nullptr; }

private:
    Device &device;
    std::unique_ptr<MLP> mlp;
    std::unique_ptr<HashGridEncoding> hashGrid;

    uint32_t lastSampleCount{1};
    uint32_t maxSampleCount{0};

    // Intermediate buffers (sized to maxSampleCount on first use)
    std::unique_ptr<StorageBufferResource> encodedBuffer;
    std::unique_ptr<StorageBufferResource> activationsBuffer;
    std::unique_ptr<StorageBufferResource> mlpOutputBuffer;
    std::unique_ptr<StorageBufferResource> dInputBuffer;

    // Loss tracking
    std::unique_ptr<StorageBufferResource> lossGpuBuffer;
    VkBuffer lossReadbackBuffer{VK_NULL_HANDLE};
    VkDeviceMemory lossReadbackMemory{VK_NULL_HANDLE};
    void *lossReadbackMapped{nullptr};

    PFN_vkGetBufferDeviceAddressKHR vkGetBufferDeviceAddressKHR{nullptr};
    uint64_t getBufferDeviceAddress(VkBuffer buffer);

    void ensureIntermediateBuffers(uint32_t sampleCount);
    void allocateLossBuffers();
    void computeBarrier(VkCommandBuffer cmd);
};
