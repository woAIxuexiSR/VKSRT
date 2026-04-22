#pragma once

#include "device.h"
#include "resource.h"
#include "mlp.h"
#include "encoding.h"

#include <vector>
#include <memory>
#include <cstdint>
#include <string>

class NeuralNetwork
{
public:
    using LayerConfig = MLP::LayerConfig;

    // Construct directly from JSON network config.
    // Expected schema:
    //   { "encoding": [ {type, inputDim, ...}, ... ],
    //     "mlp": { "outputSize": N, "hiddenSize": 64, "hiddenLayers": 2 },
    //     "useEMA": bool, "emaAlpha": float }
    NeuralNetwork(Device &device, const json &netJson);
    ~NeuralNetwork();

    NeuralNetwork(const NeuralNetwork &) = delete;
    NeuralNetwork &operator=(const NeuralNetwork &) = delete;

    void initWeights(unsigned int seed = 42);
    void createPipelines();

    void ensureBuffers(uint32_t sampleCount);

    void recordForward(VkCommandBuffer cmd, VkBuffer inputBuffer,
                       VkBuffer outputBuffer, uint32_t sampleCount);

    void recordTrain(VkCommandBuffer cmd, VkBuffer inputBuffer,
                     VkBuffer gtBuffer, uint32_t sampleCount);

    float readLoss() const;

    // Persistence: serialize/deserialize trainable params (MLP + encodings).
    // File format: "VKNN" magic + version + MLP segment + encoding segment (per-encoding payload).
    void saveParameters(const std::string &path) const;
    void loadParameters(const std::string &path);

    int getTotalParams() const;
    int getInputSize() const;
    int getOutputSize() const;
    int getTotalEncodedDim() const { return totalEncodedDim; }
    int getTotalRawInputDim() const { return totalRawInputDim; }
    bool hasEncodings() const { return !encodings.empty(); }
    bool isEMAEnabled() const { return useEMA; }

private:
    Device &device;
    std::unique_ptr<MLP> mlp;
    std::vector<std::unique_ptr<Encoding>> encodings;
    std::vector<uint32_t> inputFieldOffsets;
    std::vector<uint32_t> outputFieldOffsets;
    uint32_t totalRawInputDim{0};
    uint32_t totalEncodedDim{0};

    uint32_t lastSampleCount{1};
    uint32_t maxSampleCount{0};

    std::unique_ptr<StorageBufferResource> concatBuffer;
    std::unique_ptr<StorageBufferResource> activationsBuffer;
    std::unique_ptr<StorageBufferResource> mlpOutputBuffer;

    std::unique_ptr<StorageBufferResource> lossGpuBuffer;
    VkBuffer lossReadbackBuffer{VK_NULL_HANDLE};
    VkDeviceMemory lossReadbackMemory{VK_NULL_HANDLE};
    void *lossReadbackMapped{nullptr};

    bool useEMA{false};
    float emaAlpha{0.99f};

    std::string loadPath;
    std::string savePath;

    std::unique_ptr<StorageBufferResource> inferMlpParamBuffer;
    std::unique_ptr<StorageBufferResource> inferMlpLayerAddrBuffer;
    std::vector<std::unique_ptr<StorageBufferResource>> inferEncParamBuffers;
    std::vector<uint64_t> inferEncParamAddrs;
    std::unique_ptr<ComputePipeline> emaPipeline;

    void allocateEMABuffers();
    void recordEMAUpdate(VkCommandBuffer cmd);

    void ensureIntermediateBuffers(uint32_t sampleCount);
    void allocateLossBuffers();
    void computeBarrier(VkCommandBuffer cmd);
};
