#pragma once

#include "device.h"
#include "resource.h"
#include "mlp.h"
#include "encoding.h"
#include "encoding_factory.h"
#include "hash_grid_encoding.h"

#include <vector>
#include <memory>
#include <cstdint>

class NeuralNetwork
{
public:
    using LayerConfig = MLP::LayerConfig;

    struct EncodingConfig
    {
        std::string type;
        int inputDim;
        json params;
    };

    struct Config
    {
        std::vector<LayerConfig> layers;
        std::vector<EncodingConfig> encodings;
        bool useEMA{false};
        float emaAlpha{0.99f};
        // Legacy single-encoding mode:
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

    void ensureBuffers(uint32_t sampleCount);

    void recordForward(VkCommandBuffer cmd, VkBuffer inputBuffer,
                       VkBuffer outputBuffer, uint32_t sampleCount);

    void recordTrain(VkCommandBuffer cmd, VkBuffer inputBuffer,
                     VkBuffer gtBuffer, uint32_t sampleCount);

    float readLoss() const;

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
    std::unique_ptr<StorageBufferResource> dInputBuffer;

    std::unique_ptr<StorageBufferResource> lossGpuBuffer;
    VkBuffer lossReadbackBuffer{VK_NULL_HANDLE};
    VkDeviceMemory lossReadbackMemory{VK_NULL_HANDLE};
    void *lossReadbackMapped{nullptr};

    PFN_vkGetBufferDeviceAddressKHR vkGetBufferDeviceAddressKHR{nullptr};
    uint64_t getBufferDeviceAddress(VkBuffer buffer);

    bool useEMA{false};
    float emaAlpha{0.99f};

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
