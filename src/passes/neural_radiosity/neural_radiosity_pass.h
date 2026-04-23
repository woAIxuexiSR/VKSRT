#pragma once

#include "pass_base.h"
#include "pipeline.h"
#include "resource.h"
#include "ray_tracing_model.h"
#include "neural_network.h"

#include <memory>

// Neural Radiosity (Hadadan et al. 2021): standalone compute-only pass that fits
// the outgoing radiance field L_o(x, omega) with a neural network via self-consistent
// bootstrapping.
//
// Offline mode (--offline N):
//   Sample surface points x area-weighted over the scene, estimate target radiance
//   as NEE(x) + 1-bounce-to-y (use EMA network eval at y for non-emissive y, or
//   emission with MIS for emissive y), train the network via MSE against this target.
//   Output image is black.
// Online mode:
//   Primary ray per pixel, query the trained network at the first non-delta hit,
//   compositing emission pass-through for emissive hits.

struct NRUniform
{
    alignas(16) glm::mat4 viewInverse;
    alignas(16) glm::mat4 projInverse;
};

struct NRSampleGenPC
{
    uint64_t lhsInputAddr;      // B * 12 floats: [x, n_x, omega_o, albedo_x]
    uint64_t rhsInputAddr;      // B * S * 12 floats: features at y (one per RHS sub-sample)
    uint64_t auxAddr;           // B * S * 12 floats: per-sub-sample aux (NEE + throughput + emission + flags)
    uint32_t batchSize;
    uint32_t rhsSamples;        // S: Monte Carlo sub-samples per training entry for RHS averaging
    uint32_t frameIndex;
    uint32_t surfaceTriCount;
    int lightCount;
    float totalSurfaceArea;
    uint32_t _pad0;
};

struct NRBuildTargetPC
{
    uint64_t auxAddr;
    uint64_t rhsPredAddr;       // B * S * 3 floats (EMA network forward output, per sub-sample)
    uint64_t gtAddr;            // B * 3 floats (target: mean over S sub-samples)
    uint32_t batchSize;
    uint32_t rhsSamples;        // S
};

struct NRQueryGenPC
{
    uint64_t queryInputAddr;    // P * 12 floats
    uint64_t primaryInfoAddr;   // P * vec4 (rgb = direct emission or 0, a = 1 if non-emissive hit, 0 otherwise)
    uint32_t width;
    uint32_t height;
    uint32_t frameIndex;
    uint32_t _pad0;
};

struct NRCompositePC
{
    uint64_t queryOutputAddr;   // P * 3 floats
    uint64_t primaryInfoAddr;
    uint32_t width;
    uint32_t height;
    uint32_t showBlack;         // 1 when offline (trainOnly): colorImage written as pure black
    uint32_t _pad0;
};

class NeuralRadiosityPass : public PassBase
{
    REGISTER_RENDER_PASS(NeuralRadiosityPass);

private:
    std::unique_ptr<ComputePipeline> sampleGenPipeline;
    std::unique_ptr<ComputePipeline> buildTargetPipeline;
    std::unique_ptr<ComputePipeline> queryGenPipeline;
    std::unique_ptr<ComputePipeline> compositePipeline;
    std::unique_ptr<NeuralNetwork> network;

    ImageResource colorImage;
    NRUniform ubo;
    UniformBufferResource uniformBuffer;

    std::unique_ptr<StorageBufferResource> lhsInputBuffer;
    std::unique_ptr<StorageBufferResource> rhsInputBuffer;
    std::unique_ptr<StorageBufferResource> rhsPredBuffer;
    std::unique_ptr<StorageBufferResource> auxBuffer;
    std::unique_ptr<StorageBufferResource> gtBuffer;

    std::unique_ptr<StorageBufferResource> queryInputBuffer;
    std::unique_ptr<StorageBufferResource> queryOutputBuffer;
    std::unique_ptr<StorageBufferResource> primaryInfoBuffer;

    uint64_t lhsInputAddr{0}, rhsInputAddr{0}, auxAddr{0}, rhsPredAddr{0}, gtAddr{0};
    uint64_t queryInputAddr{0}, queryOutputAddr{0}, primaryInfoAddr{0};

    Camera *camera{nullptr};
    RayTracingModel *scene{nullptr};

    uint32_t batchSize{65536};
    uint32_t rhsSamples{1};
    uint32_t pixelCount{0};
    uint32_t frameIndex{0};
    int totalRawInputDim{12};
    float currentLoss{0.0f};
    double lossAccum{0.0};
    uint32_t lossCount{0};

public:
    NeuralRadiosityPass(Device &_d, SwapChain &_sc, const json &params);

    std::string getName() const override { return "NeuralRadiosity"; }
    bool canDisable() const override { return false; }
    PassImageSlot getOutputSlot() const override;

    void setCamera(Camera *c) override { camera = c; }
    void setScene(RayTracingModel *s) override { scene = s; }

    void init() override;
    void drawUI() override;
    void update(uint32_t currentFrame, InputState &inputState) override;
    PassImageSlot recordCommand(VkCommandBuffer commandBuffer,
                                const PassImageSlot &inputSlot,
                                uint32_t currentFrame, uint32_t imageIndex) override;
};
