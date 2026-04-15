#pragma once

#include "pass_base.h"
#include "pipeline.h"
#include "resource.h"
#include "ray_tracing_model.h"
#include "scene_loader.h"

#include <memory>
#include <array>
#include <algorithm>

// C++ mirror of Slang BrPTVertex (std430 layout, 128 bytes)
struct BrPTVertex
{
    glm::vec3 throughput;        // 12
    uint32_t seed;               // 4
    glm::vec3 radiance;          // 12
    int pixelIndex;              // 4
    glm::vec3 wo;                // 12
    int depth;                   // 4
    int matIndex;                // 4
    int firstChild;              // 4
    int nextSibling;             // 4
    int parent;                  // 4
    int isHit;                   // 4   1 = valid hit, 0 = miss
    int nInnerSamples;           // 4
    int childIndex;              // 4
    int debiasRadianceIdx;       // 4
    glm::vec3 hitPos;            // 12
    float prevBsdfPdf;           // 4
    glm::vec3 hitNormal;         // 12
    int styleType;               // 4
    float styleParam0;           // 4
    int debiasingSamples;        // 4
    int actualBranchingDepth;    // 4
    float mis;                   // 4
};
static_assert(sizeof(BrPTVertex) == 128, "BrPTVertex must be 128 bytes");

struct BrPTUniform
{
    alignas(16) glm::mat4 viewInverse;
    alignas(16) glm::mat4 projInverse;
};

struct BrPTPushConstants
{
    int frameIndex{0};
    int maxDepth{8};
    int lightCount{0};
    int useNEE{1};
    int useMIS{1};
    int screenWidth{0};
    int screenHeight{0};
    int currentDepth{0};         // current depth being processed
    int useDebiasing{1};
    float debiasR{0.5f};         // geometric distribution parameter
    int innerSamples0{4};        // branching factor at branching depth 0
    int innerSamples1{2};        // branching depth 1
    int innerSamples2{1};        // branching depth 2
    int innerSamples3{1};        // branching depth 3+ (last value reused)
    int maxVertices{0};
    int tilePixelOffset{0};      // starting pixel index for current tile
    int tilePixelCount{0};       // number of pixels in current tile
    int rrDepth{3};              // Russian Roulette starts at this depth
};
static_assert(sizeof(BrPTPushConstants) == 72, "BrPTPushConstants must be 72 bytes");

// Per-depth range: [startIdx, count]
struct DepthRange
{
    int startIdx;
    int count;
};

class BranchPTPass : public PassBase
{
    REGISTER_RENDER_PASS(BranchPTPass);

private:
    // 5 compute pipelines (megakernel: no separate extend/shadow)
    std::unique_ptr<ComputePipeline> initPipeline;
    std::unique_ptr<ComputePipeline> advancePipeline;      // megakernel: advance + trace + shadow
    std::unique_ptr<ComputePipeline> propagatePipeline;
    std::unique_ptr<ComputePipeline> accumulatePipeline;
    std::unique_ptr<ComputePipeline> prepareIndirectPipeline;

    // Output image
    ImageResource colorImage;

    // GPU buffers
    std::unique_ptr<StorageBufferResource> vertexBuffer;         // BrPTVertex[]
    std::unique_ptr<StorageBufferResource> counterBuffer;        // 5 uints: [vtxCounter, debiasCounter, dispatchX, 1, 1]
    std::unique_ptr<StorageBufferResource> depthRangeBuffer;     // DepthRange per depth
    std::unique_ptr<StorageBufferResource> debiasDirectBuffer;   // float4 per child
    std::unique_ptr<StorageBufferResource> debiasIndirectBuffer; // float4 per child

    // Camera uniform
    BrPTUniform ubo;
    UniformBufferResource uniformBuffer;
    BrPTPushConstants pushConstants;

    // Scene
    RayTracingModel model;
    bool firstFrame{true};

    uint32_t totalPixels{0};
    uint32_t maxVertices{0};            // verticesPerPixel * pixelsPerPass
    uint32_t maxDebiasEntries{0};
    uint32_t vramBudgetMB{4096};        // VRAM budget for BrPT buffers in MB
    uint32_t verticesPerPixel{0};       // worst-case vertices per pixel
    uint32_t pixelsPerPass{0};          // tile size in pixels
    std::array<int, 4> innerSamples{4, 2, 1, 1};

    void computeBarrier(VkCommandBuffer cmd);

    // Compute worst-case vertices per pixel
    uint32_t calcVerticesPerPixel() const;
    int getInnerSamples(int depth) const;

public:
    BranchPTPass(Device &_d, SwapChain &_sc, const json &params);

    std::string getName() const override { return "BranchPT"; }
    bool canDisable() const override { return false; }
    PassImageSlot getOutputSlot() const override;

    void init() override;
    void drawUI() override;
    void update(uint32_t currentFrame, InputState &inputState) override;
    PassImageSlot recordCommand(VkCommandBuffer commandBuffer,
                                const PassImageSlot &inputSlot,
                                uint32_t currentFrame, uint32_t imageIndex) override;
};
