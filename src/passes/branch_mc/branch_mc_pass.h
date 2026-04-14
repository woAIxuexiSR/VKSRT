#pragma once

#include "pass_base.h"
#include "pipeline.h"
#include "resource.h"
#include "ray_tracing_model.h"
#include "scene_loader.h"

#include <memory>
#include <array>
#include <algorithm>

// C++ mirror of Slang BMCVertex (std430 layout, 128 bytes)
struct BMCVertex
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
    int alive;                   // 4
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
static_assert(sizeof(BMCVertex) == 128, "BMCVertex must be 128 bytes");

// Reuse wavefront-compatible ray/hit/shadow structs
struct BMCRayData
{
    glm::vec3 origin;
    float tMin;
    glm::vec3 direction;
    float tMax;
};
static_assert(sizeof(BMCRayData) == 32, "BMCRayData must be 32 bytes");

struct BMCHitInfo
{
    glm::vec3 hitPos;
    int matIndex;
    glm::vec3 hitNormal;
    int isHit;
};
static_assert(sizeof(BMCHitInfo) == 32, "BMCHitInfo must be 32 bytes");

struct BMCShadowRay
{
    glm::vec3 origin;
    float tMax;
    glm::vec3 direction;
    int pathIndex;
    glm::vec3 contribution;
    float _pad;
};
static_assert(sizeof(BMCShadowRay) == 48, "BMCShadowRay must be 48 bytes");

struct BMCUniform
{
    alignas(16) glm::mat4 viewInverse;
    alignas(16) glm::mat4 projInverse;
};

struct BMCPushConstants
{
    int frameIndex{0};
    int maxDepth{8};
    int rrDepth{3};
    int lightCount{0};
    int useNEE{1};
    int useMIS{1};
    int screenWidth{0};
    int screenHeight{0};
    int currentDepth{0};         // current depth being processed
    int depthVertexStart{0};     // vertex start index for current depth
    int depthVertexCount{0};     // vertex count for current depth
    int useDebiasing{1};
    float debiasR{0.5f};         // geometric distribution parameter
    int innerSamples0{4};        // branching factor at depth 0
    int innerSamples1{2};        // depth 1
    int innerSamples2{1};        // depth 2
    int innerSamples3{1};        // depth 3+ (last value reused)
    int maxVertices{0};
    int _pad0{0};
    int _pad1{0};
};
static_assert(sizeof(BMCPushConstants) == 80, "BMCPushConstants must be 80 bytes");

// Per-depth range: [startIdx, count]
struct DepthRange
{
    int startIdx;
    int count;
};

class BranchMCPass : public PassBase
{
    REGISTER_RENDER_PASS(BranchMCPass);

private:
    // 7 compute pipelines
    std::unique_ptr<ComputePipeline> initPipeline;
    std::unique_ptr<ComputePipeline> advancePipeline;
    std::unique_ptr<ComputePipeline> extendPipeline;
    std::unique_ptr<ComputePipeline> shadowPipeline;
    std::unique_ptr<ComputePipeline> propagatePipeline;
    std::unique_ptr<ComputePipeline> accumulatePipeline;
    std::unique_ptr<ComputePipeline> prepareIndirectPipeline;

    // Output image
    ImageResource colorImage;

    // GPU buffers
    std::unique_ptr<StorageBufferResource> vertexBuffer;     // BMCVertex[]
    std::unique_ptr<StorageBufferResource> rayBuffer;
    std::unique_ptr<StorageBufferResource> hitInfoBuffer;
    std::unique_ptr<StorageBufferResource> shadowRayBuffer;
    std::unique_ptr<StorageBufferResource> counterBuffer;    // 8 uints
    std::unique_ptr<StorageBufferResource> depthRangeBuffer; // DepthRange per depth
    std::unique_ptr<StorageBufferResource> debiasDirectBuffer;   // float3 per child
    std::unique_ptr<StorageBufferResource> debiasIndirectBuffer; // float3 per child
    std::unique_ptr<StorageBufferResource> debiasCounterBuffer;  // 1 uint

    // Camera uniform
    BMCUniform ubo;
    UniformBufferResource uniformBuffer;
    BMCPushConstants pushConstants;

    // Scene
    RayTracingModel model;
    bool firstFrame{true};

    uint32_t totalPixels{0};
    uint32_t maxVertices{0};
    uint32_t maxDebiasEntries{0};
    uint32_t vramBudgetMB{4096}; // VRAM budget for BMC buffers in MB
    std::array<int, 4> innerSamples{4, 2, 1, 1};

    void computeBarrier(VkCommandBuffer cmd);

    // Compute max vertices based on inner_samples and pixel count
    uint32_t calcMaxVertices() const;
    int getInnerSamples(int depth) const;

public:
    BranchMCPass(Device &_d, SwapChain &_sc, const json &params);

    std::string getName() const override { return "BranchMC"; }
    bool canDisable() const override { return false; }
    PassImageSlot getOutputSlot() const override;

    void init() override;
    void drawUI() override;
    void update(uint32_t currentFrame, InputState &inputState) override;
    PassImageSlot recordCommand(VkCommandBuffer commandBuffer,
                                const PassImageSlot &inputSlot,
                                uint32_t currentFrame, uint32_t imageIndex) override;
};
