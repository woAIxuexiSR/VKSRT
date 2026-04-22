#pragma once

#include "pass_base.h"
#include "pipeline.h"
#include "resource.h"
#include "ray_tracing_model.h"

#include <memory>

// C++ mirror of Slang wavefront_structs (std430 layout)
struct WFPathState
{
    glm::vec3 throughput;
    uint32_t seed;
    glm::vec3 L;
    int pixelIndex;
    float prevBsdfPdf;
    int depth;
    int alive;
    int _pad;
};
static_assert(sizeof(WFPathState) == 48, "WFPathState must be 48 bytes");

struct WFRayData
{
    glm::vec3 origin;
    float tMin;
    glm::vec3 direction;
    float tMax;
};
static_assert(sizeof(WFRayData) == 32, "WFRayData must be 32 bytes");

struct WFHitInfo
{
    glm::vec3 hitPos;
    int matIndex;
    glm::vec3 hitNormal;
    int isHit;
};
static_assert(sizeof(WFHitInfo) == 32, "WFHitInfo must be 32 bytes");

struct WFShadowRay
{
    glm::vec3 origin;
    float tMax;
    glm::vec3 direction;
    int pathIndex;
    glm::vec3 contribution;
    float _pad;
};
static_assert(sizeof(WFShadowRay) == 48, "WFShadowRay must be 48 bytes");

// MeshInfo matches HitSBTRecord layout (12 bytes, std430 stride = 12)
// No padding needed: StructuredBuffer<MeshInfo> stride = max_align(int)=4, roundup(12,4)=12

struct WFPTUniform
{
    alignas(16) glm::mat4 viewInverse;
    alignas(16) glm::mat4 projInverse;
};

struct WFPTPushConstants
{
    int frameIndex{0};
    int maxDepth{8};
    int rrDepth{3};
    int lightCount{0};
    int useNEE{1};
    int useMIS{1};
    int screenWidth{0};
    int screenHeight{0};
    int totalPaths{0};
    int bounceIndex{0};
    int queueOffset{0}; // 0 = first half, totalPaths = second half
    int _pad{0};
};

class WavefrontPTPass : public PassBase
{
    REGISTER_RENDER_PASS(WavefrontPTPass);

private:
    // 6 compute pipelines
    std::unique_ptr<ComputePipeline> generatePipeline;
    std::unique_ptr<ComputePipeline> prepareIndirectPipeline;
    std::unique_ptr<ComputePipeline> extendPipeline;
    std::unique_ptr<ComputePipeline> shadePipeline;
    std::unique_ptr<ComputePipeline> shadowPipeline;
    std::unique_ptr<ComputePipeline> accumulatePipeline;

    // Output image
    ImageResource colorImage;

    // GPU buffers
    std::unique_ptr<StorageBufferResource> pathStateBuffer;
    std::unique_ptr<StorageBufferResource> rayBuffer;
    std::unique_ptr<StorageBufferResource> hitInfoBuffer;
    std::unique_ptr<StorageBufferResource> shadowRayBuffer;
    std::unique_ptr<StorageBufferResource> counterBuffer;
    std::unique_ptr<StorageBufferResource> queueBuffer;

    // Camera uniform
    WFPTUniform ubo;
    UniformBufferResource uniformBuffer;
    WFPTPushConstants pushConstants;

    // Scene (injected)
    Camera *camera{nullptr};
    GBuffer *gbuffer{nullptr};
    RayTracingModel *scene{nullptr};
    bool firstFrame{true};

    uint32_t totalPaths{0};

    void computeBarrier(VkCommandBuffer cmd);

public:
    WavefrontPTPass(Device &_d, SwapChain &_sc, const json &params);

    std::string getName() const override { return "WavefrontPT"; }
    bool canDisable() const override { return false; }
    PassImageSlot getOutputSlot() const override;

    void setCamera(Camera *c) override { camera = c; }
    void setGBuffer(GBuffer *gb) override { gbuffer = gb; }
    void setScene(RayTracingModel *s) override { scene = s; }

    void init() override;
    void drawUI() override;
    void update(uint32_t currentFrame, InputState &inputState) override;
    PassImageSlot recordCommand(VkCommandBuffer commandBuffer,
                                const PassImageSlot &inputSlot,
                                uint32_t currentFrame, uint32_t imageIndex) override;
};
