#pragma once

#include "pass_base.h"
#include "pipeline.h"
#include "resource.h"
#include "ray_tracing_model.h"
#include "neural_network.h"

#include <memory>

struct NRCUniform
{
    alignas(16) glm::mat4 viewInverse;
    alignas(16) glm::mat4 projInverse;
};

struct NRCPushConstants
{
    int frameIndex{0};
    int maxDepth{8};
    int rrDepth{3};
    int lightCount{0};
    int useNEE{1};
    int useMIS{1};
    int nrcQueryDepth{2};
    int trainFraction{16};
    int screenWidth{0};
    int screenHeight{0};
    int useNRC{1};
    uint64_t trainInputAddr{0};
    uint64_t trainGTAddr{0};
    uint64_t trainCounterAddr{0};
    uint64_t queryInputAddr{0};
};

struct NRCCompositePushConstants
{
    uint64_t queryOutputAddr{0};
    uint32_t width{0};
    uint32_t height{0};
};

class NRCPass : public PassBase
{
    REGISTER_RENDER_PASS(NRCPass);

private:
    std::unique_ptr<RayTracingPipeline> rtPipeline;
    std::unique_ptr<ComputePipeline> compositePipeline;
    std::unique_ptr<NeuralNetwork> network;

    ImageResource colorImage;
    ImageResource shortPathImage;
    ImageResource throughputImage;

    NRCUniform ubo;
    UniformBufferResource uniformBuffer;
    NRCPushConstants pushConstants;

    Camera *camera{nullptr};
    RayTracingModel *scene{nullptr};
    bool firstFrame{true};

    std::unique_ptr<StorageBufferResource> trainInputBuffer;
    std::unique_ptr<StorageBufferResource> trainGTBuffer;
    std::unique_ptr<StorageBufferResource> trainCounterBuffer;
    std::unique_ptr<StorageBufferResource> queryInputBuffer;
    std::unique_ptr<StorageBufferResource> queryOutputBuffer;

    uint32_t pixelCount{0};
    uint32_t trainBatchSize{0};
    int totalRawInputDim{12};

    bool trainEnabled{true};
    float currentLoss{0.0f};

public:
    NRCPass(Device &_d, SwapChain &_sc, const json &params);

    std::string getName() const override { return "NRC"; }
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
