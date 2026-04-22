#pragma once

#include "pass_base.h"
#include "pipeline.h"
#include "resource.h"
#include "neural_network.h"

#include <memory>

class NetworkTestPass : public PassBase
{
    REGISTER_RENDER_PASS(NetworkTestPass);

private:
    std::unique_ptr<NeuralNetwork> network;

    ImageResource outputImage;

    std::unique_ptr<StorageBufferResource> inputBuffer;
    std::unique_ptr<StorageBufferResource> gtBuffer;

    std::unique_ptr<StorageBufferResource> inferenceInputBuffer;
    std::unique_ptr<StorageBufferResource> inferenceOutputBuffer;

    std::unique_ptr<ComputePipeline> dataGenPipeline;
    std::unique_ptr<ComputePipeline> writeImagePipeline;

    int totalRawInputDim{2};
    int batchSize{256};
    int maxBatchSize{4096};
    bool showGT{false};
    bool trainEnabled{true};
    float currentLoss{0.0f};
    uint32_t frameIndex{0};

public:
    NetworkTestPass(Device &_d, SwapChain &_sc, const json &params);

    std::string getName() const override { return "Network Test"; }
    bool canDisable() const override { return true; }
    PassImageSlot getOutputSlot() const override;

    void init() override;
    void update(uint32_t currentFrame, InputState &inputState) override;
    void drawUI() override;
    PassImageSlot recordCommand(VkCommandBuffer commandBuffer,
                                const PassImageSlot &inputSlot,
                                uint32_t currentFrame, uint32_t imageIndex) override;
};
