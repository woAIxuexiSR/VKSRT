#include "neural_network.h"

#include <cstring>
#include <stdexcept>

NeuralNetwork::NeuralNetwork(Device &_d, const std::vector<LayerConfig> &_layers)
    : NeuralNetwork(_d, Config{_layers, {}, false, {}})
{
}

NeuralNetwork::NeuralNetwork(Device &_d, const Config &cfg)
    : device(_d)
{
    vkGetBufferDeviceAddressKHR = device.loadDeviceFunction<PFN_vkGetBufferDeviceAddressKHR>(
        "vkGetBufferDeviceAddressKHR");

    // Multi-encoding mode
    if (!cfg.encodings.empty())
    {
        totalRawInputDim = 0;
        totalEncodedDim = 0;
        for (auto &ec : cfg.encodings)
        {
            inputFieldOffsets.push_back(totalRawInputDim);
            auto enc = createEncoding(device, ec.type, ec.inputDim, ec.params);
            totalRawInputDim += enc->getInputDim();
            outputFieldOffsets.push_back(totalEncodedDim);
            totalEncodedDim += enc->getOutputDim();
            encodings.push_back(std::move(enc));
        }

        if (cfg.layers.front().inputSize != (int)totalEncodedDim)
            throw std::runtime_error("NeuralNetwork: MLP inputSize (" +
                std::to_string(cfg.layers.front().inputSize) + ") must equal totalEncodedDim (" +
                std::to_string(totalEncodedDim) + ")");
    }
    // Legacy single-encoding mode
    else if (cfg.useEncoding)
    {
        auto hg = std::make_unique<HashGridEncoding>(device, cfg.encoding);
        totalRawInputDim = hg->getInputDim();
        totalEncodedDim = hg->getOutputDim();
        inputFieldOffsets.push_back(0);
        outputFieldOffsets.push_back(0);
        if (cfg.layers.front().inputSize != (int)totalEncodedDim)
            throw std::runtime_error("NeuralNetwork: MLP inputSize must equal HashGrid encodedDim");
        encodings.push_back(std::move(hg));
    }

    mlp = std::make_unique<MLP>(device, cfg.layers);

    allocateLossBuffers();
}

NeuralNetwork::~NeuralNetwork()
{
    if (lossReadbackMapped)
        vkUnmapMemory(device.getDevice(), lossReadbackMemory);
    if (lossReadbackBuffer != VK_NULL_HANDLE)
        vkDestroyBuffer(device.getDevice(), lossReadbackBuffer, nullptr);
    if (lossReadbackMemory != VK_NULL_HANDLE)
        vkFreeMemory(device.getDevice(), lossReadbackMemory, nullptr);
}

uint64_t NeuralNetwork::getBufferDeviceAddress(VkBuffer buffer)
{
    VkBufferDeviceAddressInfoKHR info{};
    info.sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO_KHR;
    info.buffer = buffer;
    return vkGetBufferDeviceAddressKHR(device.getDevice(), &info);
}

void NeuralNetwork::allocateLossBuffers()
{
    lossGpuBuffer = std::make_unique<StorageBufferResource>(
        device, sizeof(float),
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
            VK_BUFFER_USAGE_TRANSFER_DST_BIT |
            VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
            VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT);

    device.createBuffer(sizeof(float),
                        VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                        lossReadbackBuffer, lossReadbackMemory);
    vkMapMemory(device.getDevice(), lossReadbackMemory, 0, sizeof(float), 0, &lossReadbackMapped);
    memset(lossReadbackMapped, 0, sizeof(float));
}

void NeuralNetwork::ensureIntermediateBuffers(uint32_t sampleCount)
{
    if (sampleCount <= maxSampleCount)
        return;

    maxSampleCount = sampleCount;

    auto bufUsage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                    VK_BUFFER_USAGE_TRANSFER_DST_BIT |
                    VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;

    activationsBuffer = std::make_unique<StorageBufferResource>(
        device, (VkDeviceSize)sampleCount * mlp->getActStride() * sizeof(float), bufUsage);

    mlpOutputBuffer = std::make_unique<StorageBufferResource>(
        device, (VkDeviceSize)sampleCount * mlp->getOutputSize() * sizeof(float), bufUsage);

    if (hasEncodings())
    {
        concatBuffer = std::make_unique<StorageBufferResource>(
            device, (VkDeviceSize)sampleCount * totalEncodedDim * sizeof(float), bufUsage);
        dInputBuffer = nullptr;
    }
    else
    {
        concatBuffer = nullptr;
        dInputBuffer = std::make_unique<StorageBufferResource>(
            device, (VkDeviceSize)sampleCount * mlp->getInputSize() * sizeof(float), bufUsage);
    }
}

void NeuralNetwork::initWeights(unsigned int seed)
{
    mlp->initWeights(seed);
    mlp->resetAdamState();

    unsigned int encSeed = seed ^ 0x9E3779B9u;
    for (auto &enc : encodings)
    {
        enc->initParams(encSeed);
        enc->resetAdamState();
        encSeed = encSeed * 2654435761u + 1;
    }
}

void NeuralNetwork::createPipelines()
{
    mlp->createPipelines();
    for (auto &enc : encodings)
        enc->createPipelines();
}

int NeuralNetwork::getTotalParams() const
{
    int total = mlp->getTotalParams();
    for (auto &enc : encodings)
        total += enc->getTrainableParamCount();
    return total;
}

int NeuralNetwork::getInputSize() const
{
    return hasEncodings() ? (int)totalRawInputDim : mlp->getInputSize();
}

int NeuralNetwork::getOutputSize() const
{
    return mlp->getOutputSize();
}

void NeuralNetwork::computeBarrier(VkCommandBuffer cmd)
{
    device.memoryBarrier(cmd,
                         VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT | VK_ACCESS_2_SHADER_WRITE_BIT,
                         VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_SHADER_READ_BIT |
                             VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                         VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                         VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT);
}

float NeuralNetwork::readLoss() const
{
    float totalLoss = *static_cast<const float *>(lossReadbackMapped);
    return lastSampleCount > 0 ? totalLoss / (float)lastSampleCount : 0.0f;
}

void NeuralNetwork::ensureBuffers(uint32_t sampleCount)
{
    ensureIntermediateBuffers(sampleCount);
}

void NeuralNetwork::recordForward(VkCommandBuffer cmd, VkBuffer inputBuffer,
                                  VkBuffer outputBuffer, uint32_t sampleCount)
{
    VkBuffer mlpInput = inputBuffer;

    if (hasEncodings())
    {
        for (size_t i = 0; i < encodings.size(); i++)
        {
            encodings[i]->recordForward(cmd,
                inputBuffer, inputFieldOffsets[i], totalRawInputDim,
                concatBuffer->getBuffer(), outputFieldOffsets[i], totalEncodedDim,
                sampleCount);
        }
        computeBarrier(cmd);
        mlpInput = concatBuffer->getBuffer();
    }

    mlp->recordForward(cmd, mlpInput, outputBuffer, activationsBuffer->getBuffer(), sampleCount);
}

void NeuralNetwork::recordTrain(VkCommandBuffer cmd, VkBuffer inputBuffer,
                                VkBuffer gtBuffer, uint32_t sampleCount)
{
    lastSampleCount = sampleCount;

    VkBuffer mlpInput = inputBuffer;

    // 1. Forward: encoding(s) → MLP
    if (hasEncodings())
    {
        for (size_t i = 0; i < encodings.size(); i++)
        {
            encodings[i]->recordForward(cmd,
                inputBuffer, inputFieldOffsets[i], totalRawInputDim,
                concatBuffer->getBuffer(), outputFieldOffsets[i], totalEncodedDim,
                sampleCount);
        }
        computeBarrier(cmd);
        mlpInput = concatBuffer->getBuffer();
    }

    mlp->recordForward(cmd, mlpInput, mlpOutputBuffer->getBuffer(),
                       activationsBuffer->getBuffer(), sampleCount);
    computeBarrier(cmd);

    // 2. Zero grads + loss
    mlp->recordZeroGrads(cmd);
    vkCmdFillBuffer(cmd, lossGpuBuffer->getBuffer(), 0, sizeof(float), 0);
    for (auto &enc : encodings)
    {
        if (enc->hasTrainableParams())
            enc->recordZeroGrads(cmd);
    }

    device.memoryBarrier(cmd,
                         VK_ACCESS_2_TRANSFER_WRITE_BIT,
                         VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                         VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                         VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT);

    // 3. MLP backward
    VkBuffer dInputBuf = hasEncodings() ? concatBuffer->getBuffer() : dInputBuffer->getBuffer();
    mlp->recordBackward(cmd, activationsBuffer->getBuffer(),
                        mlpOutputBuffer->getBuffer(), gtBuffer, dInputBuf,
                        lossGpuBuffer->getBuffer(), sampleCount);
    computeBarrier(cmd);

    // 4. Encoding backward (trainable only)
    for (size_t i = 0; i < encodings.size(); i++)
    {
        if (encodings[i]->hasTrainableParams())
        {
            encodings[i]->recordBackward(cmd,
                concatBuffer->getBuffer(), outputFieldOffsets[i], totalEncodedDim,
                inputBuffer, inputFieldOffsets[i], totalRawInputDim,
                sampleCount);
        }
    }
    if (!encodings.empty())
        computeBarrier(cmd);

    // 5. Adam updates
    mlp->recordAdam(cmd);
    for (auto &enc : encodings)
    {
        if (enc->hasTrainableParams())
            enc->recordAdam(cmd);
    }

    // 6. Copy loss to readback
    device.memoryBarrier(cmd,
                         VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                         VK_ACCESS_2_TRANSFER_READ_BIT,
                         VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                         VK_PIPELINE_STAGE_2_TRANSFER_BIT);

    VkBufferCopy copyRegion{};
    copyRegion.size = sizeof(float);
    vkCmdCopyBuffer(cmd, lossGpuBuffer->getBuffer(), lossReadbackBuffer, 1, &copyRegion);

    device.memoryBarrier(cmd,
                         VK_ACCESS_2_TRANSFER_WRITE_BIT,
                         VK_ACCESS_2_HOST_READ_BIT,
                         VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                         VK_PIPELINE_STAGE_2_HOST_BIT);
}
