#include "neural_network.h"

#include <cstring>

NeuralNetwork::NeuralNetwork(Device &_d, const std::vector<LayerConfig> &_layers)
    : NeuralNetwork(_d, Config{_layers, false, {}})
{
}

NeuralNetwork::NeuralNetwork(Device &_d, const Config &cfg)
    : device(_d)
{
    vkGetBufferDeviceAddressKHR = device.loadDeviceFunction<PFN_vkGetBufferDeviceAddressKHR>(
        "vkGetBufferDeviceAddressKHR");

    mlp = std::make_unique<MLP>(device, cfg.layers);

    if (cfg.useEncoding)
    {
        hashGrid = std::make_unique<HashGridEncoding>(device, cfg.encoding);
        if (mlp->getInputSize() != hashGrid->getEncodedDim())
            throw std::runtime_error("NeuralNetwork: MLP inputSize must equal HashGrid encodedDim");
    }

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

    // Activations: sampleCount * actStride * sizeof(float)
    activationsBuffer = std::make_unique<StorageBufferResource>(
        device, (VkDeviceSize)sampleCount * mlp->getActStride() * sizeof(float), bufUsage);

    // MLP output: sampleCount * outputSize * sizeof(float)
    mlpOutputBuffer = std::make_unique<StorageBufferResource>(
        device, (VkDeviceSize)sampleCount * mlp->getOutputSize() * sizeof(float), bufUsage);

    if (hashGrid)
    {
        // Encoded buffer: sampleCount * encodedDim * sizeof(float)
        // Also used as dInput buffer during backward (MLP dInput = dEncoded)
        encodedBuffer = std::make_unique<StorageBufferResource>(
            device, (VkDeviceSize)sampleCount * hashGrid->getEncodedDim() * sizeof(float), bufUsage);

        // dInput not needed separately — encodedBuffer is reused as dEncoded
        dInputBuffer = nullptr;
    }
    else
    {
        encodedBuffer = nullptr;
        // dInput still needed for MLP backward (but we don't use it further)
        dInputBuffer = std::make_unique<StorageBufferResource>(
            device, (VkDeviceSize)sampleCount * mlp->getInputSize() * sizeof(float), bufUsage);
    }
}

void NeuralNetwork::initWeights(unsigned int seed)
{
    mlp->initWeights(seed);
    mlp->resetAdamState();

    if (hashGrid)
    {
        hashGrid->initTable(seed ^ 0x9E3779B9u);
        hashGrid->resetAdamState();
    }
}

void NeuralNetwork::createPipelines()
{
    mlp->createPipelines();
    if (hashGrid)
        hashGrid->createPipelines();
}

int NeuralNetwork::getTotalParams() const
{
    int total = mlp->getTotalParams();
    if (hashGrid)
        total += hashGrid->getTotalFeatures();
    return total;
}

int NeuralNetwork::getInputSize() const
{
    return hashGrid ? hashGrid->getConfig().inputDim : mlp->getInputSize();
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

    if (hashGrid)
    {
        hashGrid->recordForward(cmd, inputBuffer, encodedBuffer->getBuffer(), sampleCount);
        computeBarrier(cmd);
        mlpInput = encodedBuffer->getBuffer();
    }

    mlp->recordForward(cmd, mlpInput, outputBuffer, activationsBuffer->getBuffer(), sampleCount);
}

void NeuralNetwork::recordTrain(VkCommandBuffer cmd, VkBuffer inputBuffer,
                                VkBuffer gtBuffer, uint32_t sampleCount)
{
    lastSampleCount = sampleCount;

    VkBuffer mlpInput = inputBuffer;

    // 1. Forward: [hashgrid_forward →] mlp_forward
    if (hashGrid)
    {
        hashGrid->recordForward(cmd, inputBuffer, encodedBuffer->getBuffer(), sampleCount);
        computeBarrier(cmd);
        mlpInput = encodedBuffer->getBuffer();
    }

    mlp->recordForward(cmd, mlpInput, mlpOutputBuffer->getBuffer(),
                       activationsBuffer->getBuffer(), sampleCount);
    computeBarrier(cmd);

    // 2. Zero grads + loss
    mlp->recordZeroGrads(cmd);
    vkCmdFillBuffer(cmd, lossGpuBuffer->getBuffer(), 0, sizeof(float), 0);
    if (hashGrid)
        hashGrid->recordZeroGrads(cmd);

    device.memoryBarrier(cmd,
                         VK_ACCESS_2_TRANSFER_WRITE_BIT,
                         VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                         VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                         VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT);

    // 3. MLP backward: activations + output + gt → dInput + weight grads + loss
    VkBuffer dInputBuf = hashGrid ? encodedBuffer->getBuffer() : dInputBuffer->getBuffer();
    mlp->recordBackward(cmd, activationsBuffer->getBuffer(),
                        mlpOutputBuffer->getBuffer(), gtBuffer, dInputBuf,
                        lossGpuBuffer->getBuffer(), sampleCount);
    computeBarrier(cmd);

    // 4. HashGrid backward (if present): dInput(=encodedBuffer) + rawInput → tableGrad
    if (hashGrid)
    {
        hashGrid->recordBackward(cmd, encodedBuffer->getBuffer(), inputBuffer, sampleCount);
        computeBarrier(cmd);
    }

    // 5. Adam updates
    mlp->recordAdam(cmd);
    if (hashGrid)
        hashGrid->recordAdam(cmd);

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
