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

    useEMA = cfg.useEMA;
    emaAlpha = cfg.emaAlpha;

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

    if (useEMA)
    {
        std::vector<DescriptorLayoutBinding> emptyBindings{};
        emaPipeline = std::make_unique<ComputePipeline>(
            device, 1, emptyBindings,
            "build/shaders/neural/ema_kernel.slang.spv",
            sizeof(uint64_t) * 2 + sizeof(uint32_t) + sizeof(float));

        allocateEMABuffers();
    }
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

void NeuralNetwork::allocateEMABuffers()
{
    auto bufUsage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                    VK_BUFFER_USAGE_TRANSFER_DST_BIT |
                    VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                    VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;

    // Shadow MLP param buffer (params region only, no gradients)
    VkDeviceSize mlpParamSize = mlp->getGradientOffset();
    inferMlpParamBuffer = std::make_unique<StorageBufferResource>(device, mlpParamSize, bufUsage);
    uint64_t inferMlpAddr = getBufferDeviceAddress(inferMlpParamBuffer->getBuffer());

    // Build infer layer address buffer: same structure as MLP's but pointing to shadow buffer
    auto &layers = mlp->getLayers();
    VkDeviceSize layerBufSize = layers.size() * 4 * sizeof(uint64_t);
    inferMlpLayerAddrBuffer = std::make_unique<StorageBufferResource>(device, layerBufSize, bufUsage);

    // Read the original layer addresses and remap param pointers to infer buffer
    uint64_t trainBase = mlp->getParamBufferAddress();
    uint64_t gradBase = trainBase + mlp->getGradientOffset();

    // The layer address buffer has 4 uint64 per layer: weights, weightsGrad, bias, biasGrad
    // We remap weights and bias to infer buffer, keep grad pointers pointing to train buffer
    std::vector<uint64_t> inferAddresses(layers.size() * 4);

    // We need to read the original addresses to compute offsets
    // Original: addr[i*4+0] = trainBase + weightsOffset, addr[i*4+2] = trainBase + biasOffset
    // Infer:    addr[i*4+0] = inferBase + weightsOffset, addr[i*4+2] = inferBase + biasOffset
    // Grad pointers are unused in forward, but set them to train buffer grad region for safety
    size_t totalBufSize = 0;
    struct LayerOff { size_t wOff, wSize, bOff, bSize; };
    std::vector<LayerOff> offsets(layers.size());

    auto align64 = [](size_t s) { return (s + 63) & ~63; };
    for (size_t i = 0; i < layers.size(); i++)
    {
        size_t wSize = (size_t)layers[i].inputSize * layers[i].outputSize * sizeof(float);
        size_t bSize = (size_t)layers[i].outputSize * sizeof(float);
        offsets[i].wOff = totalBufSize; offsets[i].wSize = wSize;
        totalBufSize += align64(wSize);
        offsets[i].bOff = totalBufSize; offsets[i].bSize = bSize;
        totalBufSize += align64(bSize);
    }

    for (size_t i = 0; i < layers.size(); i++)
    {
        inferAddresses[i * 4 + 0] = inferMlpAddr + offsets[i].wOff;
        inferAddresses[i * 4 + 1] = gradBase + (offsets[i].wOff); // unused in forward
        inferAddresses[i * 4 + 2] = inferMlpAddr + offsets[i].bOff;
        inferAddresses[i * 4 + 3] = gradBase + (offsets[i].bOff); // unused in forward
    }
    inferMlpLayerAddrBuffer->update(inferAddresses.data());

    // Shadow encoding param buffers (trainable encodings only)
    inferEncParamBuffers.resize(encodings.size());
    inferEncParamAddrs.resize(encodings.size(), 0);
    for (size_t i = 0; i < encodings.size(); i++)
    {
        if (encodings[i]->hasTrainableParams())
        {
            VkDeviceSize encSize = encodings[i]->getParamBufferSize();
            inferEncParamBuffers[i] = std::make_unique<StorageBufferResource>(device, encSize, bufUsage);
            inferEncParamAddrs[i] = getBufferDeviceAddress(inferEncParamBuffers[i]->getBuffer());
        }
    }

    // Copy initial weights to infer buffers
    VkCommandBuffer cmd = device.beginSingleTimeCommands();
    VkBufferCopy region{};
    region.size = mlpParamSize;
    vkCmdCopyBuffer(cmd, mlp->getParamBuffer(), inferMlpParamBuffer->getBuffer(), 1, &region);
    for (size_t i = 0; i < encodings.size(); i++)
    {
        if (inferEncParamBuffers[i])
        {
            VkBufferCopy encRegion{};
            encRegion.size = encodings[i]->getParamBufferSize();
            vkCmdCopyBuffer(cmd, dynamic_cast<HashGridEncoding*>(encodings[i].get())->getTableBuffer(),
                            inferEncParamBuffers[i]->getBuffer(), 1, &encRegion);
        }
    }
    device.endSingleTimeCommands(cmd);
}

struct EmaPushConstants
{
    uint64_t inferParams;
    uint64_t trainParams;
    uint32_t count;
    float alpha;
};

void NeuralNetwork::recordEMAUpdate(VkCommandBuffer cmd)
{
    // EMA update MLP params (only the param region, not gradients)
    {
        uint32_t paramFloats = (uint32_t)(mlp->getGradientOffset() / sizeof(float));
        EmaPushConstants pc{};
        pc.inferParams = getBufferDeviceAddress(inferMlpParamBuffer->getBuffer());
        pc.trainParams = mlp->getParamBufferAddress();
        pc.count = paramFloats;
        pc.alpha = emaAlpha;

        emaPipeline->bindPipeline(cmd);
        emaPipeline->pushConstants(cmd, &pc);
        vkCmdDispatch(cmd, (paramFloats + 255) / 256, 1, 1);
    }

    // EMA update trainable encoding params
    for (size_t i = 0; i < encodings.size(); i++)
    {
        if (inferEncParamBuffers[i])
        {
            uint32_t count = (uint32_t)(encodings[i]->getParamBufferSize() / sizeof(float));
            EmaPushConstants pc{};
            pc.inferParams = inferEncParamAddrs[i];
            pc.trainParams = encodings[i]->getParamBufferAddress();
            pc.count = count;
            pc.alpha = emaAlpha;

            emaPipeline->bindPipeline(cmd);
            emaPipeline->pushConstants(cmd, &pc);
            vkCmdDispatch(cmd, (count + 255) / 256, 1, 1);
        }
    }
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
            if (useEMA && inferEncParamBuffers[i])
            {
                encodings[i]->recordForwardWithParams(cmd, inferEncParamAddrs[i],
                    inputBuffer, inputFieldOffsets[i], totalRawInputDim,
                    concatBuffer->getBuffer(), outputFieldOffsets[i], totalEncodedDim,
                    sampleCount);
            }
            else
            {
                encodings[i]->recordForward(cmd,
                    inputBuffer, inputFieldOffsets[i], totalRawInputDim,
                    concatBuffer->getBuffer(), outputFieldOffsets[i], totalEncodedDim,
                    sampleCount);
            }
        }
        computeBarrier(cmd);
        mlpInput = concatBuffer->getBuffer();
    }

    if (useEMA)
        mlp->recordForwardWithLayerAddr(cmd, inferMlpLayerAddrBuffer->getBuffer(),
                                        mlpInput, outputBuffer, activationsBuffer->getBuffer(), sampleCount);
    else
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

    // 5.5. EMA update (train weights → infer weights)
    if (useEMA)
    {
        computeBarrier(cmd);
        recordEMAUpdate(cmd);
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
