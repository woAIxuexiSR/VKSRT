#include "neural_network.h"

#include "paths.h"

#include <cstring>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>

NeuralNetwork::NeuralNetwork(Device &_d, const json &netJson)
    : device(_d)
{
    totalRawInputDim = 0;
    totalEncodedDim = 0;

    // Top-level default LR; per-encoding "learningRate" in its own JSON takes precedence
    // because the encoding ctor already reads that field itself.
    const bool hasTopLevelLR = netJson.contains("learningRate");
    const float topLevelLR = hasTopLevelLR ? netJson["learningRate"].get<float>() : 0.01f;

    // Build encodings from JSON array (if present)
    if (netJson.contains("encoding") && netJson["encoding"].is_array())
    {
        for (auto &enc : netJson["encoding"])
        {
            std::string type = enc["type"].get<std::string>();
            inputFieldOffsets.push_back(totalRawInputDim);
            auto e = EncodingFactory::create(device, type, enc);
            totalRawInputDim += e->getInputDim();
            outputFieldOffsets.push_back(totalEncodedDim);
            totalEncodedDim += e->getOutputDim();
            // Apply top-level LR only if the encoding's own JSON didn't specify one.
            if (hasTopLevelLR && !enc.contains("learningRate"))
                e->setLearningRate(topLevelLR);
            encodings.push_back(std::move(e));
        }
    }

    // MLP config
    int hiddenSize = 64, hiddenLayers = 2, outputSize = 3;
    int mlpInputSize = hasEncodings() ? (int)totalEncodedDim : 0;
    if (netJson.contains("mlp"))
    {
        const auto &mlp = netJson["mlp"];
        if (mlp.contains("outputSize"))   outputSize = mlp["outputSize"].get<int>();
        if (mlp.contains("hiddenSize"))   hiddenSize = mlp["hiddenSize"].get<int>();
        if (mlp.contains("hiddenLayers")) hiddenLayers = mlp["hiddenLayers"].get<int>();
        if (!hasEncodings() && mlp.contains("inputSize"))
            mlpInputSize = mlp["inputSize"].get<int>();
    }

    if (!hasEncodings())
        totalRawInputDim = mlpInputSize;

    std::vector<LayerConfig> layers;
    layers.push_back({mlpInputSize, hiddenSize});
    for (int i = 0; i < hiddenLayers; i++)
        layers.push_back({hiddenSize, hiddenSize});
    layers.push_back({hiddenSize, outputSize});

    mlp = std::make_unique<MLP>(device, layers);
    if (hasTopLevelLR)
        mlp->setLearningRate(topLevelLR);

    if (netJson.contains("useEMA"))   useEMA   = netJson["useEMA"].get<bool>();
    if (netJson.contains("emaAlpha")) emaAlpha = netJson["emaAlpha"].get<float>();

    if (netJson.contains("loadPath")) loadPath = netJson["loadPath"].get<std::string>();
    if (netJson.contains("savePath")) savePath = netJson["savePath"].get<std::string>();

    allocateLossBuffers();
}

NeuralNetwork::~NeuralNetwork()
{
    if (!savePath.empty())
    {
        const std::string resolved = configRelPath(savePath);
        try
        {
            saveParameters(resolved);
            std::cout << "NeuralNetwork: saved parameters to " << resolved << std::endl;
        }
        catch (const std::exception &e)
        {
            std::cerr << "NeuralNetwork: failed to save parameters to " << resolved
                      << ": " << e.what() << std::endl;
        }
    }

    if (lossReadbackMapped)
        vkUnmapMemory(device.getDevice(), lossReadbackMemory);
    if (lossReadbackBuffer != VK_NULL_HANDLE)
        vkDestroyBuffer(device.getDevice(), lossReadbackBuffer, nullptr);
    if (lossReadbackMemory != VK_NULL_HANDLE)
        vkFreeMemory(device.getDevice(), lossReadbackMemory, nullptr);
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

    // With encodings: holds concatenated encoded features (MLP forward input, MLP dInput).
    // Without encodings: holds MLP dInput only. In both cases size = sampleCount * mlpInputSize.
    concatBuffer = std::make_unique<StorageBufferResource>(
        device, (VkDeviceSize)sampleCount * mlp->getInputSize() * sizeof(float), bufUsage);
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

    if (!loadPath.empty())
    {
        const std::string resolved = configRelPath(loadPath);
        try
        {
            loadParameters(resolved);
            std::cout << "NeuralNetwork: loaded parameters from " << resolved << std::endl;
        }
        catch (const std::exception &e)
        {
            std::cerr << "NeuralNetwork: failed to load parameters from " << resolved
                      << ": " << e.what() << std::endl;
            throw;
        }
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
            shaderPath("neural/ema_kernel.spv"),
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
    uint64_t inferMlpAddr = device.getBufferDeviceAddress(inferMlpParamBuffer->getBuffer());

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
            inferEncParamAddrs[i] = device.getBufferDeviceAddress(inferEncParamBuffers[i]->getBuffer());
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
            vkCmdCopyBuffer(cmd, encodings[i]->getParamBuffer(),
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
        pc.inferParams = device.getBufferDeviceAddress(inferMlpParamBuffer->getBuffer());
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

    // 3. MLP backward (dInput lands in concatBuffer; reused by encoding backward when present)
    mlp->recordBackward(cmd, activationsBuffer->getBuffer(),
                        mlpOutputBuffer->getBuffer(), gtBuffer, concatBuffer->getBuffer(),
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

namespace
{
constexpr char kVknnMagic[4] = {'V', 'K', 'N', 'N'};
constexpr uint32_t kVknnVersion = 1;
}

void NeuralNetwork::saveParameters(const std::string &path) const
{
    std::ofstream os(path, std::ios::binary);
    if (!os)
        throw std::runtime_error("saveParameters: cannot open '" + path + "' for writing");

    os.write(kVknnMagic, sizeof(kVknnMagic));
    const uint32_t version = kVknnVersion;
    os.write(reinterpret_cast<const char *>(&version), sizeof(version));

    mlp->serialize(os);

    const uint32_t encodingCount = (uint32_t)encodings.size();
    os.write(reinterpret_cast<const char *>(&encodingCount), sizeof(encodingCount));
    for (const auto &enc : encodings)
    {
        const std::string name = enc->typeName();
        if (name.size() > 255)
            throw std::runtime_error("saveParameters: encoding typeName too long: " + name);
        uint8_t nameLen = (uint8_t)name.size();
        os.write(reinterpret_cast<const char *>(&nameLen), sizeof(nameLen));
        os.write(name.data(), (std::streamsize)name.size());

        std::ostringstream payload(std::ios::binary);
        enc->serialize(payload);
        const std::string blob = payload.str();
        uint64_t payloadBytes = (uint64_t)blob.size();
        os.write(reinterpret_cast<const char *>(&payloadBytes), sizeof(payloadBytes));
        os.write(blob.data(), (std::streamsize)blob.size());
    }

    if (!os)
        throw std::runtime_error("saveParameters: write failed on '" + path + "'");
}

void NeuralNetwork::loadParameters(const std::string &path)
{
    std::ifstream is(path, std::ios::binary);
    if (!is)
        throw std::runtime_error("loadParameters: cannot open '" + path + "' for reading");

    char magic[4];
    is.read(magic, sizeof(magic));
    if (std::memcmp(magic, kVknnMagic, sizeof(kVknnMagic)) != 0)
        throw std::runtime_error("loadParameters: bad magic (expected VKNN)");

    uint32_t version = 0;
    is.read(reinterpret_cast<char *>(&version), sizeof(version));
    if (version != kVknnVersion)
        throw std::runtime_error("loadParameters: unsupported version " + std::to_string(version));

    mlp->deserialize(is);

    uint32_t encodingCount = 0;
    is.read(reinterpret_cast<char *>(&encodingCount), sizeof(encodingCount));
    if (encodingCount != encodings.size())
        throw std::runtime_error("loadParameters: encodingCount mismatch (file=" +
                                 std::to_string(encodingCount) + ", config=" +
                                 std::to_string(encodings.size()) + ")");

    for (uint32_t i = 0; i < encodingCount; i++)
    {
        uint8_t nameLen = 0;
        is.read(reinterpret_cast<char *>(&nameLen), sizeof(nameLen));
        std::string name(nameLen, '\0');
        is.read(name.data(), nameLen);

        const std::string expected = encodings[i]->typeName();
        if (name != expected)
            throw std::runtime_error("loadParameters: encoding[" + std::to_string(i) +
                                     "] type mismatch (file=" + name + ", config=" + expected + ")");

        uint64_t payloadBytes = 0;
        is.read(reinterpret_cast<char *>(&payloadBytes), sizeof(payloadBytes));

        std::string blob((size_t)payloadBytes, '\0');
        is.read(blob.data(), (std::streamsize)payloadBytes);
        if (!is || (uint64_t)is.gcount() != payloadBytes)
            throw std::runtime_error("loadParameters: short read on encoding[" +
                                     std::to_string(i) + "] payload");

        std::istringstream payload(blob, std::ios::binary);
        encodings[i]->deserialize(payload);
    }
}
