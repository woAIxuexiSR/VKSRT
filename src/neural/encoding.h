#pragma once

#include "device.h"
#include "resource.h"
#include "pipeline.h"

#include <nlohmann/json.hpp>

#include <cassert>
#include <functional>
#include <iosfwd>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <cstdint>

using json = nlohmann::json;

class Encoding
{
public:
    virtual ~Encoding() = default;

    virtual int getInputDim() const = 0;
    virtual int getOutputDim() const = 0;
    virtual bool hasTrainableParams() const = 0;
    virtual int getTrainableParamCount() const { return 0; }
    virtual std::string typeName() const = 0;

    virtual void createPipelines() = 0;
    virtual void initParams(unsigned int seed) {}
    virtual void resetAdamState() {}
    virtual void setLearningRate(float lr) {}

    virtual void recordForward(VkCommandBuffer cmd,
                               VkBuffer rawInput, uint32_t inputOffset, uint32_t inputStride,
                               VkBuffer encodedOutput, uint32_t outputOffset, uint32_t outputStride,
                               uint32_t sampleCount) = 0;

    virtual void recordBackward(VkCommandBuffer cmd,
                                VkBuffer dEncoded, uint32_t dOffset, uint32_t dStride,
                                VkBuffer rawInput, uint32_t inputOffset, uint32_t inputStride,
                                uint32_t sampleCount) = 0;

    virtual void recordZeroGrads(VkCommandBuffer cmd) {}
    virtual void recordAdam(VkCommandBuffer cmd) {}

    virtual uint64_t getParamBufferAddress() const { return 0; }
    virtual VkDeviceSize getParamBufferSize() const { return 0; }
    virtual VkBuffer getParamBuffer() const { return VK_NULL_HANDLE; }

    // Persistence: trainable encodings override both to dump/restore their param buffer.
    // Non-trainable encodings (Frequency/SH/OneBlob/Identity) use default no-op.
    virtual void serialize(std::ostream &os) const {}
    virtual void deserialize(std::istream &is) {}

    // EMA inference path: forward with caller-provided param address (shadow/infer weights)
    // instead of the encoding's own training params.
    //
    // Default impl ignores paramAddr and delegates to recordForward — correct ONLY when the
    // encoding has no trainable params (nothing to override). Trainable encodings MUST
    // override this and plug paramAddr into their forward push constants.
    virtual void recordForwardWithParams(VkCommandBuffer cmd,
                                         uint64_t paramAddr,
                                         VkBuffer rawInput, uint32_t inputOffset, uint32_t inputStride,
                                         VkBuffer encodedOutput, uint32_t outputOffset, uint32_t outputStride,
                                         uint32_t sampleCount)
    {
        assert(!hasTrainableParams() &&
               "Trainable Encoding must override recordForwardWithParams to honor paramAddr");
        recordForward(cmd, rawInput, inputOffset, inputStride,
                      encodedOutput, outputOffset, outputStride, sampleCount);
    }
};

// Self-registering factory. Each encoding places REGISTER_ENCODING(T) in its class
// declaration and REGISTER_ENCODING_CPP(T, "name") at file scope in its .cpp.
// Encoding constructors take (Device&, const json&).
class EncodingFactory
{
public:
    using BuildFunction = std::function<std::unique_ptr<Encoding>(Device &, const json &)>;

private:
    using MapType = std::unordered_map<std::string, BuildFunction>;

public:
    static MapType &getMap()
    {
        static MapType map;
        return map;
    }

    template <class T>
    struct Register
    {
        Register(const std::string &name)
        {
            getMap().emplace(name, [](Device &d, const json &params)
                             { return std::make_unique<T>(d, params); });
        }
    };

    static std::unique_ptr<Encoding> create(Device &device, const std::string &type,
                                            const json &params)
    {
        auto &map = getMap();
        auto it = map.find(type);
        if (it == map.end())
            throw std::runtime_error("EncodingFactory: encoding '" + type + "' not registered");
        return it->second(device, params);
    }
};

// Place in class declaration (.h)
#define REGISTER_ENCODING(T) \
    static EncodingFactory::Register<T> reg

// Place in .cpp file, at file scope
#define REGISTER_ENCODING_CPP(T, Name) \
    EncodingFactory::Register<T> T::reg(Name)
