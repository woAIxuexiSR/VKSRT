#pragma once

#include "device.h"
#include "swap_chain.h"
#include "resource.h"

#include <string>
#include <memory>
#include <unordered_map>
#include <functional>
#include <iostream>
#include <nlohmann/json.hpp>

using json = nlohmann::json;

struct InputState; // forward declare

// Image slot for inter-pass communication
struct PassImageSlot
{
    VkImage image{VK_NULL_HANDLE};
    VkImageView imageView{VK_NULL_HANDLE};
    VkSampler sampler{VK_NULL_HANDLE};
    VkFormat format{VK_FORMAT_UNDEFINED};
    VkExtent2D extent{0, 0};
    VkImageLayout layout{VK_IMAGE_LAYOUT_UNDEFINED};
};

// Base class for all render passes in the chain.
class PassBase
{
protected:
    Device &device;
    SwapChain &swapChain;
    bool enabled = true;

    std::unordered_map<std::string, PassImageSlot> inputs;
    std::unordered_map<std::string, PassImageSlot> outputs;

public:
    PassBase(Device &_d, SwapChain &_sc) : device(_d), swapChain(_sc) {}
    virtual ~PassBase() = default;

    PassBase(const PassBase &) = delete;
    PassBase &operator=(const PassBase &) = delete;

    // --- Identity ---
    virtual std::string getName() const = 0;

    // --- Enable/Disable ---
    void setEnabled(bool e) { enabled = e; }
    bool isEnabled() const { return enabled; }

    // --- Image slot wiring ---
    void setInput(const std::string &name, const PassImageSlot &slot)
    {
        inputs[name] = slot;
    }

    PassImageSlot getOutput(const std::string &name) const
    {
        auto it = outputs.find(name);
        if (it != outputs.end())
            return it->second;
        return {};
    }

    // --- Lifecycle ---
    virtual void init() {}
    virtual void update(uint32_t currentFrame, InputState &inputState) {}
    virtual void recordCommand(VkCommandBuffer commandBuffer,
                               uint32_t currentFrame, uint32_t imageIndex) = 0;
    virtual void endFrame() {}
    virtual void drawUI() {}
};

// Factory for creating render passes by name
class RenderPassFactory
{
public:
    using BuildFunction = std::function<std::shared_ptr<PassBase>(Device &, SwapChain &, const json &)>;

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
            getMap().emplace(name, [](Device &d, SwapChain &sc, const json &params)
                             { return std::make_shared<T>(d, sc, params); });
        }
    };

    static std::shared_ptr<PassBase> createPass(const std::string &name,
                                                Device &device, SwapChain &swapChain,
                                                const json &params = {})
    {
        auto &map = getMap();
        auto it = map.find(name);
        if (it == map.end())
            throw std::runtime_error("RenderPassFactory: pass '" + name + "' not registered");
        return it->second(device, swapChain, params);
    }

    static void printRegistered()
    {
        auto &map = getMap();
        std::cout << "Registered passes:" << std::endl;
        for (auto &[name, _] : map)
            std::cout << "  " << name << std::endl;
    }
};

// Place in class declaration (.h)
#define REGISTER_RENDER_PASS(T) \
    static RenderPassFactory::Register<T> reg

// Place in .cpp file, at file scope
#define REGISTER_RENDER_PASS_CPP(T, Name) \
    RenderPassFactory::Register<T> T::reg(Name)
