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
#include "imgui.h"

using json = nlohmann::json;

struct InputState;       // forward declare
class Camera;            // forward declare
class GBuffer;           // forward declare
class RayTracingModel;   // forward declare

// Image slot for inter-pass communication.
//
// Output-image barrier convention:
//   - Pure-write output (compute/fragment overwrites every texel, no frame-to-frame
//     dependency): use VK_IMAGE_LAYOUT_UNDEFINED as the old layout. srcAccess=0,
//     srcStage=TOP_OF_PIPE. Contents of the previous frame are discarded.
//     Examples: tonemap, bilateral, network_test LDR output.
//   - History-dependent output (the pass itself reads the previous frame's write,
//     e.g. accumulation, history buffers): track a Fresh/firstFrame flag so the
//     first use picks UNDEFINED and subsequent uses transition from the prior layout.
//     Examples: path_tracing (post-downstream SHADER_READ_ONLY -> GENERAL),
//     TAA historyImage, wavefront_pt color output.
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
    bool offlineMode = false;

    PassImageSlot initInputSlot;

    // Descriptor-set rebind tracking: many passes bind inputSlot.{imageView,sampler}
    // into slot 0 of their pipeline. When the upstream pass toggles enable, the
    // slot identity can change mid-run. Helpers below detect that and pair with
    // a targeted updateDescriptorSets() rewrite in the pass.
    VkImageView lastBoundInputView{VK_NULL_HANDLE};
    VkSampler lastBoundInputSampler{VK_NULL_HANDLE};

    bool inputSlotChanged(const PassImageSlot &slot)
    {
        bool changed = slot.imageView != lastBoundInputView ||
                       slot.sampler != lastBoundInputSampler;
        if (changed)
            markInputSlotBound(slot);
        return changed;
    }

    void markInputSlotBound(const PassImageSlot &slot)
    {
        lastBoundInputView = slot.imageView;
        lastBoundInputSampler = slot.sampler;
    }

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
    virtual bool canDisable() const { return true; }

    // --- Dependency injection (App-managed, non-owning). Override in passes that need them. ---
    virtual void setCamera(Camera *) {}
    virtual void setGBuffer(GBuffer *) {}
    virtual void setScene(RayTracingModel *) {}

    // Offline mode switch. Called by Application::drawOffline before the first recordCommand.
    // The base tracks the flag in `offlineMode`; override to react to the transition (e.g.,
    // BlitPass allocates a readback image). Always call through or set the field yourself.
    virtual void setOfflineMode(bool offline) { offlineMode = offline; }
    bool isOfflineMode() const { return offlineMode; }

    // --- Image slot wiring ---
    void setInputSlot(const PassImageSlot &slot) { initInputSlot = slot; }
    virtual PassImageSlot getOutputSlot() const { return initInputSlot; }

    // --- Lifecycle ---
    virtual void init() {}
    // Called every frame even when disabled; each pass handles enabled state internally.
    virtual void update(uint32_t currentFrame, InputState &inputState) {}
    virtual PassImageSlot recordCommand(VkCommandBuffer commandBuffer,
                                        const PassImageSlot &inputSlot,
                                        uint32_t currentFrame, uint32_t imageIndex)
    {
        return inputSlot;
    }
    virtual void endFrame() {}
    virtual void drawUI() {}

    // Non-virtual: draws enable checkbox (if canDisable) + calls drawUI
    void renderUI()
    {
        if (canDisable())
        {
            bool en = enabled;
            if (ImGui::Checkbox("Enabled", &en))
                enabled = en;
        }
        if (enabled)
            drawUI();
    }
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
