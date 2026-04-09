#pragma once

#include "device.h"
#include "resource.h"
#include "swap_chain.h"

#include <string>
#include <vector>
#include <cassert>
#include <fstream>
#include <variant>

struct DescriptorLayoutBinding
{
    VkDescriptorType type;
    VkShaderStageFlags flags;
};

using DescriptorInfo = std::variant<
    VkDescriptorBufferInfo,
    VkDescriptorImageInfo,
    VkWriteDescriptorSetAccelerationStructureKHR>;

class Pipeline
{
protected:
    Device &device;

    VkPipelineLayout pipelineLayout;
    VkPipeline pipeline;

    uint32_t descriptorSetCount;
    std::vector<DescriptorLayoutBinding> descriptorBindings;
    VkDescriptorSetLayout descriptorSetLayout;
    VkDescriptorPool descriptorPool;
    std::vector<VkDescriptorSet> descriptorSets;

    void createDescriptorSetLayout();
    void createDescriptorPool();
    void allocateDescriptorSets();

    std::vector<char> readFile(const std::string &filename);
    VkShaderModule createShaderModule(const std::vector<char> &code);
    virtual void createPipeline() = 0;

public:
    Pipeline(Device &_d, uint32_t _cnt, const std::vector<DescriptorLayoutBinding> &bindings);
    virtual ~Pipeline();

    Pipeline(const Pipeline &) = delete;
    Pipeline &operator=(const Pipeline &) = delete;

    VkPipelineLayout getPipelineLayout() { return pipelineLayout; }
    VkDescriptorSet getDescriptorSet(int index) { return descriptorSets[index]; }
    VkPipeline getPipeline() { return pipeline; }

    virtual void bindPipeline(VkCommandBuffer commandBuffer) = 0;
    virtual void bindDescriptorSets(VkCommandBuffer commandBuffer, int currentFrame) = 0;
    void updateDescriptorSets(const std::vector<std::vector<DescriptorInfo>> &infos);
};

// Graphics pipeline: single slang spv with vertex + fragment entry points
class GraphicsPipeline : public Pipeline
{
protected:
    std::string spvPath;
    std::string vertexEntry;
    std::string fragmentEntry;
    std::vector<VkVertexInputBindingDescription> bindingDescriptions;
    std::vector<VkVertexInputAttributeDescription> attributeDescriptions;
    std::vector<VkFormat> targetFormat;
    VkPrimitiveTopology topology;
    uint32_t pushConstantSize;

    void createPipeline() override;

public:
    GraphicsPipeline(Device &_d, uint32_t _cnt, const std::vector<DescriptorLayoutBinding> &bindings,
                     const std::vector<VkVertexInputBindingDescription> &binding,
                     const std::vector<VkVertexInputAttributeDescription> &attributes,
                     const std::string &_spvPath,
                     const std::vector<VkFormat> &_targetFormat,
                     const std::string &_vertexEntry = "vertexMain",
                     const std::string &_fragmentEntry = "fragmentMain",
                     VkPrimitiveTopology _topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
                     size_t _pushConstantSize = 0)
        : Pipeline(_d, _cnt, bindings), spvPath(_spvPath),
          vertexEntry(_vertexEntry), fragmentEntry(_fragmentEntry),
          bindingDescriptions(binding), attributeDescriptions(attributes), targetFormat(_targetFormat),
          topology(_topology), pushConstantSize((uint32_t)_pushConstantSize) { createPipeline(); }

    void bindPipeline(VkCommandBuffer commandBuffer) override;
    void bindDescriptorSets(VkCommandBuffer commandBuffer, int currentFrame) override;
    void pushConstants(VkCommandBuffer commandBuffer, void *data);
};

// Compute pipeline: single slang compute shader
class ComputePipeline : public Pipeline
{
protected:
    std::string computeSpvPath;
    uint32_t pushConstantSize;

    void createPipeline() override;

public:
    ComputePipeline(Device &_d, uint32_t _cnt, const std::vector<DescriptorLayoutBinding> &bindings,
                    const std::string &_computeSpvPath, size_t _pushConstantSize = 0)
        : Pipeline(_d, _cnt, bindings), computeSpvPath(_computeSpvPath),
          pushConstantSize((uint32_t)_pushConstantSize) { createPipeline(); }

    void bindPipeline(VkCommandBuffer commandBuffer) override;
    void bindDescriptorSets(VkCommandBuffer commandBuffer, int currentFrame) override;
    void pushConstants(VkCommandBuffer commandBuffer, void *data);
};

// Ray tracing pipeline
struct HitSBTRecord
{
    int materialIndex;
    int vertexOffset;
    int indexOffset;
};

class RayTracingPipeline : public Pipeline
{
protected:
    std::string spvPath;
    std::string raygenEntryName;
    std::string missEntryName;
    std::string hitEntryName;

    std::unique_ptr<StorageBufferResource> raygenSBT;
    std::unique_ptr<StorageBufferResource> missSBT;
    std::unique_ptr<StorageBufferResource> hitSBT;

    VkStridedDeviceAddressRegionKHR raygenEntry{};
    VkStridedDeviceAddressRegionKHR missEntry{};
    VkStridedDeviceAddressRegionKHR hitEntry{};
    VkStridedDeviceAddressRegionKHR callEntry{};

    PFN_vkCmdTraceRaysKHR vkCmdTraceRaysKHR{nullptr};
    PFN_vkGetRayTracingShaderGroupHandlesKHR vkGetRayTracingShaderGroupHandlesKHR{nullptr};
    PFN_vkGetBufferDeviceAddressKHR vkGetBufferDeviceAddressKHR{nullptr};

    uint64_t getBufferDeviceAddress(VkBuffer buffer);

    void createPipeline() override;
    void createSBTs(const std::vector<HitSBTRecord> &hitRecords);

public:
    RayTracingPipeline(Device &_d, uint32_t _cnt, const std::vector<DescriptorLayoutBinding> &bindings,
                       const std::string &_spvPath,
                       const std::string &_raygenEntry, const std::string &_missEntry, const std::string &_hitEntry,
                       const std::vector<HitSBTRecord> &hitRecords)
        : Pipeline(_d, _cnt, bindings), spvPath(_spvPath),
          raygenEntryName(_raygenEntry), missEntryName(_missEntry), hitEntryName(_hitEntry)
    {
        createPipeline();
        createSBTs(hitRecords);
    }

    void traceRays(VkCommandBuffer commandBuffer, VkExtent3D extent);
    void bindPipeline(VkCommandBuffer commandBuffer) override;
    void bindDescriptorSets(VkCommandBuffer commandBuffer, int currentFrame) override;
};
