#pragma once

#include "device.h"
#include "resource.h"
#include "pipeline.h"
#include "material.h"

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <vector>
#include <memory>

class RayTracingModel
{
private:
    Device &device;
    bool finishBuild{false};

    std::vector<glm::vec3> vertices;
    std::vector<glm::vec3> normals;
    std::vector<glm::vec2> texcoords;
    std::vector<glm::uvec3> indices;
    std::vector<HitSBTRecord> hitSBTRecords;
    std::vector<Material> materials;

    std::unique_ptr<StorageBufferResource> vertexBuffer;
    std::unique_ptr<StorageBufferResource> normalBuffer;
    std::unique_ptr<StorageBufferResource> texcoordBuffer;
    std::unique_ptr<StorageBufferResource> indexBuffer;
    std::unique_ptr<StorageBufferResource> materialBuffer;
    std::unique_ptr<StorageBufferResource> transformBuffer;
    std::unique_ptr<StorageBufferResource> instanceBuffer;
    std::unique_ptr<StorageBufferResource> lightBuffer;
    std::unique_ptr<StorageBufferResource> meshInfoBuffer;
    int lightCount{0};

    std::vector<VkAccelerationStructureKHR> blas;
    std::vector<std::unique_ptr<StorageBufferResource>> blasBuffers;
    std::vector<uint64_t> blasAddresses;

    VkAccelerationStructureKHR tlas{VK_NULL_HANDLE};
    std::unique_ptr<StorageBufferResource> tlasBuffer;

    PFN_vkGetBufferDeviceAddressKHR vkGetBufferDeviceAddressKHR;
    PFN_vkCreateAccelerationStructureKHR vkCreateAccelerationStructureKHR;
    PFN_vkCmdBuildAccelerationStructuresKHR vkCmdBuildAccelerationStructuresKHR;
    PFN_vkGetAccelerationStructureDeviceAddressKHR vkGetAccelerationStructureDeviceAddressKHR;
    PFN_vkGetAccelerationStructureBuildSizesKHR vkGetAccelerationStructureBuildSizesKHR;
    PFN_vkDestroyAccelerationStructureKHR vkDestroyAccelerationStructureKHR;

    void loadFunctions();
    uint64_t getBufferDeviceAddress(VkBuffer buffer);
    uint32_t alignedSize(uint32_t value, uint32_t alignment);

    void createBLAS();
    void createTLAS();
    void buildLightBuffer();
    void buildMeshInfoBuffer();

public:
    RayTracingModel(Device &_d);
    ~RayTracingModel();

    RayTracingModel(const RayTracingModel &) = delete;
    RayTracingModel &operator=(const RayTracingModel &) = delete;

    void insertMesh(const std::vector<glm::vec3> &_vertices,
                    const std::vector<glm::uvec3> &_indices,
                    const Material &material,
                    const std::vector<glm::vec3> &_normals = {},
                    const std::vector<glm::vec2> &_texcoords = {});

    void buildAccelerationStructures();

    int getLightCount() const { return lightCount; }
    const std::vector<HitSBTRecord> &getHitSBTRecords() const { return hitSBTRecords; }

    // stageFlags: which shader stages need access (e.g. RAYGEN|CLOSEST_HIT for RT, COMPUTE for wavefront)
    // includeMeshInfo: append meshInfoBuffer binding/info (needed for ray-query-based passes)
    std::vector<DescriptorLayoutBinding> getDescriptorBindings(VkShaderStageFlags stageFlags, bool includeMeshInfo = false) const;
    std::vector<std::vector<DescriptorInfo>> getDescriptorInfos(bool includeMeshInfo = false) const;
};
