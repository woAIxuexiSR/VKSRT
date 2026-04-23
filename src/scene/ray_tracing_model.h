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
    std::vector<glm::mat4> instanceTransforms;

    std::unique_ptr<StorageBufferResource> vertexBuffer;
    std::unique_ptr<StorageBufferResource> normalBuffer;
    std::unique_ptr<StorageBufferResource> texcoordBuffer;
    std::unique_ptr<StorageBufferResource> indexBuffer;
    std::unique_ptr<StorageBufferResource> materialBuffer;
    std::unique_ptr<StorageBufferResource> blasTransformBuffer;
    std::unique_ptr<StorageBufferResource> instanceBuffer;
    std::unique_ptr<StorageBufferResource> lightBuffer;
    std::unique_ptr<StorageBufferResource> instanceTransformBuffer;
    std::unique_ptr<StorageBufferResource> meshInfoBuffer;
    std::unique_ptr<StorageBufferResource> surfaceTriangleBuffer;
    int lightCount{0};
    int surfaceTriangleCount{0};
    float totalSurfaceArea{0.0f};

    std::vector<VkAccelerationStructureKHR> blas;
    std::vector<std::unique_ptr<StorageBufferResource>> blasBuffers;
    std::vector<uint64_t> blasAddresses;

    VkAccelerationStructureKHR tlas{VK_NULL_HANDLE};
    std::unique_ptr<StorageBufferResource> tlasBuffer;

    PFN_vkCreateAccelerationStructureKHR vkCreateAccelerationStructureKHR;
    PFN_vkCmdBuildAccelerationStructuresKHR vkCmdBuildAccelerationStructuresKHR;
    PFN_vkGetAccelerationStructureDeviceAddressKHR vkGetAccelerationStructureDeviceAddressKHR;
    PFN_vkGetAccelerationStructureBuildSizesKHR vkGetAccelerationStructureBuildSizesKHR;
    PFN_vkDestroyAccelerationStructureKHR vkDestroyAccelerationStructureKHR;

    void loadFunctions();
    uint32_t alignedSize(uint32_t value, uint32_t alignment);

    void createBLAS();
    void createTLAS();
    void buildLightBuffer();
    void buildInstanceTransformBuffer();
    void buildMeshInfoBuffer();
    void buildSurfaceSampler();

public:
    RayTracingModel(Device &_d);
    ~RayTracingModel();

    RayTracingModel(const RayTracingModel &) = delete;
    RayTracingModel &operator=(const RayTracingModel &) = delete;

    void insertMesh(const std::vector<glm::vec3> &_vertices,
                    const std::vector<glm::uvec3> &_indices,
                    const Material &material,
                    const std::vector<glm::vec3> &_normals = {},
                    const std::vector<glm::vec2> &_texcoords = {},
                    const glm::mat4 &instanceTransform = glm::mat4(1.0f));

    void buildAccelerationStructures();

    int getLightCount() const { return lightCount; }
    int getSurfaceTriangleCount() const { return surfaceTriangleCount; }
    float getTotalSurfaceArea() const { return totalSurfaceArea; }
    const std::vector<HitSBTRecord> &getHitSBTRecords() const { return hitSBTRecords; }

    // stageFlags: which shader stages need access (e.g. RAYGEN|CLOSEST_HIT for RT, COMPUTE for wavefront)
    // includeMeshInfo: append meshInfoBuffer binding/info (needed for ray-query-based passes)
    // includeSurfaceSampler: append surfaceTriangleBuffer (Neural Radiosity surface sampling)
    std::vector<DescriptorLayoutBinding> getDescriptorBindings(VkShaderStageFlags stageFlags,
                                                               bool includeMeshInfo = false,
                                                               bool includeSurfaceSampler = false) const;
    std::vector<std::vector<DescriptorInfo>> getDescriptorInfos(bool includeMeshInfo = false,
                                                                bool includeSurfaceSampler = false) const;
};
