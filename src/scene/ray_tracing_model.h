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

    void loadFunctions()
    {
        vkGetBufferDeviceAddressKHR = device.loadDeviceFunction<PFN_vkGetBufferDeviceAddressKHR>("vkGetBufferDeviceAddressKHR");
        vkCreateAccelerationStructureKHR = device.loadDeviceFunction<PFN_vkCreateAccelerationStructureKHR>("vkCreateAccelerationStructureKHR");
        vkCmdBuildAccelerationStructuresKHR = device.loadDeviceFunction<PFN_vkCmdBuildAccelerationStructuresKHR>("vkCmdBuildAccelerationStructuresKHR");
        vkGetAccelerationStructureDeviceAddressKHR = device.loadDeviceFunction<PFN_vkGetAccelerationStructureDeviceAddressKHR>("vkGetAccelerationStructureDeviceAddressKHR");
        vkGetAccelerationStructureBuildSizesKHR = device.loadDeviceFunction<PFN_vkGetAccelerationStructureBuildSizesKHR>("vkGetAccelerationStructureBuildSizesKHR");
        vkDestroyAccelerationStructureKHR = device.loadDeviceFunction<PFN_vkDestroyAccelerationStructureKHR>("vkDestroyAccelerationStructureKHR");
    }

    uint64_t getBufferDeviceAddress(VkBuffer buffer)
    {
        VkBufferDeviceAddressInfoKHR info{};
        info.sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO_KHR;
        info.buffer = buffer;
        return vkGetBufferDeviceAddressKHR(device.getDevice(), &info);
    }

    uint32_t alignedSize(uint32_t value, uint32_t alignment)
    {
        return ((value + alignment - 1) / alignment) * alignment;
    }

    void createBLAS()
    {
        const size_t meshCount = hitSBTRecords.size();

        blas.resize(meshCount);
        blasBuffers.resize(meshCount);
        blasAddresses.resize(meshCount);

        VkTransformMatrixKHR transformMatrix = {
            1.0f, 0.0f, 0.0f, 0.0f,
            0.0f, 1.0f, 0.0f, 0.0f,
            0.0f, 0.0f, 1.0f, 0.0f};
        transformBuffer = std::make_unique<StorageBufferResource>(device, sizeof(transformMatrix),
                                                                  VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT);
        transformBuffer->update(&transformMatrix);

        VkBufferUsageFlags usage = VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR |
                                   VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT |
                                   VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                                   VK_BUFFER_USAGE_TRANSFER_DST_BIT;
        vertexBuffer = std::make_unique<StorageBufferResource>(device, sizeof(glm::vec3) * vertices.size(), usage);
        vertexBuffer->update(vertices.data());
        normalBuffer = std::make_unique<StorageBufferResource>(device, sizeof(glm::vec3) * normals.size(),
                                                                VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT);
        normalBuffer->update(normals.data());
        texcoordBuffer = std::make_unique<StorageBufferResource>(device, sizeof(glm::vec2) * texcoords.size(),
                                                                  VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT);
        texcoordBuffer->update(texcoords.data());
        indexBuffer = std::make_unique<StorageBufferResource>(device, sizeof(glm::uvec3) * indices.size(), usage);
        indexBuffer->update(indices.data());
        materialBuffer = std::make_unique<StorageBufferResource>(device, sizeof(Material) * materials.size(),
                                                                  VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT);
        materialBuffer->update(materials.data());

        std::vector<VkAccelerationStructureGeometryKHR> geometries(meshCount);
        std::vector<VkAccelerationStructureBuildGeometryInfoKHR> buildInfos(meshCount);
        std::vector<VkAccelerationStructureBuildRangeInfoKHR> rangeInfos(meshCount);
        std::vector<VkAccelerationStructureBuildRangeInfoKHR *> rangeInfoPtrs(meshCount);

        std::vector<uint32_t> primitiveCounts(meshCount);
        VkDeviceSize maxScratchSize = 0;
        uint64_t vertexAddress = getBufferDeviceAddress(vertexBuffer->getBuffer());
        uint64_t indexAddress = getBufferDeviceAddress(indexBuffer->getBuffer());

        for (size_t i = 0; i < meshCount; i++)
        {
            VkAccelerationStructureGeometryKHR &geometry = geometries[i];
            geometry = {};
            geometry.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR;
            geometry.flags = VK_GEOMETRY_OPAQUE_BIT_KHR;
            geometry.geometryType = VK_GEOMETRY_TYPE_TRIANGLES_KHR;
            geometry.geometry.triangles.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_TRIANGLES_DATA_KHR;
            geometry.geometry.triangles.vertexFormat = VK_FORMAT_R32G32B32_SFLOAT;
            geometry.geometry.triangles.vertexData.deviceAddress = vertexAddress + hitSBTRecords[i].vertexOffset * sizeof(glm::vec3);
            size_t vertexCount = (i == meshCount - 1) ? vertices.size() - hitSBTRecords[i].vertexOffset : hitSBTRecords[i + 1].vertexOffset - hitSBTRecords[i].vertexOffset;
            geometry.geometry.triangles.maxVertex = static_cast<uint32_t>(vertexCount - 1);
            geometry.geometry.triangles.vertexStride = sizeof(float) * 3;
            geometry.geometry.triangles.indexType = VK_INDEX_TYPE_UINT32;
            geometry.geometry.triangles.indexData.deviceAddress = indexAddress + hitSBTRecords[i].indexOffset * sizeof(glm::uvec3);
            geometry.geometry.triangles.transformData.deviceAddress = getBufferDeviceAddress(transformBuffer->getBuffer());

            VkAccelerationStructureBuildGeometryInfoKHR buildGeometrySizeInfo{};
            buildGeometrySizeInfo.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR;
            buildGeometrySizeInfo.type = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;
            buildGeometrySizeInfo.flags = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR;
            buildGeometrySizeInfo.geometryCount = 1;
            buildGeometrySizeInfo.pGeometries = &geometry;

            VkAccelerationStructureBuildSizesInfoKHR buildSizesInfo{};
            buildSizesInfo.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR;

            size_t primitiveCount = (i == meshCount - 1) ? indices.size() - hitSBTRecords[i].indexOffset : hitSBTRecords[i + 1].indexOffset - hitSBTRecords[i].indexOffset;
            primitiveCounts[i] = static_cast<uint32_t>(primitiveCount);
            vkGetAccelerationStructureBuildSizesKHR(device.getDevice(), VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR, &buildGeometrySizeInfo, &primitiveCounts[i], &buildSizesInfo);

            blasBuffers[i] = std::make_unique<StorageBufferResource>(device, buildSizesInfo.accelerationStructureSize,
                                                                     VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT);

            VkAccelerationStructureCreateInfoKHR createInfo{};
            createInfo.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR;
            createInfo.buffer = blasBuffers[i]->getBuffer();
            createInfo.size = buildSizesInfo.accelerationStructureSize;
            createInfo.type = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;
            if (vkCreateAccelerationStructureKHR(device.getDevice(), &createInfo, nullptr, &blas[i]) != VK_SUCCESS)
                throw std::runtime_error("failed to create BLAS!");

            VkAccelerationStructureBuildGeometryInfoKHR &buildGeometryInfo = buildInfos[i];
            buildGeometryInfo = {};
            buildGeometryInfo.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR;
            buildGeometryInfo.type = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;
            buildGeometryInfo.flags = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR;
            buildGeometryInfo.mode = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;
            buildGeometryInfo.dstAccelerationStructure = blas[i];
            buildGeometryInfo.geometryCount = 1;
            buildGeometryInfo.pGeometries = &geometry;

            VkAccelerationStructureBuildRangeInfoKHR &buildRangeInfo = rangeInfos[i];
            buildRangeInfo = {};
            buildRangeInfo.primitiveCount = primitiveCounts[i];
            rangeInfoPtrs[i] = &buildRangeInfo;

            maxScratchSize = std::max(maxScratchSize, buildSizesInfo.buildScratchSize);
        }

        VkDeviceSize scratchAlignment = device.getPhysicalDeviceASProperties().minAccelerationStructureScratchOffsetAlignment;
        VkDeviceSize alignedScratch = (VkDeviceSize)alignedSize((uint32_t)maxScratchSize, (uint32_t)scratchAlignment);
        StorageBufferResource scratchBuffer{device, alignedScratch * meshCount,
                                            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT};
        VkDeviceAddress scratchAddress = getBufferDeviceAddress(scratchBuffer.getBuffer());

        for (size_t i = 0; i < meshCount; i++)
            buildInfos[i].scratchData.deviceAddress = scratchAddress + i * alignedScratch;

        VkCommandBuffer commandBuffer = device.beginSingleTimeCommands();
        vkCmdBuildAccelerationStructuresKHR(commandBuffer, static_cast<uint32_t>(meshCount), buildInfos.data(), rangeInfoPtrs.data());
        device.endSingleTimeCommands(commandBuffer);

        for (size_t i = 0; i < meshCount; i++)
        {
            VkAccelerationStructureDeviceAddressInfoKHR addressInfo{};
            addressInfo.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_DEVICE_ADDRESS_INFO_KHR;
            addressInfo.accelerationStructure = blas[i];
            blasAddresses[i] = vkGetAccelerationStructureDeviceAddressKHR(device.getDevice(), &addressInfo);
        }
    }

    void createTLAS()
    {
        const size_t instanceCount = blas.size();

        std::vector<VkAccelerationStructureInstanceKHR> instances(instanceCount);
        VkTransformMatrixKHR transform = {
            1.0f, 0.0f, 0.0f, 0.0f,
            0.0f, 1.0f, 0.0f, 0.0f,
            0.0f, 0.0f, 1.0f, 0.0f};

        for (size_t i = 0; i < instanceCount; i++)
        {
            instances[i].transform = transform;
            instances[i].instanceCustomIndex = static_cast<uint32_t>(i);
            instances[i].mask = 0xFF;
            instances[i].instanceShaderBindingTableRecordOffset = static_cast<uint32_t>(i);
            instances[i].flags = VK_GEOMETRY_INSTANCE_TRIANGLE_FACING_CULL_DISABLE_BIT_KHR;
            instances[i].accelerationStructureReference = blasAddresses[i];
        }

        instanceBuffer = std::make_unique<StorageBufferResource>(device, sizeof(VkAccelerationStructureInstanceKHR) * instanceCount,
                                                                  VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT);
        instanceBuffer->update(instances.data());

        VkAccelerationStructureGeometryKHR geometry{};
        geometry.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR;
        geometry.geometryType = VK_GEOMETRY_TYPE_INSTANCES_KHR;
        geometry.flags = VK_GEOMETRY_OPAQUE_BIT_KHR;
        geometry.geometry.instances.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_INSTANCES_DATA_KHR;
        geometry.geometry.instances.arrayOfPointers = VK_FALSE;
        geometry.geometry.instances.data.deviceAddress = getBufferDeviceAddress(instanceBuffer->getBuffer());

        VkAccelerationStructureBuildGeometryInfoKHR sizeInfo{};
        sizeInfo.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR;
        sizeInfo.type = VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR;
        sizeInfo.flags = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR;
        sizeInfo.geometryCount = 1;
        sizeInfo.pGeometries = &geometry;

        uint32_t primitiveCount = static_cast<uint32_t>(instanceCount);

        VkAccelerationStructureBuildSizesInfoKHR buildSizes{};
        buildSizes.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR;
        vkGetAccelerationStructureBuildSizesKHR(device.getDevice(), VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR, &sizeInfo, &primitiveCount, &buildSizes);

        tlasBuffer = std::make_unique<StorageBufferResource>(device, buildSizes.accelerationStructureSize,
                                                              VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT);

        VkAccelerationStructureCreateInfoKHR createInfo{};
        createInfo.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR;
        createInfo.type = VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR;
        createInfo.size = buildSizes.accelerationStructureSize;
        createInfo.buffer = tlasBuffer->getBuffer();

        if (vkCreateAccelerationStructureKHR(device.getDevice(), &createInfo, nullptr, &tlas) != VK_SUCCESS)
            throw std::runtime_error("failed to create TLAS!");

        VkDeviceSize scratchAlignment = device.getPhysicalDeviceASProperties().minAccelerationStructureScratchOffsetAlignment;
        VkDeviceSize scratchSize = (VkDeviceSize)alignedSize((uint32_t)buildSizes.buildScratchSize, (uint32_t)scratchAlignment);

        StorageBufferResource scratchBuffer(device, scratchSize, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT);

        VkAccelerationStructureBuildGeometryInfoKHR buildInfo{};
        buildInfo.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR;
        buildInfo.type = VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR;
        buildInfo.flags = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR;
        buildInfo.mode = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;
        buildInfo.dstAccelerationStructure = tlas;
        buildInfo.geometryCount = 1;
        buildInfo.pGeometries = &geometry;
        buildInfo.scratchData.deviceAddress = getBufferDeviceAddress(scratchBuffer.getBuffer());

        VkAccelerationStructureBuildRangeInfoKHR rangeInfo{};
        rangeInfo.primitiveCount = primitiveCount;
        std::vector<VkAccelerationStructureBuildRangeInfoKHR *> rangeInfos = {&rangeInfo};

        VkCommandBuffer cmd = device.beginSingleTimeCommands();
        vkCmdBuildAccelerationStructuresKHR(cmd, 1, &buildInfo, rangeInfos.data());
        device.endSingleTimeCommands(cmd);
    }

public:
    RayTracingModel(Device &_d) : device(_d)
    {
        loadFunctions();
    }

    ~RayTracingModel()
    {
        if (!finishBuild)
            return;
        for (auto b : blas)
            vkDestroyAccelerationStructureKHR(device.getDevice(), b, nullptr);
        if (tlas != VK_NULL_HANDLE)
            vkDestroyAccelerationStructureKHR(device.getDevice(), tlas, nullptr);
    }

    RayTracingModel(const RayTracingModel &) = delete;
    RayTracingModel &operator=(const RayTracingModel &) = delete;

    void insertMesh(const std::vector<glm::vec3> &_vertices,
                    const std::vector<glm::uvec3> &_indices,
                    const Material &material,
                    const std::vector<glm::vec3> &_normals = {},
                    const std::vector<glm::vec2> &_texcoords = {})
    {
        hitSBTRecords.push_back({static_cast<int>(materials.size()), static_cast<int>(vertices.size()), static_cast<int>(indices.size())});
        vertices.insert(vertices.end(), _vertices.begin(), _vertices.end());
        indices.insert(indices.end(), _indices.begin(), _indices.end());
        materials.push_back(material);

        if (!_normals.empty())
            normals.insert(normals.end(), _normals.begin(), _normals.end());
        else
            normals.resize(vertices.size(), glm::vec3(0.0f, 0.0f, 1.0f));

        if (!_texcoords.empty())
            texcoords.insert(texcoords.end(), _texcoords.begin(), _texcoords.end());
        else
            texcoords.resize(vertices.size(), glm::vec2(0.0f));
    }

    void buildAccelerationStructures()
    {
        if (finishBuild)
            throw std::runtime_error("Acceleration structures already built!");
        createBLAS();
        createTLAS();
        finishBuild = true;
    }

    const std::vector<HitSBTRecord> &getHitSBTRecords() const { return hitSBTRecords; }

    std::vector<DescriptorLayoutBinding> getDescriptorLayoutBindings() const
    {
        VkShaderStageFlags hitStages = VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR | VK_SHADER_STAGE_RAYGEN_BIT_KHR;
        return {
            {VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR, VK_SHADER_STAGE_RAYGEN_BIT_KHR},
            {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, hitStages}, // vertices
            {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, hitStages}, // indices
            {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, hitStages}, // materials
            {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, hitStages}, // normals
            {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, hitStages}, // texcoords
        };
    }

    std::vector<std::vector<DescriptorInfo>> getDescriptorInfos() const
    {
        return {
            {VkWriteDescriptorSetAccelerationStructureKHR{
                VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET_ACCELERATION_STRUCTURE_KHR, nullptr, 1, &tlas}},
            {VkDescriptorBufferInfo{vertexBuffer->getBuffer(), 0, VK_WHOLE_SIZE}},
            {VkDescriptorBufferInfo{indexBuffer->getBuffer(), 0, VK_WHOLE_SIZE}},
            {VkDescriptorBufferInfo{materialBuffer->getBuffer(), 0, VK_WHOLE_SIZE}},
            {VkDescriptorBufferInfo{normalBuffer->getBuffer(), 0, VK_WHOLE_SIZE}},
            {VkDescriptorBufferInfo{texcoordBuffer->getBuffer(), 0, VK_WHOLE_SIZE}},
        };
    }
};
