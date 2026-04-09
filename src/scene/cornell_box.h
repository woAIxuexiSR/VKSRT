#pragma once

#include "device.h"
#include "resource.h"
#include "pipeline.h"

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <vector>
#include <memory>

class CornellBox
{
private:
    Device &device;

    std::vector<glm::vec3> vertices;
    std::vector<glm::uvec3> indices;
    std::vector<HitSBTRecord> hitSBTRecords;
    std::vector<glm::vec4> materials; // rgb=color, a=emission

    std::unique_ptr<StorageBufferResource> vertexBuffer;
    std::unique_ptr<StorageBufferResource> indexBuffer;
    std::unique_ptr<StorageBufferResource> materialBuffer;
    std::unique_ptr<StorageBufferResource> hitRecordBuffer;
    std::unique_ptr<StorageBufferResource> transformBuffer;
    std::unique_ptr<StorageBufferResource> instanceBuffer;

    std::vector<VkAccelerationStructureKHR> blas;
    std::vector<std::unique_ptr<StorageBufferResource>> blasBuffers;
    std::vector<uint64_t> blasAddresses;

    VkAccelerationStructureKHR tlas;
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

    void insertMesh(const std::vector<glm::vec3> &_vertices, const std::vector<glm::uvec3> &_indices, glm::vec4 material)
    {
        hitSBTRecords.push_back({static_cast<int>(materials.size()), static_cast<int>(vertices.size()), static_cast<int>(indices.size())});
        vertices.insert(vertices.end(), _vertices.begin(), _vertices.end());
        indices.insert(indices.end(), _indices.begin(), _indices.end());
        materials.push_back(material);
    }

    void insertPlaneXY(const glm::vec3 center, const glm::vec2 size, const glm::vec4 color)
    {
        glm::vec2 half = size * 0.5f;
        std::vector<glm::vec3> v = {{center.x + half.x, center.y + half.y, center.z},
                                    {center.x - half.x, center.y + half.y, center.z},
                                    {center.x - half.x, center.y - half.y, center.z},
                                    {center.x + half.x, center.y - half.y, center.z}};
        std::vector<glm::uvec3> idx = {{0, 1, 2}, {0, 2, 3}};
        insertMesh(v, idx, color);
    }

    void insertPlaneYZ(const glm::vec3 center, const glm::vec2 size, const glm::vec4 color)
    {
        glm::vec2 half = size * 0.5f;
        std::vector<glm::vec3> v = {{center.x, center.y + half.x, center.z + half.y},
                                    {center.x, center.y - half.x, center.z + half.y},
                                    {center.x, center.y - half.x, center.z - half.y},
                                    {center.x, center.y + half.x, center.z - half.y}};
        std::vector<glm::uvec3> idx = {{0, 1, 2}, {0, 2, 3}};
        insertMesh(v, idx, color);
    }

    void insertPlaneXZ(const glm::vec3 center, const glm::vec2 size, const glm::vec4 color)
    {
        glm::vec2 half = size * 0.5f;
        std::vector<glm::vec3> v = {{center.x + half.x, center.y, center.z + half.y},
                                    {center.x - half.x, center.y, center.z + half.y},
                                    {center.x - half.x, center.y, center.z - half.y},
                                    {center.x + half.x, center.y, center.z - half.y}};
        std::vector<glm::uvec3> idx = {{0, 1, 2}, {0, 2, 3}};
        insertMesh(v, idx, color);
    }

    void insertBox(const glm::vec3 center, const glm::vec3 size, const glm::vec4 color, float yaw = 0.0f, float pitch = 0.0f, float roll = 0.0f)
    {
        glm::vec3 half = size * 0.5f;
        std::vector<glm::vec3> v = {{half.x, half.y, half.z},
                                    {-half.x, half.y, half.z},
                                    {-half.x, -half.y, half.z},
                                    {half.x, -half.y, half.z},
                                    {half.x, half.y, -half.z},
                                    {-half.x, half.y, -half.z},
                                    {-half.x, -half.y, -half.z},
                                    {half.x, -half.y, -half.z}};
        glm::mat4 rot = glm::rotate(glm::mat4(1.0f), yaw, glm::vec3(0.0f, 0.0f, 1.0f)) *
                        glm::rotate(glm::mat4(1.0f), pitch, glm::vec3(1.0f, 0.0f, 0.0f)) *
                        glm::rotate(glm::mat4(1.0f), roll, glm::vec3(0.0f, 1.0f, 0.0f));
        for (auto &vert : v)
            vert = glm::vec3(rot * glm::vec4(vert, 1.0f)) + center;

        std::vector<glm::uvec3> idx = {{0, 1, 2}, {0, 2, 3}, {4, 5, 6}, {4, 6, 7},
                                        {0, 1, 5}, {0, 5, 4}, {1, 2, 6}, {1, 6, 5},
                                        {2, 3, 7}, {2, 7, 6}, {3, 0, 4}, {3, 4, 7}};
        insertMesh(v, idx, color);
    }

    void createScene()
    {
        float roomWidth = 2.76f;
        float roomDepth = 2.56f;
        float roomHeight = 2.29f;

        glm::vec4 white(0.9f, 0.9f, 0.9f, 0.0f);
        glm::vec4 gray(0.73f, 0.73f, 0.73f, 0.0f);
        glm::vec4 red(0.63f, 0.065f, 0.05f, 0.0f);
        glm::vec4 green(0.14f, 0.45f, 0.091f, 0.0f);
        glm::vec4 lightColor(15.0f, 15.0f, 15.0f, 1.0f);

        insertPlaneXY(glm::vec3(0.0f, 0.0f, 0.0f), glm::vec2(roomWidth, roomDepth), white);
        insertPlaneXY(glm::vec3(0.0f, 0.0f, roomHeight), glm::vec2(roomWidth, roomDepth), white);
        insertPlaneYZ(glm::vec3(-roomWidth / 2, 0.0f, roomHeight / 2), glm::vec2(roomDepth, roomHeight), red);
        insertPlaneYZ(glm::vec3(roomWidth / 2, 0.0f, roomHeight / 2), glm::vec2(roomDepth, roomHeight), green);
        insertPlaneXZ(glm::vec3(0.0f, roomDepth / 2, roomHeight / 2), glm::vec2(roomWidth, roomHeight), white);

        insertPlaneXY(glm::vec3(0.0f, 0.0f, roomHeight - 0.01f), glm::vec2(0.5f, 0.5f), lightColor);

        insertBox(glm::vec3(-0.5f, -0.3f, 0.3f), glm::vec3(0.6f, 0.6f, 0.6f), gray, glm::radians(15.0f));
        insertBox(glm::vec3(0.5f, 0.4f, 0.6f), glm::vec3(0.6f, 0.6f, 1.2f), gray, glm::radians(-18.0f));
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
        indexBuffer = std::make_unique<StorageBufferResource>(device, sizeof(glm::uvec3) * indices.size(), usage);
        indexBuffer->update(indices.data());
        materialBuffer = std::make_unique<StorageBufferResource>(device, sizeof(glm::vec4) * materials.size(),
                                                                  VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT);
        materialBuffer->update(materials.data());

        // Build hit record buffer for shader access (padded to 16 bytes per record)
        struct alignas(16) PaddedHitRecord { int matIndex; int vertexOffset; int indexOffset; int _pad; };
        std::vector<PaddedHitRecord> paddedRecords(meshCount);
        for (size_t i = 0; i < meshCount; i++)
            paddedRecords[i] = {hitSBTRecords[i].materialIndex, hitSBTRecords[i].vertexOffset, hitSBTRecords[i].indexOffset, 0};
        hitRecordBuffer = std::make_unique<StorageBufferResource>(device, sizeof(PaddedHitRecord) * meshCount,
                                                                   VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT);
        hitRecordBuffer->update(paddedRecords.data());

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
    CornellBox(Device &_d) : device(_d), tlas(VK_NULL_HANDLE)
    {
        loadFunctions();
        createScene();
        createBLAS();
        createTLAS();
    }

    ~CornellBox()
    {
        for (auto b : blas)
            vkDestroyAccelerationStructureKHR(device.getDevice(), b, nullptr);
        if (tlas != VK_NULL_HANDLE)
            vkDestroyAccelerationStructureKHR(device.getDevice(), tlas, nullptr);
    }

    CornellBox(const CornellBox &) = delete;
    CornellBox &operator=(const CornellBox &) = delete;

    const std::vector<HitSBTRecord> &getHitSBTRecords() const { return hitSBTRecords; }

    std::vector<DescriptorLayoutBinding> getDescriptorLayoutBindings() const
    {
        return {
            {VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR, VK_SHADER_STAGE_RAYGEN_BIT_KHR},
            {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR | VK_SHADER_STAGE_RAYGEN_BIT_KHR},
            {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR | VK_SHADER_STAGE_RAYGEN_BIT_KHR},
            {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR | VK_SHADER_STAGE_RAYGEN_BIT_KHR},
            {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR},
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
            {VkDescriptorBufferInfo{hitRecordBuffer->getBuffer(), 0, VK_WHOLE_SIZE}},
        };
    }
};
