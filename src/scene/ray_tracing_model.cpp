#include "ray_tracing_model.h"

#include <stdexcept>
#include <algorithm>

// --- Private helpers ---

void RayTracingModel::loadFunctions()
{
    vkCreateAccelerationStructureKHR = device.loadDeviceFunction<PFN_vkCreateAccelerationStructureKHR>("vkCreateAccelerationStructureKHR");
    vkCmdBuildAccelerationStructuresKHR = device.loadDeviceFunction<PFN_vkCmdBuildAccelerationStructuresKHR>("vkCmdBuildAccelerationStructuresKHR");
    vkGetAccelerationStructureDeviceAddressKHR = device.loadDeviceFunction<PFN_vkGetAccelerationStructureDeviceAddressKHR>("vkGetAccelerationStructureDeviceAddressKHR");
    vkGetAccelerationStructureBuildSizesKHR = device.loadDeviceFunction<PFN_vkGetAccelerationStructureBuildSizesKHR>("vkGetAccelerationStructureBuildSizesKHR");
    vkDestroyAccelerationStructureKHR = device.loadDeviceFunction<PFN_vkDestroyAccelerationStructureKHR>("vkDestroyAccelerationStructureKHR");
}

uint32_t RayTracingModel::alignedSize(uint32_t value, uint32_t alignment)
{
    return ((value + alignment - 1) / alignment) * alignment;
}

// --- Light buffer ---

void RayTracingModel::buildLightBuffer()
{
    std::vector<LightTriangle> lights;

    for (size_t meshIdx = 0; meshIdx < hitSBTRecords.size(); meshIdx++)
    {
        if (materials[meshIdx].type != MAT_EMISSIVE)
            continue;

        int vOff = hitSBTRecords[meshIdx].vertexOffset;
        int iOff = hitSBTRecords[meshIdx].indexOffset;
        int triCount;
        if (meshIdx + 1 < hitSBTRecords.size())
            triCount = hitSBTRecords[meshIdx + 1].indexOffset - iOff;
        else
            triCount = static_cast<int>(indices.size()) - iOff;

        glm::mat4 xform = instanceTransforms[meshIdx];

        for (int t = 0; t < triCount; t++)
        {
            glm::uvec3 tri = indices[iOff + t];
            glm::vec3 v0 = glm::vec3(xform * glm::vec4(vertices[tri.x + vOff], 1.0f));
            glm::vec3 v1 = glm::vec3(xform * glm::vec4(vertices[tri.y + vOff], 1.0f));
            glm::vec3 v2 = glm::vec3(xform * glm::vec4(vertices[tri.z + vOff], 1.0f));
            float area = 0.5f * glm::length(glm::cross(v1 - v0, v2 - v0));

            LightTriangle lt{};
            lt.area = area;
            lt.indexOffset = iOff + t;
            lt.vertexOffset = vOff;
            lt.matIndex = hitSBTRecords[meshIdx].materialIndex;
            lt.instanceIndex = static_cast<int>(meshIdx);
            lights.push_back(lt);
        }
    }

    lightCount = static_cast<int>(lights.size());
    if (lights.empty())
        lights.push_back(LightTriangle{});

    lightBuffer = std::make_unique<StorageBufferResource>(
        device, sizeof(LightTriangle) * lights.size(),
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT);
    lightBuffer->update(lights.data());
}

// --- Instance transform buffer ---

void RayTracingModel::buildInstanceTransformBuffer()
{
    // Slang StructuredBuffer<float3x4> uses column-major std430 layout:
    // 4 columns × vec3 (padded to 16 bytes each) = 64 bytes per matrix
    // Column c, row r = m[c][r] in glm (also column-major)
    size_t count = instanceTransforms.size();
    std::vector<float> data(count * 16); // 16 floats per matrix (4 cols × 4 floats with padding)
    for (size_t i = 0; i < count; i++)
    {
        const glm::mat4 &m = instanceTransforms[i];
        for (int col = 0; col < 4; col++)
        {
            data[i * 16 + col * 4 + 0] = m[col][0]; // row 0
            data[i * 16 + col * 4 + 1] = m[col][1]; // row 1
            data[i * 16 + col * 4 + 2] = m[col][2]; // row 2
            data[i * 16 + col * 4 + 3] = 0.0f;      // padding
        }
    }
    instanceTransformBuffer = std::make_unique<StorageBufferResource>(
        device, sizeof(float) * 16 * count,
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT);
    instanceTransformBuffer->update(data.data());
}

// --- MeshInfo buffer ---

void RayTracingModel::buildMeshInfoBuffer()
{
    // HitSBTRecord layout matches MeshInfo (matIndex, vertexOffset, indexOffset) - upload directly
    meshInfoBuffer = std::make_unique<StorageBufferResource>(
        device, sizeof(HitSBTRecord) * hitSBTRecords.size(),
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT);
    meshInfoBuffer->update(hitSBTRecords.data());
}

// --- BLAS ---

void RayTracingModel::createBLAS()
{
    const size_t meshCount = hitSBTRecords.size();

    blas.resize(meshCount);
    blasBuffers.resize(meshCount);
    blasAddresses.resize(meshCount);

    VkTransformMatrixKHR transformMatrix = {
        1.0f, 0.0f, 0.0f, 0.0f,
        0.0f, 1.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 1.0f, 0.0f};
    blasTransformBuffer = std::make_unique<StorageBufferResource>(device, sizeof(transformMatrix),
                                                              VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT);
    blasTransformBuffer->update(&transformMatrix);

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
    uint64_t vertexAddress = device.getBufferDeviceAddress(vertexBuffer->getBuffer());
    uint64_t indexAddress = device.getBufferDeviceAddress(indexBuffer->getBuffer());

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
        geometry.geometry.triangles.transformData.deviceAddress = device.getBufferDeviceAddress(blasTransformBuffer->getBuffer());

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
    VkDeviceAddress scratchAddress = device.getBufferDeviceAddress(scratchBuffer.getBuffer());

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

// --- TLAS ---

void RayTracingModel::createTLAS()
{
    const size_t instanceCount = blas.size();

    std::vector<VkAccelerationStructureInstanceKHR> instances(instanceCount);

    for (size_t i = 0; i < instanceCount; i++)
    {
        // Convert glm::mat4 (column-major) to VkTransformMatrixKHR (row-major 3x4)
        const glm::mat4 &m = instanceTransforms[i];
        VkTransformMatrixKHR vkXform;
        for (int row = 0; row < 3; row++)
            for (int col = 0; col < 4; col++)
                vkXform.matrix[row][col] = m[col][row];

        instances[i].transform = vkXform;
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
    geometry.geometry.instances.data.deviceAddress = device.getBufferDeviceAddress(instanceBuffer->getBuffer());

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
    buildInfo.scratchData.deviceAddress = device.getBufferDeviceAddress(scratchBuffer.getBuffer());

    VkAccelerationStructureBuildRangeInfoKHR rangeInfo{};
    rangeInfo.primitiveCount = primitiveCount;
    std::vector<VkAccelerationStructureBuildRangeInfoKHR *> rangeInfos = {&rangeInfo};

    VkCommandBuffer cmd = device.beginSingleTimeCommands();
    vkCmdBuildAccelerationStructuresKHR(cmd, 1, &buildInfo, rangeInfos.data());
    device.endSingleTimeCommands(cmd);
}

// --- Public methods ---

RayTracingModel::RayTracingModel(Device &_d) : device(_d)
{
    loadFunctions();
}

RayTracingModel::~RayTracingModel()
{
    if (!finishBuild)
        return;
    for (auto b : blas)
        vkDestroyAccelerationStructureKHR(device.getDevice(), b, nullptr);
    if (tlas != VK_NULL_HANDLE)
        vkDestroyAccelerationStructureKHR(device.getDevice(), tlas, nullptr);
}

void RayTracingModel::insertMesh(const std::vector<glm::vec3> &_vertices,
                                  const std::vector<glm::uvec3> &_indices,
                                  const Material &material,
                                  const std::vector<glm::vec3> &_normals,
                                  const std::vector<glm::vec2> &_texcoords,
                                  const glm::mat4 &instanceTransform)
{
    hitSBTRecords.push_back({static_cast<int>(materials.size()), static_cast<int>(vertices.size()), static_cast<int>(indices.size())});
    vertices.insert(vertices.end(), _vertices.begin(), _vertices.end());
    indices.insert(indices.end(), _indices.begin(), _indices.end());
    materials.push_back(material);
    instanceTransforms.push_back(instanceTransform);

    if (!_normals.empty())
        normals.insert(normals.end(), _normals.begin(), _normals.end());
    else
        normals.resize(vertices.size(), glm::vec3(0.0f, 0.0f, 1.0f));

    if (!_texcoords.empty())
        texcoords.insert(texcoords.end(), _texcoords.begin(), _texcoords.end());
    else
        texcoords.resize(vertices.size(), glm::vec2(0.0f));
}

void RayTracingModel::buildAccelerationStructures()
{
    if (finishBuild)
        throw std::runtime_error("Acceleration structures already built!");
    createBLAS();
    createTLAS();
    buildLightBuffer();
    buildInstanceTransformBuffer();
    buildMeshInfoBuffer();
    finishBuild = true;
}

// --- Descriptor helpers ---

std::vector<DescriptorLayoutBinding> RayTracingModel::getDescriptorBindings(VkShaderStageFlags stageFlags, bool includeMeshInfo) const
{
    std::vector<DescriptorLayoutBinding> bindings = {
        {VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR, stageFlags}, // TLAS
        {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, stageFlags},             // vertices
        {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, stageFlags},             // indices
        {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, stageFlags},             // materials
        {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, stageFlags},             // normals
        {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, stageFlags},             // texcoords
        {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, stageFlags},             // lights
        {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, stageFlags},             // instanceTransforms
    };
    if (includeMeshInfo)
        bindings.push_back({VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, stageFlags}); // meshInfo
    return bindings;
}

std::vector<std::vector<DescriptorInfo>> RayTracingModel::getDescriptorInfos(bool includeMeshInfo) const
{
    std::vector<std::vector<DescriptorInfo>> infos = {
        {VkWriteDescriptorSetAccelerationStructureKHR{
            VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET_ACCELERATION_STRUCTURE_KHR, nullptr, 1, &tlas}},
        {VkDescriptorBufferInfo{vertexBuffer->getBuffer(), 0, VK_WHOLE_SIZE}},
        {VkDescriptorBufferInfo{indexBuffer->getBuffer(), 0, VK_WHOLE_SIZE}},
        {VkDescriptorBufferInfo{materialBuffer->getBuffer(), 0, VK_WHOLE_SIZE}},
        {VkDescriptorBufferInfo{normalBuffer->getBuffer(), 0, VK_WHOLE_SIZE}},
        {VkDescriptorBufferInfo{texcoordBuffer->getBuffer(), 0, VK_WHOLE_SIZE}},
        {VkDescriptorBufferInfo{lightBuffer->getBuffer(), 0, VK_WHOLE_SIZE}},
        {VkDescriptorBufferInfo{instanceTransformBuffer->getBuffer(), 0, VK_WHOLE_SIZE}},
    };
    if (includeMeshInfo)
        infos.push_back({VkDescriptorBufferInfo{meshInfoBuffer->getBuffer(), 0, VK_WHOLE_SIZE}});
    return infos;
}
