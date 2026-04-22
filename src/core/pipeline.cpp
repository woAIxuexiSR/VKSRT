#include "pipeline.h"

// ---- Pipeline base ----

void Pipeline::createDescriptorSetLayout()
{
    std::vector<VkDescriptorSetLayoutBinding> bindings(descriptorBindings.size());
    for (size_t i = 0; i < descriptorBindings.size(); i++)
    {
        bindings[i].binding = static_cast<uint32_t>(i);
        bindings[i].descriptorCount = 1;
        bindings[i].descriptorType = descriptorBindings[i].type;
        bindings[i].pImmutableSamplers = nullptr;
        bindings[i].stageFlags = descriptorBindings[i].flags;
    }

    VkDescriptorSetLayoutCreateInfo layoutInfo{};
    layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    layoutInfo.bindingCount = static_cast<uint32_t>(bindings.size());
    layoutInfo.pBindings = bindings.data();

    if (vkCreateDescriptorSetLayout(device.getDevice(), &layoutInfo, nullptr, &descriptorSetLayout) != VK_SUCCESS)
        throw std::runtime_error("failed to create descriptor set layout!");
}

void Pipeline::createDescriptorPool()
{
    if (descriptorBindings.empty())
    {
        descriptorPool = VK_NULL_HANDLE;
        return;
    }

    std::vector<VkDescriptorPoolSize> poolSizes(descriptorBindings.size());
    for (size_t i = 0; i < descriptorBindings.size(); i++)
    {
        poolSizes[i].type = descriptorBindings[i].type;
        poolSizes[i].descriptorCount = descriptorSetCount;
    }

    VkDescriptorPoolCreateInfo poolInfo{};
    poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    poolInfo.poolSizeCount = static_cast<uint32_t>(poolSizes.size());
    poolInfo.pPoolSizes = poolSizes.data();
    poolInfo.maxSets = descriptorSetCount;

    if (vkCreateDescriptorPool(device.getDevice(), &poolInfo, nullptr, &descriptorPool) != VK_SUCCESS)
        throw std::runtime_error("failed to create descriptor pool!");
}

void Pipeline::allocateDescriptorSets()
{
    if (descriptorPool == VK_NULL_HANDLE)
        return;

    std::vector<VkDescriptorSetLayout> layouts(descriptorSetCount, descriptorSetLayout);
    VkDescriptorSetAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    allocInfo.descriptorPool = descriptorPool;
    allocInfo.descriptorSetCount = descriptorSetCount;
    allocInfo.pSetLayouts = layouts.data();

    descriptorSets.resize(descriptorSetCount);
    if (vkAllocateDescriptorSets(device.getDevice(), &allocInfo, descriptorSets.data()) != VK_SUCCESS)
        throw std::runtime_error("failed to allocate descriptor sets!");
}

void Pipeline::updateDescriptorSets(const std::vector<std::vector<DescriptorInfo>> &infos)
{
    assert(infos.size() == descriptorBindings.size());

    for (size_t i = 0; i < descriptorSetCount; i++)
    {
        std::vector<VkWriteDescriptorSet> writes(descriptorBindings.size());
        for (size_t j = 0; j < descriptorBindings.size(); j++)
        {
            writes[j].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            writes[j].dstSet = descriptorSets[i];
            writes[j].dstBinding = static_cast<uint32_t>(j);
            writes[j].dstArrayElement = 0;
            writes[j].descriptorType = descriptorBindings[j].type;
            writes[j].descriptorCount = 1;

            auto &info = infos[j].size() == descriptorSetCount ? infos[j][i] : infos[j][0];

            std::visit([&writes, j](auto &&arg)
                       {
                using T = std::decay_t<decltype(arg)>;
                if constexpr (std::is_same_v<T, VkDescriptorBufferInfo>)
                    writes[j].pBufferInfo = &arg;
                else if constexpr (std::is_same_v<T, VkDescriptorImageInfo>)
                    writes[j].pImageInfo = &arg;
                else if constexpr (std::is_same_v<T, VkWriteDescriptorSetAccelerationStructureKHR>)
                    writes[j].pNext = &arg; }, info);
        }

        vkUpdateDescriptorSets(device.getDevice(), static_cast<uint32_t>(writes.size()), writes.data(), 0, nullptr);
    }
}

std::vector<char> Pipeline::readFile(const std::string &filename)
{
    std::ifstream file(filename, std::ios::ate | std::ios::binary);
    if (!file.is_open())
        throw std::runtime_error("failed to open file " + filename + "!");

    size_t fileSize = (size_t)file.tellg();
    std::vector<char> buffer(fileSize);
    file.seekg(0);
    file.read(buffer.data(), fileSize);
    file.close();
    return buffer;
}

VkShaderModule Pipeline::createShaderModule(const std::vector<char> &code)
{
    VkShaderModuleCreateInfo createInfo{};
    createInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    createInfo.codeSize = code.size();
    createInfo.pCode = reinterpret_cast<const uint32_t *>(code.data());

    VkShaderModule shaderModule;
    if (vkCreateShaderModule(device.getDevice(), &createInfo, nullptr, &shaderModule) != VK_SUCCESS)
        throw std::runtime_error("failed to create shader module!");

    return shaderModule;
}

Pipeline::Pipeline(Device &_d, uint32_t _cnt, const std::vector<DescriptorLayoutBinding> &bindings)
    : device(_d), descriptorSetCount(_cnt), descriptorBindings(bindings),
      pipelineLayout(VK_NULL_HANDLE), pipeline(VK_NULL_HANDLE),
      descriptorSetLayout(VK_NULL_HANDLE), descriptorPool(VK_NULL_HANDLE)
{
    createDescriptorSetLayout();
    createDescriptorPool();
    allocateDescriptorSets();
}

Pipeline::~Pipeline()
{
    vkDestroyPipeline(device.getDevice(), pipeline, nullptr);
    vkDestroyPipelineLayout(device.getDevice(), pipelineLayout, nullptr);
    vkDestroyDescriptorSetLayout(device.getDevice(), descriptorSetLayout, nullptr);
    if (descriptorPool != VK_NULL_HANDLE)
        vkDestroyDescriptorPool(device.getDevice(), descriptorPool, nullptr);
}

void Pipeline::loadPipelineCache(Device &device, const std::string &filename)
{
    std::vector<char> cacheData;
    std::ifstream file(filename, std::ios::ate | std::ios::binary);
    if (file.is_open())
    {
        size_t fileSize = (size_t)file.tellg();
        cacheData.resize(fileSize);
        file.seekg(0);
        file.read(cacheData.data(), fileSize);
        file.close();
        std::cout << "Pipeline cache loaded from " << filename << " (" << fileSize << " bytes)" << std::endl;
    }

    VkPipelineCacheCreateInfo cacheCreateInfo{};
    cacheCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO;
    cacheCreateInfo.initialDataSize = cacheData.size();
    cacheCreateInfo.pInitialData = cacheData.empty() ? nullptr : cacheData.data();

    if (vkCreatePipelineCache(device.getDevice(), &cacheCreateInfo, nullptr, &pipelineCache) != VK_SUCCESS)
        throw std::runtime_error("failed to create pipeline cache!");
}

void Pipeline::savePipelineCache(Device &device, const std::string &filename)
{
    if (pipelineCache == VK_NULL_HANDLE)
        return;

    size_t dataSize = 0;
    vkGetPipelineCacheData(device.getDevice(), pipelineCache, &dataSize, nullptr);
    if (dataSize == 0)
        return;

    std::vector<char> cacheData(dataSize);
    if (vkGetPipelineCacheData(device.getDevice(), pipelineCache, &dataSize, cacheData.data()) != VK_SUCCESS)
    {
        std::cout << "Failed to retrieve pipeline cache data" << std::endl;
        return;
    }

    std::ofstream file(filename, std::ios::binary);
    if (file.is_open())
    {
        file.write(cacheData.data(), dataSize);
        file.close();
        std::cout << "Pipeline cache saved to " << filename << " (" << dataSize << " bytes)" << std::endl;
    }
    else
        std::cout << "Failed to save pipeline cache to " << filename << std::endl;
}

void Pipeline::destroyPipelineCache(Device &device)
{
    if (pipelineCache != VK_NULL_HANDLE)
    {
        vkDestroyPipelineCache(device.getDevice(), pipelineCache, nullptr);
        pipelineCache = VK_NULL_HANDLE;
    }
}

// ---- GraphicsPipeline ----

void GraphicsPipeline::createPipeline()
{
    auto shaderCode = readFile(spvPath);
    VkShaderModule shaderModule = createShaderModule(shaderCode);

    VkPipelineShaderStageCreateInfo vertShaderStageInfo{};
    vertShaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    vertShaderStageInfo.stage = VK_SHADER_STAGE_VERTEX_BIT;
    vertShaderStageInfo.module = shaderModule;
    vertShaderStageInfo.pName = vertexEntry.c_str();

    VkPipelineShaderStageCreateInfo fragShaderStageInfo{};
    fragShaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    fragShaderStageInfo.stage = VK_SHADER_STAGE_FRAGMENT_BIT;
    fragShaderStageInfo.module = shaderModule;
    fragShaderStageInfo.pName = fragmentEntry.c_str();

    VkPipelineShaderStageCreateInfo shaderStages[] = {vertShaderStageInfo, fragShaderStageInfo};

    VkPipelineVertexInputStateCreateInfo vertexInputInfo{};
    vertexInputInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
    vertexInputInfo.vertexBindingDescriptionCount = static_cast<uint32_t>(bindingDescriptions.size());
    vertexInputInfo.vertexAttributeDescriptionCount = static_cast<uint32_t>(attributeDescriptions.size());
    vertexInputInfo.pVertexBindingDescriptions = bindingDescriptions.data();
    vertexInputInfo.pVertexAttributeDescriptions = attributeDescriptions.data();

    VkPipelineInputAssemblyStateCreateInfo inputAssembly{};
    inputAssembly.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
    inputAssembly.topology = topology;
    inputAssembly.primitiveRestartEnable = VK_FALSE;

    VkPipelineViewportStateCreateInfo viewportState{};
    viewportState.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
    viewportState.viewportCount = 1;
    viewportState.scissorCount = 1;

    VkPipelineRasterizationStateCreateInfo rasterizer{};
    rasterizer.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
    rasterizer.depthClampEnable = VK_FALSE;
    rasterizer.rasterizerDiscardEnable = VK_FALSE;
    rasterizer.polygonMode = VK_POLYGON_MODE_FILL;
    rasterizer.lineWidth = 1.0f;
    rasterizer.cullMode = VK_CULL_MODE_NONE;
    rasterizer.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
    rasterizer.depthBiasEnable = VK_FALSE;

    VkPipelineMultisampleStateCreateInfo multisampling{};
    multisampling.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
    multisampling.sampleShadingEnable = VK_FALSE;
    multisampling.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

    VkPipelineDepthStencilStateCreateInfo depthStencil{};
    depthStencil.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
    depthStencil.depthTestEnable = VK_TRUE;
    depthStencil.depthWriteEnable = VK_TRUE;
    depthStencil.depthCompareOp = VK_COMPARE_OP_LESS;
    depthStencil.depthBoundsTestEnable = VK_FALSE;
    depthStencil.stencilTestEnable = VK_FALSE;

    std::vector<VkPipelineColorBlendAttachmentState> colorBlendAttachments(targetFormat.size());
    for (size_t i = 0; i < colorBlendAttachments.size(); i++)
    {
        colorBlendAttachments[i].colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
        colorBlendAttachments[i].blendEnable = VK_FALSE;
    }

    VkPipelineColorBlendStateCreateInfo colorBlending{};
    colorBlending.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
    colorBlending.logicOpEnable = VK_FALSE;
    colorBlending.attachmentCount = static_cast<uint32_t>(colorBlendAttachments.size());
    colorBlending.pAttachments = colorBlendAttachments.data();

    std::vector<VkDynamicState> dynamicStates = {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
    VkPipelineDynamicStateCreateInfo dynamicState{};
    dynamicState.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
    dynamicState.dynamicStateCount = static_cast<uint32_t>(dynamicStates.size());
    dynamicState.pDynamicStates = dynamicStates.data();

    VkPipelineLayoutCreateInfo pipelineLayoutInfo{};
    pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pipelineLayoutInfo.setLayoutCount = 1;
    pipelineLayoutInfo.pSetLayouts = &descriptorSetLayout;

    VkPushConstantRange pushRange{};
    if (pushConstantSize > 0)
    {
        pushRange.stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;
        pushRange.offset = 0;
        pushRange.size = pushConstantSize;
        pipelineLayoutInfo.pushConstantRangeCount = 1;
        pipelineLayoutInfo.pPushConstantRanges = &pushRange;
    }

    if (vkCreatePipelineLayout(device.getDevice(), &pipelineLayoutInfo, nullptr, &pipelineLayout) != VK_SUCCESS)
        throw std::runtime_error("failed to create pipeline layout!");

    VkGraphicsPipelineCreateInfo pipelineInfo{};
    pipelineInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
    pipelineInfo.stageCount = 2;
    pipelineInfo.pStages = shaderStages;
    pipelineInfo.pVertexInputState = &vertexInputInfo;
    pipelineInfo.pInputAssemblyState = &inputAssembly;
    pipelineInfo.pViewportState = &viewportState;
    pipelineInfo.pRasterizationState = &rasterizer;
    pipelineInfo.pMultisampleState = &multisampling;
    pipelineInfo.pDepthStencilState = &depthStencil;
    pipelineInfo.pColorBlendState = &colorBlending;
    pipelineInfo.pDynamicState = &dynamicState;
    pipelineInfo.layout = pipelineLayout;
    pipelineInfo.renderPass = VK_NULL_HANDLE;
    pipelineInfo.basePipelineHandle = VK_NULL_HANDLE;

    // Dynamic rendering
    VkPipelineRenderingCreateInfo pipelineRenderingInfo{};
    pipelineRenderingInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO;
    pipelineRenderingInfo.viewMask = 0;
    pipelineRenderingInfo.colorAttachmentCount = static_cast<uint32_t>(targetFormat.size());
    pipelineRenderingInfo.pColorAttachmentFormats = targetFormat.data();
    pipelineRenderingInfo.depthAttachmentFormat = device.findDepthFormat();
    pipelineRenderingInfo.stencilAttachmentFormat = VK_FORMAT_UNDEFINED;
    pipelineInfo.pNext = &pipelineRenderingInfo;

    if (vkCreateGraphicsPipelines(device.getDevice(), pipelineCache, 1, &pipelineInfo, nullptr, &pipeline) != VK_SUCCESS)
        throw std::runtime_error("failed to create graphics pipeline!");

    vkDestroyShaderModule(device.getDevice(), shaderModule, nullptr);
}

void GraphicsPipeline::bindPipeline(VkCommandBuffer commandBuffer)
{
    vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);
}

void GraphicsPipeline::bindDescriptorSets(VkCommandBuffer commandBuffer, int currentFrame)
{
    if (descriptorSets.empty())
        return;
    uint32_t idx = std::min(static_cast<uint32_t>(currentFrame), descriptorSetCount - 1);
    vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipelineLayout, 0, 1, &descriptorSets[idx], 0, nullptr);
}

void GraphicsPipeline::pushConstants(VkCommandBuffer commandBuffer, void *data)
{
    vkCmdPushConstants(commandBuffer, pipelineLayout, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0, pushConstantSize, data);
}

// ---- ComputePipeline ----

void ComputePipeline::createPipeline()
{
    auto computeShaderCode = readFile(computeSpvPath);
    VkShaderModule computeShaderModule = createShaderModule(computeShaderCode);

    VkPipelineShaderStageCreateInfo computeShaderStageInfo{};
    computeShaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    computeShaderStageInfo.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    computeShaderStageInfo.module = computeShaderModule;
    computeShaderStageInfo.pName = "main";

    VkPipelineLayoutCreateInfo pipelineLayoutInfo{};
    pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pipelineLayoutInfo.setLayoutCount = 1;
    pipelineLayoutInfo.pSetLayouts = &descriptorSetLayout;

    VkPushConstantRange pushRange{};
    if (pushConstantSize > 0)
    {
        pushRange.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        pushRange.offset = 0;
        pushRange.size = pushConstantSize;
        pipelineLayoutInfo.pushConstantRangeCount = 1;
        pipelineLayoutInfo.pPushConstantRanges = &pushRange;
    }

    if (vkCreatePipelineLayout(device.getDevice(), &pipelineLayoutInfo, nullptr, &pipelineLayout) != VK_SUCCESS)
        throw std::runtime_error("failed to create compute pipeline layout!");

    VkComputePipelineCreateInfo pipelineCreateInfo{};
    pipelineCreateInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    pipelineCreateInfo.layout = pipelineLayout;
    pipelineCreateInfo.stage = computeShaderStageInfo;

    if (vkCreateComputePipelines(device.getDevice(), pipelineCache, 1, &pipelineCreateInfo, nullptr, &pipeline) != VK_SUCCESS)
        throw std::runtime_error("failed to create compute pipeline!");

    vkDestroyShaderModule(device.getDevice(), computeShaderModule, nullptr);
}

void ComputePipeline::bindPipeline(VkCommandBuffer commandBuffer)
{
    vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
}

void ComputePipeline::bindDescriptorSets(VkCommandBuffer commandBuffer, int currentFrame)
{
    uint32_t idx = std::min(static_cast<uint32_t>(currentFrame), descriptorSetCount - 1);
    vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipelineLayout, 0, 1, &descriptorSets[idx], 0, nullptr);
}

void ComputePipeline::pushConstants(VkCommandBuffer commandBuffer, void *data)
{
    vkCmdPushConstants(commandBuffer, pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, pushConstantSize, data);
}

// ---- RayTracingPipeline ----

void RayTracingPipeline::createPipeline()
{
    auto shaderCode = readFile(spvPath);
    VkShaderModule shaderModule = createShaderModule(shaderCode);

    VkPipelineShaderStageCreateInfo raygenStage{};
    raygenStage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    raygenStage.stage = VK_SHADER_STAGE_RAYGEN_BIT_KHR;
    raygenStage.module = shaderModule;
    raygenStage.pName = raygenEntry.c_str();

    VkPipelineShaderStageCreateInfo missStage{};
    missStage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    missStage.stage = VK_SHADER_STAGE_MISS_BIT_KHR;
    missStage.module = shaderModule;
    missStage.pName = missEntryName.c_str();

    VkPipelineShaderStageCreateInfo hitStage{};
    hitStage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    hitStage.stage = VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR;
    hitStage.module = shaderModule;
    hitStage.pName = hitEntryName.c_str();

    VkRayTracingShaderGroupCreateInfoKHR raygenGroup{};
    raygenGroup.sType = VK_STRUCTURE_TYPE_RAY_TRACING_SHADER_GROUP_CREATE_INFO_KHR;
    raygenGroup.type = VK_RAY_TRACING_SHADER_GROUP_TYPE_GENERAL_KHR;
    raygenGroup.generalShader = 0;
    raygenGroup.closestHitShader = VK_SHADER_UNUSED_KHR;
    raygenGroup.anyHitShader = VK_SHADER_UNUSED_KHR;
    raygenGroup.intersectionShader = VK_SHADER_UNUSED_KHR;

    VkRayTracingShaderGroupCreateInfoKHR missGroup{};
    missGroup.sType = VK_STRUCTURE_TYPE_RAY_TRACING_SHADER_GROUP_CREATE_INFO_KHR;
    missGroup.type = VK_RAY_TRACING_SHADER_GROUP_TYPE_GENERAL_KHR;
    missGroup.generalShader = 1;
    missGroup.closestHitShader = VK_SHADER_UNUSED_KHR;
    missGroup.anyHitShader = VK_SHADER_UNUSED_KHR;
    missGroup.intersectionShader = VK_SHADER_UNUSED_KHR;

    VkRayTracingShaderGroupCreateInfoKHR hitGroup{};
    hitGroup.sType = VK_STRUCTURE_TYPE_RAY_TRACING_SHADER_GROUP_CREATE_INFO_KHR;
    hitGroup.type = VK_RAY_TRACING_SHADER_GROUP_TYPE_TRIANGLES_HIT_GROUP_KHR;
    hitGroup.generalShader = VK_SHADER_UNUSED_KHR;
    hitGroup.closestHitShader = 2;
    hitGroup.anyHitShader = VK_SHADER_UNUSED_KHR;
    hitGroup.intersectionShader = VK_SHADER_UNUSED_KHR;

    VkPipelineLayoutCreateInfo pipelineLayoutInfo{};
    pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pipelineLayoutInfo.setLayoutCount = 1;
    pipelineLayoutInfo.pSetLayouts = &descriptorSetLayout;

    VkPushConstantRange pushRange{};
    if (pushConstantSize > 0)
    {
        pushRange.stageFlags = VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR | VK_SHADER_STAGE_MISS_BIT_KHR;
        pushRange.offset = 0;
        pushRange.size = pushConstantSize;
        pipelineLayoutInfo.pushConstantRangeCount = 1;
        pipelineLayoutInfo.pPushConstantRanges = &pushRange;
    }

    if (vkCreatePipelineLayout(device.getDevice(), &pipelineLayoutInfo, nullptr, &pipelineLayout) != VK_SUCCESS)
        throw std::runtime_error("failed to create pipeline layout!");

    std::vector<VkPipelineShaderStageCreateInfo> shaderStages = {raygenStage, missStage, hitStage};
    std::vector<VkRayTracingShaderGroupCreateInfoKHR> shaderGroups = {raygenGroup, missGroup, hitGroup};

    VkRayTracingPipelineCreateInfoKHR pipelineCreateInfo{};
    pipelineCreateInfo.sType = VK_STRUCTURE_TYPE_RAY_TRACING_PIPELINE_CREATE_INFO_KHR;
    pipelineCreateInfo.stageCount = static_cast<uint32_t>(shaderStages.size());
    pipelineCreateInfo.pStages = shaderStages.data();
    pipelineCreateInfo.groupCount = static_cast<uint32_t>(shaderGroups.size());
    pipelineCreateInfo.pGroups = shaderGroups.data();
    pipelineCreateInfo.maxPipelineRayRecursionDepth = 1;
    pipelineCreateInfo.layout = pipelineLayout;

    auto vkCreateRayTracingPipelinesKHR = device.loadDeviceFunction<PFN_vkCreateRayTracingPipelinesKHR>("vkCreateRayTracingPipelinesKHR");
    if (vkCreateRayTracingPipelinesKHR(device.getDevice(), VK_NULL_HANDLE, pipelineCache, 1, &pipelineCreateInfo, nullptr, &pipeline) != VK_SUCCESS)
        throw std::runtime_error("failed to create ray tracing pipeline!");

    vkDestroyShaderModule(device.getDevice(), shaderModule, nullptr);
}

void RayTracingPipeline::bindPipeline(VkCommandBuffer commandBuffer)
{
    vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR, pipeline);
}

void RayTracingPipeline::bindDescriptorSets(VkCommandBuffer commandBuffer, int currentFrame)
{
    uint32_t idx = std::min(static_cast<uint32_t>(currentFrame), descriptorSetCount - 1);
    vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR, pipelineLayout, 0, 1, &descriptorSets[idx], 0, nullptr);
}

void RayTracingPipeline::createSBTs(const std::vector<HitSBTRecord> &hitRecords)
{
    vkCmdTraceRaysKHR = device.loadDeviceFunction<PFN_vkCmdTraceRaysKHR>("vkCmdTraceRaysKHR");
    vkGetRayTracingShaderGroupHandlesKHR = device.loadDeviceFunction<PFN_vkGetRayTracingShaderGroupHandlesKHR>("vkGetRayTracingShaderGroupHandlesKHR");

    auto props = device.getPhysicalDeviceRTPipelineProperties();
    const uint32_t handleSize = props.shaderGroupHandleSize;
    const uint32_t handleAlignment = props.shaderGroupHandleAlignment;
    const uint32_t handleSizeAligned = ((handleSize + handleAlignment - 1) / handleAlignment) * handleAlignment;
    const uint32_t baseAlignment = props.shaderGroupBaseAlignment;

    const uint32_t groupCount = 3;
    const uint32_t hitRecordSize = static_cast<uint32_t>(sizeof(HitSBTRecord));
    const uint32_t hitGroupCount = static_cast<uint32_t>(hitRecords.size());
    const uint32_t hitStride = ((handleSizeAligned + hitRecordSize + baseAlignment - 1) / baseAlignment) * baseAlignment;

    const VkBufferUsageFlags sbtFlags = VK_BUFFER_USAGE_SHADER_BINDING_TABLE_BIT_KHR | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;

    raygenSBT = std::make_unique<StorageBufferResource>(device, handleSizeAligned, sbtFlags);
    missSBT = std::make_unique<StorageBufferResource>(device, handleSizeAligned, sbtFlags);
    hitSBT = std::make_unique<StorageBufferResource>(device, hitStride * hitGroupCount, sbtFlags);

    const uint32_t handleDataSize = groupCount * handleSizeAligned;
    std::vector<uint8_t> handleData(handleDataSize);
    vkGetRayTracingShaderGroupHandlesKHR(device.getDevice(), pipeline, 0, groupCount, handleDataSize, handleData.data());

    raygenSBT->update(handleData.data());
    missSBT->update(handleData.data() + handleSizeAligned);

    const uint8_t *srcData = reinterpret_cast<const uint8_t *>(hitRecords.data());
    std::vector<uint8_t> hitData(hitStride * hitGroupCount);
    for (uint32_t i = 0; i < hitGroupCount; i++)
    {
        memcpy(hitData.data() + hitStride * i, handleData.data() + handleSizeAligned * 2, handleSizeAligned);
        memcpy(hitData.data() + hitStride * i + handleSizeAligned, srcData + hitRecordSize * i, hitRecordSize);
    }
    hitSBT->update(hitData.data());

    raygenRegion = {};
    raygenRegion.size = handleSizeAligned;
    raygenRegion.stride = handleSizeAligned;
    raygenRegion.deviceAddress = device.getBufferDeviceAddress(raygenSBT->getBuffer());

    missRegion = {};
    missRegion.size = handleSizeAligned;
    missRegion.stride = handleSizeAligned;
    missRegion.deviceAddress = device.getBufferDeviceAddress(missSBT->getBuffer());

    hitRegion = {};
    hitRegion.size = hitStride * hitGroupCount;
    hitRegion.stride = hitStride;
    hitRegion.deviceAddress = device.getBufferDeviceAddress(hitSBT->getBuffer());

    callRegion = {};
}

void RayTracingPipeline::pushConstants(VkCommandBuffer commandBuffer, void *data)
{
    vkCmdPushConstants(commandBuffer, pipelineLayout,
                       VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR | VK_SHADER_STAGE_MISS_BIT_KHR,
                       0, pushConstantSize, data);
}

void RayTracingPipeline::traceRays(VkCommandBuffer commandBuffer, VkExtent3D extent)
{
    vkCmdTraceRaysKHR(commandBuffer, &raygenRegion, &missRegion, &hitRegion, &callRegion, extent.width, extent.height, extent.depth);
}
