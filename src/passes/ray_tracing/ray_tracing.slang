// Ray tracing shader for Cornell Box (base color only)
// Single file with raygen, miss, and closesthit entry points

// Descriptor bindings
[[vk::binding(0)]]
RWTexture2D<float4> outputImage;

struct CameraUBO
{
    int frameIndex;
    int _pad0;
    int _pad1;
    int _pad2;
    float4x4 viewInverse;
    float4x4 projInverse;
};

[[vk::binding(1)]]
ConstantBuffer<CameraUBO> cam;

[[vk::binding(2)]]
RaytracingAccelerationStructure topLevelAS;

[[vk::binding(3)]]
StructuredBuffer<float> vertexData;

[[vk::binding(4)]]
StructuredBuffer<uint> indexData;

[[vk::binding(5)]]
StructuredBuffer<float4> materials;

// Per-instance hit info stored in a buffer, indexed by InstanceID
struct HitRecord
{
    int matIndex;
    int vertexOffset;
    int indexOffset;
    int _pad;
};

[[vk::binding(6)]]
StructuredBuffer<HitRecord> hitRecords;

// Ray payload
struct [raypayload] RTPayload
{
    int isHit : read(caller, closesthit, miss) : write(closesthit, miss);
    float3 color : read(caller, closesthit, miss) : write(closesthit, miss);
};

float3 getVertex(uint index)
{
    return float3(vertexData[index * 3], vertexData[index * 3 + 1], vertexData[index * 3 + 2]);
}

uint3 getIndex(uint index)
{
    return uint3(indexData[index * 3], indexData[index * 3 + 1], indexData[index * 3 + 2]);
}

[shader("raygeneration")]
void raygenMain()
{
    uint2 launchID = DispatchRaysIndex().xy;
    uint2 launchSize = DispatchRaysDimensions().xy;

    float2 pixelCenter = float2(launchID) + float2(0.5, 0.5);
    float2 inUV = pixelCenter / float2(launchSize);
    float2 d = inUV * 2.0 - 1.0;

    float4 origin = mul(cam.viewInverse, float4(0, 0, 0, 1));
    float4 target = mul(cam.projInverse, float4(d.x, d.y, 1, 1));
    float4 direction = mul(cam.viewInverse, float4(normalize(target.xyz), 0));

    RayDesc ray;
    ray.Origin = origin.xyz;
    ray.Direction = direction.xyz;
    ray.TMin = 0.001;
    ray.TMax = 10000.0;

    RTPayload payload;
    payload.isHit = 0;
    payload.color = float3(0, 0, 0);

    TraceRay(topLevelAS, RAY_FLAG_FORCE_OPAQUE, 0xff, 0, 0, 0, ray, payload);

    float3 result = payload.color;

    // Progressive accumulation
    float4 prevColor = float4(0, 0, 0, 0);
    if (cam.frameIndex > 0)
        prevColor = outputImage[launchID];
    float3 finalColor = lerp(prevColor.rgb, result, 1.0 / float(cam.frameIndex + 1));
    outputImage[launchID] = float4(finalColor, 1.0);
}

[shader("miss")]
void missMain(inout RTPayload payload)
{
    payload.isHit = 0;
    payload.color = float3(0.1, 0.1, 0.1);
}

[shader("closesthit")]
void closesthitMain(inout RTPayload payload, in BuiltInTriangleIntersectionAttributes attribs)
{
    // Use InstanceID to look up per-mesh data from buffer
    uint instanceID = InstanceID();
    HitRecord rec = hitRecords[instanceID];

    uint primitiveID = PrimitiveIndex() + rec.indexOffset;
    uint3 idx = getIndex(primitiveID);

    float4 matColor = materials[rec.matIndex];

    payload.isHit = 1;
    payload.color = matColor.rgb;
}
