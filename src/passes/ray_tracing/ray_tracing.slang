// Ray tracing shaders: visualization/debug tool
// Modes: 0=Material, 1=Position, 2=Normal, 3=UV

import material;
import brdf;
import onb;
import math_utils;

[raypayload]
struct RayPayload
{
    int isHit : read(caller) : write(caller, closesthit, miss);
    float3 hitPos : read(caller) : write(caller, closesthit, miss);
    float3 hitNormal : read(caller) : write(caller, closesthit, miss);
    float2 hitUV : read(caller) : write(caller, closesthit, miss);
    int matIndex : read(caller) : write(caller, closesthit, miss);
};

struct CameraProperties
{
    float4x4 viewInverse;
    float4x4 projInverse;
};

struct PushConstants
{
    int shadingMode;
};

[[vk::push_constant]]
PushConstants pc;

[[vk::binding(0)]]
RWTexture2D<float4> image;

[[vk::binding(1)]]
ConstantBuffer<CameraProperties> cam;

[[vk::binding(2)]]
RaytracingAccelerationStructure topLevelAS;

[[vk::binding(3)]]
StructuredBuffer<float> vertexData;

[[vk::binding(4)]]
StructuredBuffer<uint> indices;

[[vk::binding(5)]]
StructuredBuffer<Material> materials;

[[vk::binding(6)]]
StructuredBuffer<float> normalData;

[[vk::binding(7)]]
StructuredBuffer<float2> texcoords;

float3 getVertex(uint index)
{
    return float3(vertexData[index * 3], vertexData[index * 3 + 1], vertexData[index * 3 + 2]);
}

float3 getNormal(uint index)
{
    return float3(normalData[index * 3], normalData[index * 3 + 1], normalData[index * 3 + 2]);
}

struct HitGroupSBT
{
    int matIndex;
    int vertexOffset;
    int indexOffset;
};

[[vk::shader_record]]
ConstantBuffer<HitGroupSBT> sbt;

// --- Ray Generation ---

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
    ray.Direction = normalize(direction.xyz);
    ray.TMin = 0.001;
    ray.TMax = 10000.0;

    RayPayload payload;
    payload.isHit = 0;
    payload.hitPos = float3(0.0);
    payload.hitNormal = float3(0.0);
    payload.hitUV = float2(0.0);
    payload.matIndex = -1;

    TraceRay(topLevelAS, RAY_FLAG_FORCE_OPAQUE, 0xff, 0, 0, 0, ray, payload);

    float3 color = float3(0.2, 0.2, 0.2); // miss background

    if (payload.isHit != 0)
    {
        if (pc.shadingMode == 0)
        {
            // Material shading: simple directional light
            Material mat = materials[payload.matIndex];
            float3 N = payload.hitNormal;

            if (mat.isEmissive())
            {
                color = mat.emission.rgb * mat.emission.a;
            }
            else
            {
                float3 lightDir = normalize(float3(0.5, 1.0, -0.3));
                float NdotL = max(dot(N, lightDir), 0.0);
                color = mat.baseColor.rgb * (0.15 + 0.85 * NdotL);
            }
        }
        else if (pc.shadingMode == 1)
        {
            // Position visualization
            color = fract(payload.hitPos + 0.001);
        }
        else if (pc.shadingMode == 2)
        {
            // Normal visualization
            color = payload.hitNormal * 0.5 + 0.5;
        }
        else if (pc.shadingMode == 3)
        {
            // UV visualization
            color = float3(payload.hitUV, 0.0);
        }
    }

    image[launchID] = float4(color, 1.0);
}

// --- Miss ---

[shader("miss")]
void missMain(inout RayPayload payload)
{
    payload.isHit = 0;
}

// --- Closest Hit ---

[shader("closesthit")]
void closestHitMain(inout RayPayload payload, BuiltInTriangleIntersectionAttributes attribs)
{
    float3 bary = float3(1.0 - attribs.barycentrics.x - attribs.barycentrics.y,
                         attribs.barycentrics.x, attribs.barycentrics.y);

    uint triIdx = PrimitiveIndex();
    uint i0 = indices[(sbt.indexOffset + triIdx) * 3 + 0];
    uint i1 = indices[(sbt.indexOffset + triIdx) * 3 + 1];
    uint i2 = indices[(sbt.indexOffset + triIdx) * 3 + 2];

    // Interpolate object-space position and transform to world space
    float3 v0 = getVertex(i0 + sbt.vertexOffset);
    float3 v1 = getVertex(i1 + sbt.vertexOffset);
    float3 v2 = getVertex(i2 + sbt.vertexOffset);
    float3 objPos = bary.x * v0 + bary.y * v1 + bary.z * v2;
    float3 hitPos = transformPosition(ObjectToWorld3x4(), objPos);

    // Interpolate object-space normal and transform to world space
    float3 n0 = getNormal(i0 + sbt.vertexOffset);
    float3 n1 = getNormal(i1 + sbt.vertexOffset);
    float3 n2 = getNormal(i2 + sbt.vertexOffset);
    float3 objN = normalize(bary.x * n0 + bary.y * n1 + bary.z * n2);
    float3 N = transformNormal(WorldToObject3x4(), objN);

    // Ensure normal faces against the ray
    if (dot(N, WorldRayDirection()) > 0.0)
        N = -N;

    // Interpolate UV
    float2 uv0 = texcoords[i0 + sbt.vertexOffset];
    float2 uv1 = texcoords[i1 + sbt.vertexOffset];
    float2 uv2 = texcoords[i2 + sbt.vertexOffset];
    float2 uv = bary.x * uv0 + bary.y * uv1 + bary.z * uv2;

    payload.isHit = 1;
    payload.hitPos = hitPos;
    payload.hitNormal = N;
    payload.hitUV = uv;
    payload.matIndex = sbt.matIndex;
}
