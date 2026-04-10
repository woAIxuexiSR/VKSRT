// Path tracing shader: iterative multi-bounce with BRDF sampling and Russian Roulette
// Single file with raygen, miss, closesthit entry points

import material;
import brdf;
import onb;
import rng;

// --- Payload: closesthit returns geometry info, raygen does shading ---

[raypayload]
struct RayPayload
{
    int isHit : read(caller) : write(caller, closesthit, miss);
    float3 hitPos : read(caller) : write(closesthit);
    float3 hitNormal : read(caller) : write(closesthit);
    int matIndex : read(caller) : write(closesthit);
};

struct CameraProperties
{
    float4x4 viewInverse;
    float4x4 projInverse;
};

struct PushConstants
{
    int frameIndex;
    int maxDepth;
    int rrDepth;
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

// --- Ray Generation: iterative path tracing ---

[shader("raygeneration")]
void raygenMain()
{
    uint2 launchID = DispatchRaysIndex().xy;
    uint2 launchSize = DispatchRaysDimensions().xy;

    uint seed = wangHash(uint(launchID.x * 1973u + launchID.y * 9277u + uint(pc.frameIndex) * 26699u) | 1u);

    // Pixel jitter for anti-aliasing
    float2 jitter = randFloat2(seed);
    float2 pixelCenter = float2(launchID) + jitter;
    float2 inUV = pixelCenter / float2(launchSize);
    float2 d = inUV * 2.0 - 1.0;

    float4 origin = mul(cam.viewInverse, float4(0, 0, 0, 1));
    float4 target = mul(cam.projInverse, float4(d.x, d.y, 1, 1));
    float4 direction = mul(cam.viewInverse, float4(normalize(target.xyz), 0));

    float3 rayPos = origin.xyz;
    float3 rayDir = normalize(direction.xyz);

    float3 L = float3(0.0);
    float3 beta = float3(1.0);

    for (int depth = 0; depth < pc.maxDepth; depth++)
    {
        RayDesc ray;
        ray.Origin = rayPos;
        ray.Direction = rayDir;
        ray.TMin = 0.001;
        ray.TMax = 10000.0;

        RayPayload payload;
        payload.isHit = 0;
        TraceRay(topLevelAS, RAY_FLAG_FORCE_OPAQUE, 0xff, 0, 0, 0, ray, payload);

        // Miss: add background and terminate
        if (payload.isHit == 0)
        {
            L += beta * float3(0.0);
            break;
        }

        Material mat = materials[payload.matIndex];

        // Hit emissive surface: add emission and terminate
        if (mat.type == MAT_EMISSIVE)
        {
            L += beta * mat.emission.rgb * mat.emission.a;
            break;
        }

        // Determine shading normal per material type
        float3 shadingN = payload.hitNormal;
        if (mat.type != MAT_DIELECTRIC)
        {
            // Non-dielectric: flip normal to face incoming ray
            if (dot(shadingN, rayDir) > 0.0)
                shadingN = -shadingN;
        }

        // Build local coordinate frame at hit point
        ONB basis = buildONB(shadingN);
        float3 wo = worldToLocal(basis, -rayDir);

        // BRDF importance sampling
        float2 rng = randFloat2(seed);
        BRDFSample s = brdfSample(wo, mat, rng);

        if (!s.valid || s.pdf < 1e-6)
            break;

        // Update throughput
        if (s.isDelta)
            beta *= s.value;
        else
            beta *= s.value * max(s.wi.z, 0.0) / s.pdf;

        // Set up next ray
        rayPos = payload.hitPos;
        rayDir = normalize(localToWorld(basis, s.wi));

        // Russian Roulette
        if (depth >= pc.rrDepth)
        {
            float p = max(max(beta.r, max(beta.g, beta.b)), 0.05);
            if (randFloat(seed) > p)
                break;
            beta /= p;
        }
    }

    image[launchID] = float4(L, 1.0);
}

// --- Miss ---

[shader("miss")]
void missMain(inout RayPayload payload)
{
    payload.isHit = 0;
}

// --- Closest Hit: return geometry info only ---

[shader("closesthit")]
void closestHitMain(inout RayPayload payload, BuiltInTriangleIntersectionAttributes attribs)
{
    float3 bary = float3(1.0 - attribs.barycentrics.x - attribs.barycentrics.y,
                         attribs.barycentrics.x, attribs.barycentrics.y);

    uint triIdx = PrimitiveIndex();
    uint i0 = indices[(sbt.indexOffset + triIdx) * 3 + 0];
    uint i1 = indices[(sbt.indexOffset + triIdx) * 3 + 1];
    uint i2 = indices[(sbt.indexOffset + triIdx) * 3 + 2];

    // Interpolate position
    float3 v0 = getVertex(i0 + sbt.vertexOffset);
    float3 v1 = getVertex(i1 + sbt.vertexOffset);
    float3 v2 = getVertex(i2 + sbt.vertexOffset);
    float3 hitPos = bary.x * v0 + bary.y * v1 + bary.z * v2;

    // Interpolate normal
    float3 n0 = getNormal(i0 + sbt.vertexOffset);
    float3 n1 = getNormal(i1 + sbt.vertexOffset);
    float3 n2 = getNormal(i2 + sbt.vertexOffset);
    float3 N = normalize(bary.x * n0 + bary.y * n1 + bary.z * n2);

    payload.isHit = 1;
    payload.hitPos = hitPos;
    payload.hitNormal = N;
    payload.matIndex = sbt.matIndex;
}
