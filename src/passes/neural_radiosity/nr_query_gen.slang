// Neural Radiosity online query generator: for each pixel, cast a primary ray with
// jitter, inline ray-query the scene, and write 12-dim input features for the
// network forward pass into queryInputBuffer. Also fills primaryInfoBuffer:
//   - non-emissive hit:  (0, 0, 0, 1.0)   — composite uses network prediction
//   - emissive hit:      (emission.rgb, 0) — composite bypasses network
//   - miss:              (0, 0, 0, 0)

import camera;
import rng;
import material;
import math_utils;
import brdf;
import onb;
import wavefront_structs;

struct QueryGenPC
{
    uint64_t queryInputAddr;    // P * 12 floats
    uint64_t primaryInfoAddr;   // P * (2 * 4 floats) = P * 8 floats; see pass cpp for layout
    uint width;
    uint height;
    uint frameIndex;
    uint _pad0;
};

[[vk::push_constant]]
QueryGenPC pc;

[[vk::binding(0)]]
ConstantBuffer<CameraProperties> cam;

[[vk::binding(1)]]
RaytracingAccelerationStructure topLevelAS;

[[vk::binding(2)]]
StructuredBuffer<float> vertexData;

[[vk::binding(3)]]
StructuredBuffer<uint> indices;

[[vk::binding(4)]]
StructuredBuffer<Material> materials;

[[vk::binding(5)]]
StructuredBuffer<float> normalData;

// binding(6) = texcoords (unused here; scene layout keeps it)

[[vk::binding(7)]]
StructuredBuffer<LightTriangle> lights;

[[vk::binding(8)]]
StructuredBuffer<float3x4> instanceTransforms;

[[vk::binding(9)]]
StructuredBuffer<MeshInfo> meshInfos;

float3 getVertex(uint i)
{
    return float3(vertexData[i * 3], vertexData[i * 3 + 1], vertexData[i * 3 + 2]);
}

float3 getNormal(uint i)
{
    return float3(normalData[i * 3], normalData[i * 3 + 1], normalData[i * 3 + 2]);
}

struct InlineHit
{
    float3 hitPos;
    float3 hitNormal;
    int matIndex;
    int isHit;
};

InlineHit traceRay(float3 origin, float3 dir, float tMin, float tMax)
{
    RayDesc ray;
    ray.Origin = origin;
    ray.TMin = tMin;
    ray.Direction = dir;
    ray.TMax = tMax;

    RayQuery<RAY_FLAG_FORCE_OPAQUE> rq;
    rq.TraceRayInline(topLevelAS, RAY_FLAG_FORCE_OPAQUE, 0xff, ray);
    rq.Proceed();

    InlineHit r;
    if (rq.CommittedStatus() == COMMITTED_TRIANGLE_HIT)
    {
        uint instanceIdx = rq.CommittedInstanceID();
        uint primIdx = rq.CommittedPrimitiveIndex();
        float2 b2 = rq.CommittedTriangleBarycentrics();
        float3 bary = float3(1.0 - b2.x - b2.y, b2.x, b2.y);

        MeshInfo mi = meshInfos[instanceIdx];
        uint i0 = indices[(mi.indexOffset + primIdx) * 3 + 0];
        uint i1 = indices[(mi.indexOffset + primIdx) * 3 + 1];
        uint i2 = indices[(mi.indexOffset + primIdx) * 3 + 2];

        float3 v0 = getVertex(i0 + mi.vertexOffset);
        float3 v1 = getVertex(i1 + mi.vertexOffset);
        float3 v2 = getVertex(i2 + mi.vertexOffset);
        float3 n0 = getNormal(i0 + mi.vertexOffset);
        float3 n1 = getNormal(i1 + mi.vertexOffset);
        float3 n2 = getNormal(i2 + mi.vertexOffset);

        float3 objPos = bary.x * v0 + bary.y * v1 + bary.z * v2;
        float3 objN = normalize(bary.x * n0 + bary.y * n1 + bary.z * n2);
        r.hitPos = transformPosition(rq.CommittedObjectToWorld3x4(), objPos);
        r.hitNormal = transformNormal(rq.CommittedWorldToObject3x4(), objN);
        r.matIndex = mi.matIndex;
        r.isHit = 1;
    }
    else
    {
        r.hitPos = float3(0.0);
        r.hitNormal = float3(0.0);
        r.matIndex = 0;
        r.isHit = 0;
    }
    return r;
}

void writeQueryFeat(uint px, float3 pos, float3 normal, float3 dir, float3 albedo)
{
    float* q = (float*)pc.queryInputAddr;
    uint base = px * 12;
    q[base + 0]  = pos.x;    q[base + 1]  = pos.y;    q[base + 2]  = pos.z;
    q[base + 3]  = normal.x; q[base + 4]  = normal.y; q[base + 5]  = normal.z;
    q[base + 6]  = dir.x;    q[base + 7]  = dir.y;    q[base + 8]  = dir.z;
    q[base + 9]  = albedo.x; q[base + 10] = albedo.y; q[base + 11] = albedo.z;
}

void writeQueryZero(uint px)
{
    float* q = (float*)pc.queryInputAddr;
    uint base = px * 12;
    [unroll] for (int k = 0; k < 12; k++) q[base + k] = 0.0;
}

// primaryInfo layout: info[0] = (rgb direct color, a validNet flag), info[1] = throughput
void writePrimary(uint px, float4 info0, float3 throughput)
{
    float4* p = (float4*)pc.primaryInfoAddr;
    p[px * 2 + 0] = info0;
    p[px * 2 + 1] = float4(throughput, 0.0);
}

[numthreads(16, 16, 1)]
[shader("compute")]
void main(uint3 dtid : SV_DispatchThreadID)
{
    if (dtid.x >= pc.width || dtid.y >= pc.height)
        return;

    uint px = dtid.y * pc.width + dtid.x;
    uint seed = seedFromPixel(uint2(dtid.xy), pc.frameIndex);

    float3 origin, dir;
    generatePrimaryRay(cam, uint2(dtid.xy), uint2(pc.width, pc.height), seed, origin, dir);

    // Bounce through delta surfaces until we hit something non-delta (the network query
    // target), an emitter, or nothing. Identical structure to nr_sample_gen's delta loop.
    float3 rayOrigin = origin;
    float3 rayDir = dir;
    float3 throughput = float3(1.0);

    const int kMaxDeltaBounces = 8;

    [loop]
    for (int depth = 0; depth < kMaxDeltaBounces; depth++)
    {
        InlineHit hit = traceRay(rayOrigin, rayDir, 0.001, 10000.0);
        if (hit.isHit == 0)
        {
            writeQueryZero(px);
            writePrimary(px, float4(0.0, 0.0, 0.0, 0.0), float3(0.0));
            return;
        }

        Material mat = materials[hit.matIndex];

        if (mat.isEmissive())
        {
            float3 emit = mat.emission.rgb * mat.emission.a;
            writeQueryZero(px);
            writePrimary(px, float4(throughput * emit, 0.0), float3(0.0));
            return;
        }

        if (!mat.isDeltaBRDF())
        {
            // Non-emissive non-delta: this is our network query point.
            float3 n = hit.hitNormal;
            if (dot(n, rayDir) > 0.0) n = -n;
            float3 omega_o = -rayDir;
            writeQueryFeat(px, hit.hitPos, n, omega_o, mat.baseColor.rgb);
            writePrimary(px, float4(0.0, 0.0, 0.0, 1.0), throughput);
            return;
        }

        // Delta surface: sample and continue.
        float3 nLocal = hit.hitNormal;
        if (dot(nLocal, rayDir) > 0.0) nLocal = -nLocal;
        ONB basis = buildONB(nLocal);
        float3 wo_local = worldToLocal(basis, -rayDir);
        BRDFSample bs = brdfSample(wo_local, mat, randFloat3(seed));
        if (!bs.valid || bs.pdf < PDF_EPSILON)
        {
            writeQueryZero(px);
            writePrimary(px, float4(0.0, 0.0, 0.0, 0.0), float3(0.0));
            return;
        }
        // Delta BSDFs report value already cancelled against |cos|/pdf.
        throughput *= bs.value;
        rayOrigin = hit.hitPos;
        rayDir = normalize(localToWorld(basis, bs.wi));
    }

    // Ran out of delta bounces — treat as black.
    writeQueryZero(px);
    writePrimary(px, float4(0.0, 0.0, 0.0, 0.0), float3(0.0));
}
