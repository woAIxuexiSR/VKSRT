// Branch MC: Extend (trace rays for new child vertices via ray query)
// 1D indirect dispatch from counterBuffer[0..2]

import bmc_structs;
import wavefront_structs;
import material;

struct PushConstants
{
    int frameIndex;
    int maxDepth;
    int rrDepth;
    int lightCount;
    int useNEE;
    int useMIS;
    int screenWidth;
    int screenHeight;
    int currentDepth;
    int depthVertexStart;
    int depthVertexCount;
    int useDebiasing;
    float debiasR;
    int innerSamples0;
    int innerSamples1;
    int innerSamples2;
    int innerSamples3;
    int maxVertices;
    int _pad0;
    int _pad1;
};

[[vk::push_constant]]
PushConstants pc;

[[vk::binding(0)]]
RWStructuredBuffer<BMCVertex> vertices;

[[vk::binding(1)]]
RWStructuredBuffer<RayData> rays;

[[vk::binding(2)]]
RWStructuredBuffer<HitInfo> hitInfos;

[[vk::binding(8)]]
RaytracingAccelerationStructure topLevelAS;

[[vk::binding(9)]]
StructuredBuffer<float> vertexData;

[[vk::binding(10)]]
StructuredBuffer<uint> indices;

[[vk::binding(12)]]
StructuredBuffer<float> normalData;

[[vk::binding(15)]]
StructuredBuffer<MeshInfo> meshInfos;

float3 getVertex(uint index)
{
    return float3(vertexData[index * 3], vertexData[index * 3 + 1], vertexData[index * 3 + 2]);
}

float3 getNormal(uint index)
{
    return float3(normalData[index * 3], normalData[index * 3 + 1], normalData[index * 3 + 2]);
}

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    uint threadIdx = dispatchThreadID.x;

    // Child vertices are placed right after current depth's vertices
    int childIdx = pc.depthVertexStart + pc.depthVertexCount + int(threadIdx);
    if (childIdx >= pc.maxVertices) return;

    BMCVertex vtx = vertices[childIdx];
    if (vtx.alive == 0) return;

    RayData ray = rays[childIdx];

    RayDesc rayDesc;
    rayDesc.Origin = ray.origin;
    rayDesc.TMin = ray.tMin;
    rayDesc.Direction = ray.direction;
    rayDesc.TMax = ray.tMax;

    RayQuery<RAY_FLAG_FORCE_OPAQUE> rq;
    rq.TraceRayInline(topLevelAS, RAY_FLAG_FORCE_OPAQUE, 0xff, rayDesc);
    rq.Proceed();

    HitInfo hit;
    if (rq.CommittedStatus() == COMMITTED_TRIANGLE_HIT)
    {
        uint instanceIdx = rq.CommittedInstanceID();
        uint primIdx = rq.CommittedPrimitiveIndex();
        float2 bary2 = rq.CommittedTriangleBarycentrics();
        float3 bary = float3(1.0 - bary2.x - bary2.y, bary2.x, bary2.y);

        MeshInfo mi = meshInfos[instanceIdx];

        uint i0 = indices[(mi.indexOffset + primIdx) * 3 + 0];
        uint i1 = indices[(mi.indexOffset + primIdx) * 3 + 1];
        uint i2 = indices[(mi.indexOffset + primIdx) * 3 + 2];

        float3 v0 = getVertex(i0 + mi.vertexOffset);
        float3 v1 = getVertex(i1 + mi.vertexOffset);
        float3 v2 = getVertex(i2 + mi.vertexOffset);
        float3 hitPos = bary.x * v0 + bary.y * v1 + bary.z * v2;

        float3 n0 = getNormal(i0 + mi.vertexOffset);
        float3 n1 = getNormal(i1 + mi.vertexOffset);
        float3 n2 = getNormal(i2 + mi.vertexOffset);
        float3 N = normalize(bary.x * n0 + bary.y * n1 + bary.z * n2);

        hit.hitPos = hitPos;
        hit.matIndex = mi.matIndex;
        hit.hitNormal = N;
        hit.isHit = 1;
    }
    else
    {
        hit.hitPos = float3(0.0);
        hit.matIndex = 0;
        hit.hitNormal = float3(0.0);
        hit.isHit = 0;
    }

    hitInfos[childIdx] = hit;
}
