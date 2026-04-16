// Wavefront PT: Extend (trace rays via ray query)
// 1D indirect dispatch from counterBuffer[0..2]

import wavefront_structs;
import material;
import math_utils;

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
    int totalPaths;
    int bounceIndex;
    int queueOffset;
    int _pad;
};

[[vk::push_constant]]
PushConstants pc;

[[vk::binding(0)]]
RWStructuredBuffer<PathState> pathStates;

[[vk::binding(1)]]
RWStructuredBuffer<RayData> rays;

[[vk::binding(2)]]
RWStructuredBuffer<HitInfo> hitInfos;

[[vk::binding(5)]]
RWStructuredBuffer<uint> queue;

[[vk::binding(8)]]
RaytracingAccelerationStructure topLevelAS;

[[vk::binding(9)]]
StructuredBuffer<float> vertexData;

[[vk::binding(10)]]
StructuredBuffer<uint> indices;

[[vk::binding(12)]]
StructuredBuffer<float> normalData;

[[vk::binding(15)]]
StructuredBuffer<float3x4> instanceTransforms;

[[vk::binding(16)]]
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

    // Read path index from current queue
    uint pathIdx = queue[pc.queueOffset + threadIdx];
    if (pathIdx >= uint(pc.totalPaths)) return;

    PathState path = pathStates[pathIdx];
    if (path.alive == 0) return;

    RayData ray = rays[pathIdx];

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
        float3 objPos = bary.x * v0 + bary.y * v1 + bary.z * v2;

        float3 n0 = getNormal(i0 + mi.vertexOffset);
        float3 n1 = getNormal(i1 + mi.vertexOffset);
        float3 n2 = getNormal(i2 + mi.vertexOffset);
        float3 objN = normalize(bary.x * n0 + bary.y * n1 + bary.z * n2);

        hit.hitPos = transformPosition(rq.CommittedObjectToWorld3x4(), objPos);
        hit.matIndex = mi.matIndex;
        hit.hitNormal = transformNormal(rq.CommittedWorldToObject3x4(), objN);
        hit.isHit = 1;
    }
    else
    {
        hit.hitPos = float3(0.0);
        hit.matIndex = 0;
        hit.hitNormal = float3(0.0);
        hit.isHit = 0;
    }

    hitInfos[pathIdx] = hit;
}
