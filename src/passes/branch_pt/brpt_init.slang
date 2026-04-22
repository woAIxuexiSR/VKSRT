// Branch PT: Initialize depth-0 vertices from camera rays
// 1D dispatch: [(tilePixelCount+255)/256, 1, 1]
// Each pixel in the tile creates one depth-0 BrPTVertex via atomic allocation

import brpt_structs;
import wavefront_structs;
import material;
import rng;
import math_utils;
import camera;

struct PushConstants
{
    int frameIndex;
    int maxDepth;
    int lightCount;
    int useNEE;
    int useMIS;
    int screenWidth;
    int screenHeight;
    int currentDepth;
    int useDebiasing;
    float debiasR;
    int innerSamples0;
    int innerSamples1;
    int innerSamples2;
    int innerSamples3;
    int maxVertices;
    int tilePixelOffset;
    int tilePixelCount;
    int rrDepth;
};

[[vk::push_constant]]
PushConstants pc;

[[vk::binding(0)]]
RWStructuredBuffer<BrPTVertex> vertices;

[[vk::binding(1)]]
RWStructuredBuffer<uint> counters; // [0]=vtxCounter, [1]=debiasCounter, [2,3,4]=indirect dispatch

[[vk::binding(2)]]
RWStructuredBuffer<int2> depthRanges; // [startIdx, count] per depth

[[vk::binding(4)]]
ConstantBuffer<CameraProperties> cam;

[[vk::binding(5)]]
RaytracingAccelerationStructure topLevelAS;

[[vk::binding(6)]]
StructuredBuffer<float> vertexData;

[[vk::binding(7)]]
StructuredBuffer<uint> indices;

[[vk::binding(9)]]
StructuredBuffer<float> normalData;

[[vk::binding(12)]]
StructuredBuffer<float3x4> instanceTransforms;

[[vk::binding(13)]]
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
    uint tid = dispatchThreadID.x;
    if (tid >= uint(pc.tilePixelCount))
        return;

    // Map thread to full-screen pixel coordinate
    int pixelIdx = pc.tilePixelOffset + int(tid);
    int2 pixel = int2(pixelIdx % pc.screenWidth, pixelIdx / pc.screenWidth);

    // Generate camera ray with jitter
    uint seed = seedFromPixel(uint2(pixel), uint(pc.frameIndex));
    float3 rayOrigin, rayDir;
    generatePrimaryRay(cam, uint2(pixel), uint2(pc.screenWidth, pc.screenHeight), seed, rayOrigin, rayDir);

    // Trace primary ray via inline ray query
    RayDesc rayDesc;
    rayDesc.Origin = rayOrigin;
    rayDesc.TMin = 0.001;
    rayDesc.Direction = rayDir;
    rayDesc.TMax = 10000.0;

    RayQuery<RAY_FLAG_FORCE_OPAQUE> rq;
    rq.TraceRayInline(topLevelAS, RAY_FLAG_FORCE_OPAQUE, 0xff, rayDesc);
    rq.Proceed();

    // Atomically allocate vertex slot
    uint vertexIdx;
    InterlockedAdd(counters[0], 1u, vertexIdx);
    if (vertexIdx >= uint(pc.maxVertices))
        return;

    // Build vertex with hit info embedded
    BrPTVertex vtx;
    vtx.throughput = float3(1.0);
    vtx.seed = seed;
    vtx.radiance = float3(0.0);
    vtx.pixelIndex = pixelIdx;
    vtx.wo = -rayDir;
    vtx.depth = 0;
    vtx.firstChild = -1;
    vtx.nextSibling = -1;
    vtx.parent = -1;
    vtx.nInnerSamples = 0;
    vtx.childIndex = 0;
    vtx.debiasRadianceIdx = -1;
    vtx.prevBsdfPdf = 0.0;
    vtx.styleType = 0;
    vtx.styleParam0 = 0.0;
    vtx.debiasingSamples = 0;
    vtx.actualBranchingDepth = 0;
    vtx.mis = 1.0;

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

        float3 n0 = getNormal(i0 + mi.vertexOffset);
        float3 n1 = getNormal(i1 + mi.vertexOffset);
        float3 n2 = getNormal(i2 + mi.vertexOffset);

        float3 objPos = bary.x * v0 + bary.y * v1 + bary.z * v2;
        float3 objN = normalize(bary.x * n0 + bary.y * n1 + bary.z * n2);
        vtx.hitPos = transformPosition(rq.CommittedObjectToWorld3x4(), objPos);
        vtx.hitNormal = transformNormal(rq.CommittedWorldToObject3x4(), objN);
        vtx.matIndex = mi.matIndex;
        vtx.isHit = 1;
    }
    else
    {
        vtx.hitPos = float3(0.0);
        vtx.hitNormal = float3(0.0);
        vtx.matIndex = 0;
        vtx.isHit = 0;
    }

    vertices[vertexIdx] = vtx;
}
