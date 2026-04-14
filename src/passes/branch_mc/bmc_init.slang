// Branch MC: Initialize depth-0 vertices from camera rays
// 2D dispatch: [width/16, height/16, 1]
// Each pixel creates one depth-0 BMCVertex and traces primary ray via ray query

import bmc_structs;
import wavefront_structs;
import material;
import rng;

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

[[vk::binding(5)]]
RWStructuredBuffer<int2> depthRanges; // [startIdx, count] per depth

[[vk::binding(7)]]
ConstantBuffer<CameraProperties> cam;

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
[numthreads(16, 16, 1)]
void main(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    int2 pixel = int2(dispatchThreadID.xy);
    if (pixel.x >= pc.screenWidth || pixel.y >= pc.screenHeight)
        return;

    int pixelIdx = pixel.y * pc.screenWidth + pixel.x;
    int vertexIdx = pixelIdx; // depth-0 vertices occupy [0, totalPixels)

    // Generate camera ray
    uint seed = wangHash(uint(pixel.x * 1973u + pixel.y * 9277u + uint(pc.frameIndex) * 26699u) | 1u);
    float2 jitter = randFloat2(seed);
    float2 pixelCenter = float2(pixel) + jitter;
    float2 inUV = pixelCenter / float2(pc.screenWidth, pc.screenHeight);
    float2 d = inUV * 2.0 - 1.0;

    float4 origin = mul(cam.viewInverse, float4(0, 0, 0, 1));
    float4 target = mul(cam.projInverse, float4(d.x, d.y, 1, 1));
    float4 direction = mul(cam.viewInverse, float4(normalize(target.xyz), 0));

    float3 rayDir = normalize(direction.xyz);

    // Trace primary ray via ray query
    RayDesc rayDesc;
    rayDesc.Origin = origin.xyz;
    rayDesc.TMin = 0.001;
    rayDesc.Direction = rayDir;
    rayDesc.TMax = 10000.0;

    RayQuery<RAY_FLAG_FORCE_OPAQUE> rq;
    rq.TraceRayInline(topLevelAS, RAY_FLAG_FORCE_OPAQUE, 0xff, rayDesc);
    rq.Proceed();

    // Write hit info
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
    hitInfos[vertexIdx] = hit;

    // Initialize depth-0 vertex
    BMCVertex vtx;
    vtx.throughput = float3(1.0);
    vtx.seed = seed;
    vtx.radiance = float3(0.0);
    vtx.pixelIndex = pixelIdx;
    vtx.wo = -rayDir;
    vtx.depth = 0;
    vtx.matIndex = hit.matIndex;
    vtx.firstChild = -1;
    vtx.nextSibling = -1;
    vtx.parent = -1;
    vtx.alive = hit.isHit;
    vtx.nInnerSamples = 0;
    vtx.childIndex = 0;
    vtx.debiasRadianceIdx = -1;
    vtx.hitPos = hit.hitPos;
    vtx.prevBsdfPdf = 0.0;
    vtx.hitNormal = hit.hitNormal;
    vtx.styleType = 0;
    vtx.styleParam0 = 0.0;
    vtx.debiasingSamples = 0;
    vtx.actualBranchingDepth = 0;
    vtx.mis = 1.0;
    vertices[vertexIdx] = vtx;

    // Write depth range for depth 0 (first thread only)
    if (pixelIdx == 0)
    {
        depthRanges[0] = int2(0, pc.screenWidth * pc.screenHeight);
    }
}
