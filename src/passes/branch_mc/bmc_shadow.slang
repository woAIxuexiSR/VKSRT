// Branch MC: Shadow ray kernel
// Traces shadow rays for NEE, accumulates contribution if unoccluded
// 1D indirect dispatch from counterBuffer[3..5]

import bmc_structs;
import wavefront_structs;

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

[[vk::binding(3)]]
RWStructuredBuffer<ShadowRay> shadowRayBuf;

[[vk::binding(8)]]
RaytracingAccelerationStructure topLevelAS;

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    uint idx = dispatchThreadID.x;
    ShadowRay sr = shadowRayBuf[idx];
    if (sr.pathIndex < 0 || sr.pathIndex >= pc.maxVertices) return;

    RayDesc rayDesc;
    rayDesc.Origin = sr.origin;
    rayDesc.TMin = 0.001;
    rayDesc.Direction = sr.direction;
    rayDesc.TMax = sr.tMax;

    RayQuery<RAY_FLAG_FORCE_OPAQUE | RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH> rq;
    rq.TraceRayInline(topLevelAS,
                      RAY_FLAG_FORCE_OPAQUE | RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH,
                      0xff, rayDesc);
    rq.Proceed();

    if (rq.CommittedStatus() == COMMITTED_NOTHING)
    {
        // Unoccluded: add NEE contribution to the vertex's radiance
        vertices[sr.pathIndex].radiance += sr.contribution;
    }
}
