// Wavefront PT: Shadow ray kernel
// Traces shadow rays for NEE, accumulates contribution if unoccluded
// 1D indirect dispatch from counterBuffer[3..5]

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
    int totalPaths;
    int bounceIndex;
    int queueOffset;
    int _pad;
};

[[vk::push_constant]]
PushConstants pc;

[[vk::binding(0)]]
RWStructuredBuffer<PathState> pathStates;

[[vk::binding(3)]]
RWStructuredBuffer<ShadowRay> shadowRayBuf;

[[vk::binding(8)]]
RaytracingAccelerationStructure topLevelAS;

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    uint idx = dispatchThreadID.x;
    // shadowCount was used to compute dispatch size via PrepareIndirect;
    // threads beyond the actual shadow ray count will read garbage, so
    // we rely on the dispatch size being correct.
    // But for safety, check pathIndex validity.
    ShadowRay sr = shadowRayBuf[idx];
    if (sr.pathIndex < 0 || sr.pathIndex >= pc.totalPaths) return;

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
        // Unoccluded: add NEE contribution
        // Safe atomic-free write: each path has at most one shadow ray per bounce
        pathStates[sr.pathIndex].L += sr.contribution;
    }
}
