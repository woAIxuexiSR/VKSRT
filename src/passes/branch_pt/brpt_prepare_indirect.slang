// Branch PT: Prepare indirect dispatch arguments + record depth range
// Dispatch: (1, 1, 1) — single thread
//
// Called after init and after each advance dispatch.
// Reads counters[0] (total allocated vertices so far),
// computes the new depth's vertex range from the previous depth's end,
// writes indirect dispatch args and depth range for the next advance or propagate.

import brpt_structs;

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
    int backwardMode;
    int rrDepth;
};

[[vk::push_constant]]
PushConstants pc;

[[vk::binding(1)]]
RWStructuredBuffer<uint> counters; // [0]=vtxCounter, [1]=debiasCounter, [2,3,4]=indirect dispatch

[[vk::binding(2)]]
RWStructuredBuffer<int2> depthRanges; // [startIdx, count] per depth

[shader("compute")]
[numthreads(1, 1, 1)]
void main()
{
    // _pad0 == 1: backward mode — just read existing depthRanges and set indirect args
    if (pc.backwardMode == 1)
    {
        int2 range = depthRanges[pc.currentDepth];
        int depthCount = range.y;
        counters[2] = (uint(max(depthCount, 0)) + 255u) / 256u;
        counters[3] = 1u;
        counters[4] = 1u;
        return;
    }

    // Forward mode: compute depth range from cumulative counter
    // counters[0] = total vertices allocated so far (cumulative)
    uint totalVertices = counters[0];

    // Compute this depth's vertex range
    int depthStart;
    if (pc.currentDepth == 0)
    {
        depthStart = 0;
    }
    else
    {
        int2 prevRange = depthRanges[pc.currentDepth - 1];
        depthStart = prevRange.x + prevRange.y;
    }

    int depthCount = int(totalVertices) - depthStart;
    if (depthCount < 0) depthCount = 0;

    // Write this depth's range
    depthRanges[pc.currentDepth] = int2(depthStart, depthCount);

    // Write indirect dispatch args for advance kernel on this depth
    counters[2] = (uint(depthCount) + 255u) / 256u;
    counters[3] = 1u;
    counters[4] = 1u;
}
