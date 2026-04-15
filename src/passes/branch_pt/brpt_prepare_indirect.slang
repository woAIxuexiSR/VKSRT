// Branch PT: Prepare indirect dispatch arguments + record depth range
// Dispatch: (1, 1, 1) — single thread
//
// Called after init and after each advance dispatch (forward only).
// Reads counters[0] (total allocated vertices so far),
// computes the new depth's vertex range from the previous depth's end,
// writes indirect dispatch args for forward advance (counters[2,3,4])
// and stores per-depth args (counters[5+depth*3..]) for backward propagate.

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
    int rrDepth;
};

[[vk::push_constant]]
PushConstants pc;

[[vk::binding(1)]]
RWStructuredBuffer<uint> counters; // [0]=vtx, [1]=debias, [2-4]=current indirect, [5+depth*3]=per-depth indirect

[[vk::binding(2)]]
RWStructuredBuffer<int2> depthRanges; // [startIdx, count] per depth

[shader("compute")]
[numthreads(1, 1, 1)]
void main()
{
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

    uint groups = (uint(max(depthCount, 0)) + 255u) / 256u;

    // Write current indirect dispatch args (used by forward advance)
    counters[2] = groups;
    counters[3] = 1u;
    counters[4] = 1u;

    // Store per-depth indirect args (used by backward propagate)
    int base = 5 + pc.currentDepth * 3;
    counters[base + 0] = groups;
    counters[base + 1] = 1u;
    counters[base + 2] = 1u;
}
