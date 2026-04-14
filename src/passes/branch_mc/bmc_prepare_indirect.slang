// Branch MC: Prepare indirect dispatch arguments from atomic counters
// Dispatch: (1, 1, 1) — single thread

import bmc_structs;

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

[[vk::binding(4)]]
RWStructuredBuffer<uint> counters;

[shader("compute")]
[numthreads(1, 1, 1)]
void main()
{
    uint activeCount = counters[6];
    uint shadowCount = counters[7];

    // Indirect dispatch args for extend: [groupX, 1, 1]
    counters[0] = (activeCount + 255u) / 256u;
    counters[1] = 1u;
    counters[2] = 1u;

    // Indirect dispatch args for shadow: [groupX, 1, 1]
    counters[3] = (shadowCount + 255u) / 256u;
    counters[4] = 1u;
    counters[5] = 1u;

    // Reset atomic counters for next use
    counters[6] = 0u;
    counters[7] = 0u;
}
