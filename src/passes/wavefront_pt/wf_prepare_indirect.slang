// Wavefront PT: Prepare indirect dispatch arguments from atomic counters
// Dispatch: (1, 1, 1) — single thread

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

// counterBuffer layout:
// [0] = activeDispatchX
// [1] = 1 (dispatchY)
// [2] = 1 (dispatchZ)
// [3] = shadowDispatchX
// [4] = 1 (dispatchY)
// [5] = 1 (dispatchZ)
// [6] = nextActiveCount (atomic, written by shade)
// [7] = shadowCount (atomic, written by shade)

[[vk::binding(4)]]
RWStructuredBuffer<uint> counters;

[shader("compute")]
[numthreads(1, 1, 1)]
void main()
{
    uint activeCount = counters[6];
    uint shadowCount = counters[7];

    // Compute dispatch group counts (256 threads per group)
    counters[0] = (activeCount + 255u) / 256u;
    counters[1] = 1u;
    counters[2] = 1u;

    counters[3] = (shadowCount + 255u) / 256u;
    counters[4] = 1u;
    counters[5] = 1u;

    // Reset atomic counters for next bounce
    counters[6] = 0u;
    counters[7] = 0u;
}
