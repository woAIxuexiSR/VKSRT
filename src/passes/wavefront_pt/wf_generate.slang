// Wavefront PT: Generate primary camera rays
// 2D dispatch: [width/16, height/16, 1]

import wavefront_structs;
import camera;

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

[[vk::binding(5)]]
RWStructuredBuffer<uint> queue;

[[vk::binding(6)]]
[[vk::image_format("rgba32f")]]
RWTexture2D<float4> colorImage;

[[vk::binding(7)]]
ConstantBuffer<CameraProperties> cam;

// counterBuffer: [activeDispatchX, 1, 1, shadowDispatchX, 1, 1, nextActiveCount, shadowCount]
[[vk::binding(4)]]
RWStructuredBuffer<uint> counters;

[shader("compute")]
[numthreads(16, 16, 1)]
void main(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    int2 pixel = int2(dispatchThreadID.xy);
    if (pixel.x >= pc.screenWidth || pixel.y >= pc.screenHeight)
        return;

    int idx = pixel.y * pc.screenWidth + pixel.x;

    uint seed = seedFromPixel(uint2(pixel), uint(pc.frameIndex));
    float3 rayOrigin, rayDir;
    generatePrimaryRay(cam, uint2(pixel), uint2(pc.screenWidth, pc.screenHeight), seed, rayOrigin, rayDir);

    // Initialize path state
    PathState path;
    path.throughput = float3(1.0);
    path.seed = seed;
    path.L = float3(0.0);
    path.pixelIndex = idx;
    path.prevBsdfPdf = 0.0;
    path.depth = 0;
    path.alive = 1;
    path._pad = 0;
    pathStates[idx] = path;

    // Initialize ray
    RayData ray;
    ray.origin = rayOrigin;
    ray.tMin = 0.001;
    ray.direction = rayDir;
    ray.tMax = 10000.0;
    rays[idx] = ray;

    // Write to queue (first half)
    queue[idx] = uint(idx);

    // First thread writes initial activeCount
    if (idx == 0)
    {
        counters[6] = uint(pc.totalPaths); // nextActiveCount = totalPaths
    }
}
