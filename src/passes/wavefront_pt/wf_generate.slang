// Wavefront PT: Generate primary camera rays
// 2D dispatch: [width/16, height/16, 1]

import wavefront_structs;
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

    uint seed = wangHash(uint(pixel.x * 1973u + pixel.y * 9277u + uint(pc.frameIndex) * 26699u) | 1u);

    // Pixel jitter for anti-aliasing
    float2 jitter = randFloat2(seed);
    float2 pixelCenter = float2(pixel) + jitter;
    float2 inUV = pixelCenter / float2(pc.screenWidth, pc.screenHeight);
    float2 d = inUV * 2.0 - 1.0;

    float4 origin = mul(cam.viewInverse, float4(0, 0, 0, 1));
    float4 target = mul(cam.projInverse, float4(d.x, d.y, 1, 1));
    float4 direction = mul(cam.viewInverse, float4(normalize(target.xyz), 0));

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
    ray.origin = origin.xyz;
    ray.tMin = 0.001;
    ray.direction = normalize(direction.xyz);
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
