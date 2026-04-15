// Branch PT: Accumulate — write depth-0 vertex radiance to color image
// Indirect 1D dispatch using depthRanges[0].count via counterBuffer indirect args,
// or dispatched with upper bound (tilePixelCount).
// Each depth-0 vertex scatters its radiance to its pixel.

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

[[vk::binding(0)]]
RWStructuredBuffer<BrPTVertex> vertices;

[[vk::binding(2)]]
RWStructuredBuffer<int2> depthRanges;

[[vk::binding(3)]]
[[vk::image_format("rgba32f")]]
RWTexture2D<float4> colorImage;

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    // Read depth-0 range
    int2 range = depthRanges[0];
    int depthStart = range.x;
    int depthCount = range.y;

    uint tid = dispatchThreadID.x;
    if (tid >= uint(depthCount)) return;

    int vtxIdx = depthStart + int(tid);
    if (vtxIdx >= pc.maxVertices) return;

    BrPTVertex vtx = vertices[vtxIdx];
    int pixelIdx = vtx.pixelIndex;
    int2 pixel = int2(pixelIdx % pc.screenWidth, pixelIdx / pc.screenWidth);

    // Clamp to non-negative and write
    colorImage[pixel] = float4(max(vtx.radiance, float3(0.0)), 1.0);
}
