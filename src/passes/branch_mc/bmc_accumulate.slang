// Branch MC: Accumulate — write depth-0 vertex radiance to color image
// 2D dispatch: [width/16, height/16, 1]

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

[[vk::binding(0)]]
RWStructuredBuffer<BMCVertex> vertices;

[[vk::binding(6)]]
[[vk::image_format("rgba32f")]]
RWTexture2D<float4> colorImage;

[shader("compute")]
[numthreads(16, 16, 1)]
void main(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    int2 pixel = int2(dispatchThreadID.xy);
    if (pixel.x >= pc.screenWidth || pixel.y >= pc.screenHeight)
        return;

    int pixelIdx = pixel.y * pc.screenWidth + pixel.x;

    // Depth-0 vertex is at index pixelIdx
    float3 L = vertices[pixelIdx].radiance;

    // Clamp to non-negative
    colorImage[pixel] = float4(max(L, float3(0.0)), 1.0);
}
