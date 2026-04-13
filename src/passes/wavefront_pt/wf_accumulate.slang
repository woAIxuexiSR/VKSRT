// Wavefront PT: Accumulate kernel
// Writes final pixel colors from pathState.L to output image
// 2D dispatch: [width/16, height/16, 1]

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

    int idx = pixel.y * pc.screenWidth + pixel.x;
    float3 L = pathStates[idx].L;

    colorImage[pixel] = float4(L, 1.0);
}
