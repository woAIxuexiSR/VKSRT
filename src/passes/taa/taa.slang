// Temporal Anti-Aliasing compute shader
// Modes: TAA (reprojection + neighborhood clamping) or Accumulate (running average fallback)

struct TAAPushConstants
{
    float blendFactor;
    int frameIndex;
    int useAccumMode; // 0=TAA, 1=accumulate
    int _pad0;
};

struct TAAUniform
{
    float4x4 prevViewProj;
    int screenWidth;
    int screenHeight;
};

[[vk::push_constant]]
TAAPushConstants pc;

[[vk::binding(0)]]
Texture2D<float4> currentFrame;

[[vk::binding(1)]]
RWTexture2D<float4> outputImage;

[[vk::binding(2)]]
Texture2D<float4> historyBuffer;

[[vk::binding(2)]]
SamplerState historySampler;

[[vk::binding(3)]]
Texture2D<float4> gbufferPosition;

[[vk::binding(4)]]
ConstantBuffer<TAAUniform> taaParams;

// Reproject world position to previous frame UV
bool reprojectToPrevFrame(float3 worldPos, out float2 prevUV)
{
    float4 clipPos = mul(taaParams.prevViewProj, float4(worldPos, 1.0));
    if (clipPos.w <= 0.0)
        return false;
    float3 ndc = clipPos.xyz / clipPos.w;
    if (any(abs(ndc.xy) > 1.0))
        return false;
    prevUV = ndc.xy * 0.5 + 0.5;
    return true;
}

// Compute 3x3 neighborhood color AABB for clamping
void computeNeighborhoodAABB(int2 pixel, int2 dims, out float3 boxMin, out float3 boxMax)
{
    boxMin = float3(1e10);
    boxMax = float3(-1e10);

    for (int dy = -1; dy <= 1; dy++)
    {
        for (int dx = -1; dx <= 1; dx++)
        {
            int2 p = clamp(pixel + int2(dx, dy), int2(0), dims - 1);
            float3 c = currentFrame[p].rgb;
            boxMin = min(boxMin, c);
            boxMax = max(boxMax, c);
        }
    }
}

[shader("compute")]
[numthreads(16, 16, 1)]
void main(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    int2 pixel = int2(dispatchThreadID.xy);
    int2 dims = int2(taaParams.screenWidth, taaParams.screenHeight);

    if (pixel.x >= dims.x || pixel.y >= dims.y)
        return;

    float3 currentColor = currentFrame[pixel].rgb;

    // --- Accumulate mode (static camera): running average ---
    if (pc.useAccumMode != 0)
    {
        float3 result;
        if (pc.frameIndex > 0)
        {
            float3 prev = historyBuffer.SampleLevel(historySampler, (float2(pixel) + 0.5) / float2(dims), 0).rgb;
            float weight = 1.0 / float(pc.frameIndex + 1);
            result = lerp(prev, currentColor, weight);
        }
        else
        {
            result = currentColor;
        }
        outputImage[pixel] = float4(result, 1.0);
        return;
    }

    // --- TAA mode (camera moving): reprojection + clamping ---

    // Read G-buffer position for reprojection
    float4 posData = gbufferPosition[pixel];
    if (posData.a < 0.5)
    {
        // Miss pixel (background): use current frame directly
        outputImage[pixel] = float4(currentColor, 1.0);
        return;
    }

    // Reproject to previous frame
    float2 prevUV;
    bool validReproj = reprojectToPrevFrame(posData.xyz, prevUV);

    if (!validReproj || pc.frameIndex == 0)
    {
        // Reprojection failed or first frame: use current frame
        outputImage[pixel] = float4(currentColor, 1.0);
        return;
    }

    // Sample history at reprojected location (bilinear via combined_image_sampler)
    float3 historyColor = historyBuffer.SampleLevel(historySampler, prevUV, 0).rgb;

    // Neighborhood clamping: clamp history to current frame's local color range
    float3 boxMin, boxMax;
    computeNeighborhoodAABB(pixel, dims, boxMin, boxMax);
    historyColor = clamp(historyColor, boxMin, boxMax);

    // Blend: lerp(history, current, blendFactor) where blendFactor = current frame weight
    float3 result = lerp(historyColor, currentColor, pc.blendFactor);

    outputImage[pixel] = float4(result, 1.0);
}
