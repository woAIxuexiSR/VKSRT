// Tone mapping compute shader
// Reads HDR input, applies ACES tone mapping, writes linear LDR output
// Note: sRGB gamma is handled by the swapchain (VK_FORMAT_B8G8R8A8_SRGB)

[[vk::binding(0)]]
Texture2D<float4> inputImage;

[[vk::binding(1)]]
RWTexture2D<float4> outputImage;

struct TonemapParams
{
    float exposure;
};

[[vk::push_constant]]
TonemapParams params;

// ACES filmic tone mapping
float3 acesTonemap(float3 x)
{
    float a = 2.51;
    float b = 0.03;
    float c = 2.43;
    float d = 0.59;
    float e = 0.14;
    return clamp((x * (a * x + b)) / (x * (c * x + d) + e), 0.0, 1.0);
}

[shader("compute")]
[numthreads(16, 16, 1)]
void main(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    uint2 dims;
    outputImage.GetDimensions(dims.x, dims.y);
    if (dispatchThreadID.x >= dims.x || dispatchThreadID.y >= dims.y)
        return;

    float4 hdrColor = inputImage[dispatchThreadID.xy];
    float3 exposed = hdrColor.rgb * params.exposure;
    float3 mapped = acesTonemap(exposed);
    outputImage[dispatchThreadID.xy] = float4(mapped, 1.0);
}
