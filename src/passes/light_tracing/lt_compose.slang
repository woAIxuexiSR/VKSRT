// Compose shader: merge 3 single-channel float splat images into RGBA output

[[vk::binding(0)]]
[[vk::image_format("r32f")]]
RWTexture2D<float> splatR;

[[vk::binding(1)]]
[[vk::image_format("r32f")]]
RWTexture2D<float> splatG;

[[vk::binding(2)]]
[[vk::image_format("r32f")]]
RWTexture2D<float> splatB;

[[vk::binding(3)]]
RWTexture2D<float4> outputImage;

[shader("compute")]
[numthreads(16, 16, 1)]
void main(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    uint2 pixel = dispatchThreadID.xy;

    uint2 dims;
    outputImage.GetDimensions(dims.x, dims.y);
    if (pixel.x >= dims.x || pixel.y >= dims.y) return;

    float r = splatR[pixel];
    float g = splatG[pixel];
    float b = splatB[pixel];

    outputImage[pixel] = float4(r, g, b, 1.0);
}
