// Progressive accumulation compute shader
// Blends current frame with accumulated history using running average

struct AccumulateParams
{
    int frameIndex;
};

[[vk::push_constant]]
AccumulateParams params;

[[vk::binding(0)]]
Texture2D<float4> currentFrame;

[[vk::binding(1)]]
RWTexture2D<float4> accumBuffer;

[shader("compute")]
[numthreads(16, 16, 1)]
void main(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    uint2 dims;
    accumBuffer.GetDimensions(dims.x, dims.y);
    if (dispatchThreadID.x >= dims.x || dispatchThreadID.y >= dims.y)
        return;

    uint2 pixel = dispatchThreadID.xy;
    float3 newColor = currentFrame[pixel].rgb;

    float3 result;
    if (params.frameIndex > 0)
    {
        float3 prev = accumBuffer[pixel].rgb;
        float weight = 1.0 / float(params.frameIndex + 1);
        result = lerp(prev, newColor, weight);
    }
    else
    {
        result = newColor;
    }

    accumBuffer[pixel] = float4(result, 1.0);
}
