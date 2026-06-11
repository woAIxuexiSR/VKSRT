// ICV compose shader: deterministically gathers edge estimates and boundary fallbacks.

struct PushConstants
{
    int frameIndex;
    int maxDepth;
    int rrDepth;
    int lightCount;
    int useNEE;
    int useMIS;
    float a;
    float gaussianSigma;
    int screenWidth;
    int screenHeight;
    int _pad0;
    int _pad1;
};

[[vk::push_constant]]
PushConstants pc;

[[vk::binding(0)]]
Texture2D<float4> anchorImage;

[[vk::binding(1)]]
Texture2D<float4> hEdgeLeft;

[[vk::binding(2)]]
Texture2D<float4> hEdgeRight;

[[vk::binding(3)]]
Texture2D<float4> vEdgeUp;

[[vk::binding(4)]]
Texture2D<float4> vEdgeDown;

[[vk::binding(5)]]
RWTexture2D<float4> outputImage;

float gaussianWeight(float2 delta)
{
    float sigma = max(pc.gaussianSigma, 1e-3);
    return exp(-dot(delta, delta) / (2.0 * sigma * sigma));
}

void addSample(inout float3 sum, inout float weight, float4 sample, float3 fallback)
{
    float w = max(sample.a, 0.0);
    float3 v = sample.rgb;
    if (any(isnan(v)) || any(isinf(v)) || isnan(w) || isinf(w) || w <= 0.0)
        return;
    sum += v * w;
    weight += w;
}

[shader("compute")]
[numthreads(16, 16, 1)]
void main(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    int2 pixel = int2(dispatchThreadID.xy);
    if (pixel.x >= pc.screenWidth || pixel.y >= pc.screenHeight)
        return;

    float3 anchor = anchorImage[pixel].rgb;
    float3 sum = float3(0.0);
    float weight = 0.0;

    // Center sample: the pixel's own anchor estimate participates in the Gaussian stencil.
    float centerWeight = gaussianWeight(float2(0.0, 0.0));
    sum += anchor * centerWeight;
    weight += centerWeight;

    float neighborWeight = gaussianWeight(float2(1.0, 0.0));

    // Left neighbor: read the contribution to the right endpoint of edge (pixel-left, pixel).
    if (pixel.x > 0)
        addSample(sum, weight, hEdgeRight[pixel - int2(1, 0)], anchor);
    else
    {
        sum += anchor * neighborWeight;
        weight += neighborWeight;
    }

    // Right neighbor: read the contribution to the left endpoint of edge (pixel, pixel+right).
    if (pixel.x + 1 < pc.screenWidth)
        addSample(sum, weight, hEdgeLeft[pixel], anchor);
    else
    {
        sum += anchor * neighborWeight;
        weight += neighborWeight;
    }

    // Up neighbor: read the contribution to the down endpoint of edge (pixel-up, pixel).
    if (pixel.y > 0)
        addSample(sum, weight, vEdgeDown[pixel - int2(0, 1)], anchor);
    else
    {
        sum += anchor * neighborWeight;
        weight += neighborWeight;
    }

    // Down neighbor: read the contribution to the up endpoint of edge (pixel, pixel+down).
    if (pixel.y + 1 < pc.screenHeight)
        addSample(sum, weight, vEdgeUp[pixel], anchor);
    else
    {
        sum += anchor * neighborWeight;
        weight += neighborWeight;
    }

    float3 result = weight > 1e-8 ? sum / weight : anchor;
    outputImage[pixel] = float4(result, 1.0);
}
