// Joint bilateral filter: edge-preserving denoising using G-buffer
// Reads accumulated HDR color + first-hit normal/position
// Preserves edges where normals or positions differ significantly

struct BilateralParams
{
    float sigmaS;      // spatial sigma (pixels)
    float sigmaN;      // normal similarity threshold
    float sigmaP;      // position similarity threshold
    int kernelRadius;  // filter radius in pixels
};

[[vk::push_constant]]
BilateralParams params;

[[vk::binding(0)]]
Texture2D<float4> inputColor;

[[vk::binding(1)]]
RWTexture2D<float4> outputColor;

[[vk::binding(2)]]
Texture2D<float4> gbufferNormal;

[[vk::binding(3)]]
Texture2D<float4> gbufferPosition;

[shader("compute")]
[numthreads(16, 16, 1)]
void main(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    uint2 dims;
    outputColor.GetDimensions(dims.x, dims.y);
    if (dispatchThreadID.x >= dims.x || dispatchThreadID.y >= dims.y)
        return;

    int2 center = int2(dispatchThreadID.xy);

    float3 centerColor = inputColor[center].rgb;
    float3 centerNormal = gbufferNormal[center].rgb;
    float3 centerPos = gbufferPosition[center].rgb;
    float centerW = gbufferNormal[center].a;

    // If no valid geometry (miss), pass through
    if (centerW < 0.5)
    {
        outputColor[center] = float4(centerColor, 1.0);
        return;
    }

    float invSigmaS2 = 1.0 / (2.0 * params.sigmaS * params.sigmaS);
    float invSigmaN2 = 1.0 / (2.0 * params.sigmaN * params.sigmaN);
    float invSigmaP2 = 1.0 / (2.0 * params.sigmaP * params.sigmaP);

    float3 sumColor = float3(0.0);
    float sumWeight = 0.0;

    int r = params.kernelRadius;
    for (int dy = -r; dy <= r; dy++)
    {
        for (int dx = -r; dx <= r; dx++)
        {
            int2 neighbor = center + int2(dx, dy);

            // Bounds check
            if (neighbor.x < 0 || neighbor.y < 0 ||
                neighbor.x >= int(dims.x) || neighbor.y >= int(dims.y))
                continue;

            float3 nColor = inputColor[neighbor].rgb;
            float3 nNormal = gbufferNormal[neighbor].rgb;
            float3 nPos = gbufferPosition[neighbor].rgb;
            float nW = gbufferNormal[neighbor].a;

            // Skip invalid geometry
            if (nW < 0.5)
                continue;

            // Spatial weight: Gaussian based on pixel distance
            float dist2 = float(dx * dx + dy * dy);
            float wSpatial = exp(-dist2 * invSigmaS2);

            // Normal weight: penalize large normal differences
            float normalDiff = 1.0 - max(dot(centerNormal, nNormal), 0.0);
            float wNormal = exp(-normalDiff * normalDiff * invSigmaN2);

            // Position weight: penalize large position differences
            float3 posDiff = centerPos - nPos;
            float posDist2 = dot(posDiff, posDiff);
            float wPosition = exp(-posDist2 * invSigmaP2);

            float w = wSpatial * wNormal * wPosition;
            sumColor += nColor * w;
            sumWeight += w;
        }
    }

    float3 result = (sumWeight > 1e-6) ? sumColor / sumWeight : centerColor;
    outputColor[center] = float4(result, 1.0);
}
