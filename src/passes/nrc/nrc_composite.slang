struct CompositePushConstants
{
    uint64_t queryOutputAddr;
    uint width;
    uint height;
};

[[vk::push_constant]]
CompositePushConstants pc;

[[vk::binding(0, 0)]]
RWTexture2D<float4> shortPathImage;

[[vk::binding(1, 0)]]
RWTexture2D<float4> throughputImage;

[[vk::binding(2, 0)]]
RWTexture2D<float4> colorImage;

[numthreads(16, 16, 1)]
[shader("compute")]
void main(uint3 dtid : SV_DispatchThreadID)
{
    if (dtid.x >= pc.width || dtid.y >= pc.height)
        return;

    float4 sp = shortPathImage[int2(dtid.xy)];
    float3 shortPathL = sp.rgb;
    bool needsNRC = sp.a > 0.5;

    float3 finalColor;

    if (needsNRC)
    {
        float3 throughput = throughputImage[int2(dtid.xy)].rgb;

        float* queryOut = (float*)pc.queryOutputAddr;
        uint pixelIndex = dtid.y * pc.width + dtid.x;
        uint base = pixelIndex * 3;
        float3 nrcPrediction = float3(queryOut[base], queryOut[base + 1], queryOut[base + 2]);
        nrcPrediction = max(nrcPrediction, float3(0.0));

        finalColor = shortPathL + throughput * nrcPrediction;
    }
    else
    {
        finalColor = shortPathL;
    }

    colorImage[int2(dtid.xy)] = float4(finalColor, 1.0);
}
