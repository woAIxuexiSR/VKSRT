// Neural Radiosity composite: reads network prediction at primary hits from a BDA
// buffer and writes the final color image. Emissive primary hits and misses are
// passed through via primaryInfoBuffer (rgb = direct emission or 0; a = 1 if
// non-emissive hit, 0 for emissive/miss — in both non-hit cases info.rgb is used
// directly, so emissive shows as emission and miss shows as black).
//
// showBlack=1: training-only offline mode → write pure black, skip network read.

struct CompositePC
{
    uint64_t queryOutputAddr;   // P * 3 floats
    uint64_t primaryInfoAddr;   // P * 4 floats
    uint width;
    uint height;
    uint showBlack;
    uint _pad0;
};

[[vk::push_constant]]
CompositePC pc;

[[vk::binding(0, 0)]]
RWTexture2D<float4> colorImage;

[numthreads(16, 16, 1)]
[shader("compute")]
void main(uint3 dtid : SV_DispatchThreadID)
{
    if (dtid.x >= pc.width || dtid.y >= pc.height)
        return;

    if (pc.showBlack != 0)
    {
        colorImage[int2(dtid.xy)] = float4(0.0, 0.0, 0.0, 1.0);
        return;
    }

    uint px = dtid.y * pc.width + dtid.x;

    // primaryInfoBuffer is laid out as 2 float4 per pixel: [info0, throughput]
    float4* info = (float4*)pc.primaryInfoAddr;
    float4 pi = info[px * 2 + 0];

    float3 color;
    if (pi.w > 0.5)
    {
        float3 throughput = info[px * 2 + 1].rgb;
        float* netOut = (float*)pc.queryOutputAddr;
        uint base = px * 3;
        float3 lY = max(float3(netOut[base + 0], netOut[base + 1], netOut[base + 2]),
                        float3(0.0));
        color = throughput * lY;
    }
    else
    {
        color = pi.rgb;
    }

    colorImage[int2(dtid.xy)] = float4(color, 1.0);
}
