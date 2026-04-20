// Write MLP output buffer to an image. Also supports showing ground truth.
// Each thread handles one pixel.
static const float PI = 3.14159265358979323846;

struct WriteImagePushConstants
{
    uint64_t outputBuffer;  // float[width * height * outputSize]
    uint width;
    uint height;
    uint outputSize;
    uint showGT;
};

[[vk::push_constant]]
WriteImagePushConstants pc;

[[vk::binding(0, 0)]]
RWTexture2D<float4> outputImage;

[numthreads(16, 16, 1)]
[shader("compute")]
void main(uint3 dtid : SV_DispatchThreadID)
{
    if (dtid.x >= pc.width || dtid.y >= pc.height)
        return;

    float u = (float(dtid.x) + 0.5) / float(pc.width);
    float v = (float(dtid.y) + 0.5) / float(pc.height);

    float3 color;

    if (pc.showGT != 0)
    {
        color = float3(sin(PI * u), cos(PI * v), u * v);
    }
    else
    {
        float* outputs = (float*)pc.outputBuffer;
        uint idx = (dtid.y * pc.width + dtid.x) * pc.outputSize;

        color = float3(0.0);
        if (pc.outputSize >= 1) color.x = outputs[idx + 0];
        if (pc.outputSize >= 2) color.y = outputs[idx + 1];
        if (pc.outputSize >= 3) color.z = outputs[idx + 2];
    }

    color = clamp(color, float3(0.0), float3(1.0));
    outputImage[int2(dtid.x, dtid.y)] = float4(color, 1.0);
}
