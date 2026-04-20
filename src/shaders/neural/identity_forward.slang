struct IdentityForwardPC
{
    uint64_t inputBuffer;
    uint64_t outputBuffer;
    uint sampleCount;
    uint dim;
    uint inputFieldOffset;
    uint inputStride;
    uint outputFieldOffset;
    uint outputStride;
};

[[vk::push_constant]]
IdentityForwardPC pc;

[numthreads(256, 1, 1)]
[shader("compute")]
void main(uint tid : SV_DispatchThreadID)
{
    if (tid >= pc.sampleCount)
        return;

    float* inp = (float*)pc.inputBuffer;
    float* out = (float*)pc.outputBuffer;

    for (uint j = 0; j < pc.dim; j++)
        out[tid * pc.outputStride + pc.outputFieldOffset + j] =
            inp[tid * pc.inputStride + pc.inputFieldOffset + j];
}
