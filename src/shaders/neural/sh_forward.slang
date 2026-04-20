import sh;

struct SHForwardPC
{
    uint64_t inputBuffer;
    uint64_t outputBuffer;
    uint sampleCount;
    uint degree;
    uint inputFieldOffset;
    uint inputStride;
    uint outputFieldOffset;
    uint outputStride;
};

[[vk::push_constant]]
SHForwardPC pc;

[numthreads(256, 1, 1)]
[shader("compute")]
void main(uint tid : SV_DispatchThreadID)
{
    if (tid >= pc.sampleCount)
        return;

    float* inp = (float*)pc.inputBuffer;
    float* out = (float*)pc.outputBuffer;

    float x = inp[tid * pc.inputStride + pc.inputFieldOffset + 0];
    float y = inp[tid * pc.inputStride + pc.inputFieldOffset + 1];
    float z = inp[tid * pc.inputStride + pc.inputFieldOffset + 2];

    float basis[MAX_SH_COEFFS];
    for (int i = 0; i < MAX_SH_COEFFS; i++) basis[i] = 0.0;
    evaluateSHBasis(x, y, z, int(pc.degree), basis);

    uint numCoeffs = (pc.degree + 1) * (pc.degree + 1);
    for (uint j = 0; j < numCoeffs; j++)
        out[tid * pc.outputStride + pc.outputFieldOffset + j] = basis[j];
}
