struct FrequencyForwardPC
{
    uint64_t inputBuffer;
    uint64_t outputBuffer;
    uint sampleCount;
    uint inputDim;
    uint numFreqs;
    uint inputFieldOffset;
    uint inputStride;
    uint outputFieldOffset;
    uint outputStride;
};

[[vk::push_constant]]
FrequencyForwardPC pc;

void encodeFreqDim(float* out, uint baseOut, uint dimOffset, float x, uint numFreqs)
{
    [ForceUnroll]
    for (int k = 0; k < 12; k++)
    {
        if (uint(k) >= numFreqs) break;
        float scale = exp2(float(k)) * 3.14159265358979;
        float val = scale * x;
        out[baseOut + dimOffset + uint(k) * 2]     = sin(val);
        out[baseOut + dimOffset + uint(k) * 2 + 1] = cos(val);
    }
}

[numthreads(256, 1, 1)]
[shader("compute")]
void main(uint tid : SV_DispatchThreadID)
{
    if (tid >= pc.sampleCount)
        return;

    float* inp = (float*)pc.inputBuffer;
    float* out = (float*)pc.outputBuffer;

    uint baseOut = tid * pc.outputStride + pc.outputFieldOffset;
    uint baseIn  = tid * pc.inputStride + pc.inputFieldOffset;
    uint stride2 = pc.numFreqs * 2;

    encodeFreqDim(out, baseOut, 0, inp[baseIn + 0], pc.numFreqs);
    if (pc.inputDim > 1)
        encodeFreqDim(out, baseOut, stride2, inp[baseIn + 1], pc.numFreqs);
    if (pc.inputDim > 2)
        encodeFreqDim(out, baseOut, stride2 * 2, inp[baseIn + 2], pc.numFreqs);
    if (pc.inputDim > 3)
        encodeFreqDim(out, baseOut, stride2 * 3, inp[baseIn + 3], pc.numFreqs);
}
