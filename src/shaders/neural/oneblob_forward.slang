struct OneBlobForwardPC
{
    uint64_t inputBuffer;
    uint64_t outputBuffer;
    uint sampleCount;
    uint inputDim;
    uint numBins;
    float sigma;
    uint inputFieldOffset;
    uint inputStride;
    uint outputFieldOffset;
    uint outputStride;
};

[[vk::push_constant]]
OneBlobForwardPC pc;

void encodeBlobDim(float* out, uint baseOut, uint dimOffset, float x, uint numBins, float invBinsM1, float invTwoSigmaSq)
{
    [ForceUnroll]
    for (int b = 0; b < 32; b++)
    {
        if (uint(b) >= numBins) break;
        float center = float(b) * invBinsM1;
        float diff = x - center;
        out[baseOut + dimOffset + uint(b)] = exp(-diff * diff * invTwoSigmaSq);
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

    float invTwoSigmaSq = 0.5 / (pc.sigma * pc.sigma);
    uint baseOut = tid * pc.outputStride + pc.outputFieldOffset;
    uint baseIn  = tid * pc.inputStride + pc.inputFieldOffset;
    float invBinsM1 = 1.0 / float(pc.numBins - 1);

    encodeBlobDim(out, baseOut, 0, inp[baseIn + 0], pc.numBins, invBinsM1, invTwoSigmaSq);
    if (pc.inputDim > 1)
        encodeBlobDim(out, baseOut, pc.numBins, inp[baseIn + 1], pc.numBins, invBinsM1, invTwoSigmaSq);
    if (pc.inputDim > 2)
        encodeBlobDim(out, baseOut, pc.numBins * 2, inp[baseIn + 2], pc.numBins, invBinsM1, invTwoSigmaSq);
    if (pc.inputDim > 3)
        encodeBlobDim(out, baseOut, pc.numBins * 3, inp[baseIn + 3], pc.numBins, invBinsM1, invTwoSigmaSq);
}
