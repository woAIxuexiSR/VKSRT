import rng;

struct DataGenPushConstants
{
    uint64_t inputBuffer;
    uint64_t gtBuffer;
    uint   sampleCount;
    uint   seed;
    uint   inputDim;
};

[[vk::push_constant]]
DataGenPushConstants pc;

[numthreads(256, 1, 1)]
[shader("compute")]
void main(uint tid : SV_DispatchThreadID)
{
    if (tid >= pc.sampleCount)
        return;

    float* inputs = (float*)pc.inputBuffer;
    float* gts = (float*)pc.gtBuffer;

    uint rngState = wangHash(pc.seed + tid);
    float x = randFloat(rngState);
    float y = randFloat(rngState);

    // Fill all input fields with the same (x, y) pair
    for (uint i = 0; i < pc.inputDim; i += 2)
    {
        inputs[tid * pc.inputDim + i] = x;
        if (i + 1 < pc.inputDim)
            inputs[tid * pc.inputDim + i + 1] = y;
    }

    gts[tid * 3 + 0] = sin(3.14159265358979 * x);
    gts[tid * 3 + 1] = cos(3.14159265358979 * y);
    gts[tid * 3 + 2] = x * y;
}
