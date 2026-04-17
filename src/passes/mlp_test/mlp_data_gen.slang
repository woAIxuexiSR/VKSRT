// GPU data generation kernel for MLP test: random (x,y) -> gt = (sin(pi*x), cos(pi*y), x*y)
import rng;

struct DataGenPushConstants
{
    uint64_t inputBuffer;   // float[sampleCount * 2]
    uint64_t gtBuffer;      // float[sampleCount * 3]
    uint   sampleCount;
    uint   seed;
};

[[vk::push_constant]]
DataGenPushConstants pc;

static const float PI = 3.14159265358979323846;

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

    inputs[tid * 2 + 0] = x;
    inputs[tid * 2 + 1] = y;

    gts[tid * 3 + 0] = sin(PI * x);
    gts[tid * 3 + 1] = cos(PI * y);
    gts[tid * 3 + 2] = x * y;
}
