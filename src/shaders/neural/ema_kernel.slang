struct EmaPushConstants
{
    uint64_t inferParams;
    uint64_t trainParams;
    uint count;
    float alpha;
};

[[vk::push_constant]]
EmaPushConstants pc;

[numthreads(256, 1, 1)]
[shader("compute")]
void main(uint tid : SV_DispatchThreadID)
{
    if (tid >= pc.count)
        return;

    float* dst = (float*)pc.inferParams;
    float* src = (float*)pc.trainParams;

    dst[tid] = pc.alpha * dst[tid] + (1.0 - pc.alpha) * src[tid];
}
