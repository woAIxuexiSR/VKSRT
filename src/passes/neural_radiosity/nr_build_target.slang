// Neural Radiosity target builder: combines the auxiliary data written by sample_gen
// (NEE term at x, throughput at x, emission-at-y with MIS already applied) with the
// network's EMA forward prediction at y to produce the GT target for the LHS train step.
//
// target = neeContrib
//        + (validEmissive ? emissionTerm
//        : validNet       ? throughputNet * rhsPred
//                         : 0)
//
// Since rhsPred is computed with the EMA (shadow) network and written verbatim into
// gtBuffer, stop-gradient on RHS is automatic — recordTrain only differentiates w.r.t.
// the LHS forward pass and treats gtBuffer as a constant target.

struct BuildTargetPC
{
    uint64_t auxAddr;       // B * S * 12 floats
    uint64_t rhsPredAddr;   // B * S * 3 floats
    uint64_t gtAddr;        // B * 3 floats (per-batch mean over S sub-samples)
    uint batchSize;
    uint rhsSamples;
};

[[vk::push_constant]]
BuildTargetPC pc;

// Dummy descriptor to satisfy Pipeline's non-empty layout requirement.
[[vk::binding(0, 0)]]
ConstantBuffer<float4> _dummyUBO;

[numthreads(256, 1, 1)]
[shader("compute")]
void main(uint3 dtid : SV_DispatchThreadID)
{
    uint bIdx = dtid.x;
    if (bIdx >= pc.batchSize)
        return;

    float* aux  = (float*)pc.auxAddr;
    float* pred = (float*)pc.rhsPredAddr;
    float* gt   = (float*)pc.gtAddr;

    // Average (nee + indirect) across S sub-samples.
    float3 sum = float3(0.0);
    uint S = pc.rhsSamples;
    for (uint s = 0; s < S; s++)
    {
        uint slot = bIdx * S + s;
        uint ab = slot * 12;
        float3 nee           = float3(aux[ab + 0], aux[ab + 1], aux[ab + 2]);
        float3 throughputNet = float3(aux[ab + 3], aux[ab + 4], aux[ab + 5]);
        float3 emissionTerm  = float3(aux[ab + 6], aux[ab + 7], aux[ab + 8]);
        float  validNet      = aux[ab + 9];
        float  validEmissive = aux[ab + 10];

        float3 indirect;
        if (validEmissive > 0.5)
        {
            indirect = emissionTerm;
        }
        else if (validNet > 0.5)
        {
            uint pb = slot * 3;
            float3 lY = max(float3(pred[pb + 0], pred[pb + 1], pred[pb + 2]), float3(0.0));
            indirect = throughputNet * lY;
        }
        else
        {
            indirect = float3(0.0);
        }

        sum += nee + indirect;
    }
    float3 target = sum / float(S);

    uint gb = bIdx * 3;
    gt[gb + 0] = target.x;
    gt[gb + 1] = target.y;
    gt[gb + 2] = target.z;
}
