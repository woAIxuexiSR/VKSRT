import mlvec;
import hashgrid;

struct HashGridBackwardPushConstants
{
    uint64_t dEncodedBuffer;
    uint64_t inputBuffer;
    uint64_t hashTableGrad;
    uint sampleCount;
    uint inputDim;
    uint numLevels;
    uint featuresPerLevel;
    uint tableSize;
    float coarsestResolution;
    float perLevelScale;
    uint dEncodedFieldOffset;
    uint dEncodedStride;
    uint inputFieldOffset;
    uint inputStride;
};

[[vk::push_constant]]
HashGridBackwardPushConstants pc;

[numthreads(256, 1, 1)]
[shader("compute")]
void main(uint tid : SV_DispatchThreadID)
{
    if (tid >= pc.sampleCount)
        return;

    float* dEncoded   = (float*)pc.dEncodedBuffer;
    float* inputs     = (float*)pc.inputBuffer;
    Atomic<NFloat>* tableGrad = (Atomic<NFloat>*)pc.hashTableGrad;

    HashGridParams hp;
    hp.numLevels = int(pc.numLevels);
    hp.featuresPerLevel = int(pc.featuresPerLevel);
    hp.tableSize = int(pc.tableSize);
    hp.inputDim = int(pc.inputDim);
    hp.coarsestResolution = pc.coarsestResolution;
    hp.perLevelScale = pc.perLevelScale;

    MLVec coord = MLVec.zero();
    for (uint j = 0; j < pc.inputDim; j++)
        coord.data[j] = NFloat(inputs[tid * pc.inputStride + pc.inputFieldOffset + j]);

    uint encodedDim = pc.numLevels * pc.featuresPerLevel;
    MLVec gradOutput = MLVec.zero();
    for (uint j = 0; j < encodedDim; j++)
        gradOutput.data[j] = NFloat(dEncoded[tid * pc.dEncodedStride + pc.dEncodedFieldOffset + j]);

    encodeBackward(gradOutput, coord, tableGrad, hp);
}
