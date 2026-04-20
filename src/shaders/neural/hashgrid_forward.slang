import mlvec;
import hashgrid;

struct HashGridForwardPushConstants
{
    uint64_t inputBuffer;
    uint64_t outputBuffer;
    uint64_t hashTable;
    uint sampleCount;
    uint inputDim;
    uint numLevels;
    uint featuresPerLevel;
    uint tableSize;
    float coarsestResolution;
    float perLevelScale;
    uint inputFieldOffset;
    uint inputStride;
    uint outputFieldOffset;
    uint outputStride;
};

[[vk::push_constant]]
HashGridForwardPushConstants pc;

[numthreads(256, 1, 1)]
[shader("compute")]
void main(uint tid : SV_DispatchThreadID)
{
    if (tid >= pc.sampleCount)
        return;

    float* inputs  = (float*)pc.inputBuffer;
    float* outputs = (float*)pc.outputBuffer;
    NFloat* table  = (NFloat*)pc.hashTable;

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

    MLVec encoded = MLVec.zero();
    encodeForward(coord, table, hp, encoded);

    uint encodedDim = pc.numLevels * pc.featuresPerLevel;
    for (uint j = 0; j < encodedDim; j++)
        outputs[tid * pc.outputStride + pc.outputFieldOffset + j] = float(encoded.data[j]);
}
