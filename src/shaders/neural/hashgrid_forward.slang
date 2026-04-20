// HashGrid forward kernel: raw input coords → encoded features buffer.
// Each thread processes one sample.
import mlvec;
import hashgrid;

struct HashGridForwardPushConstants
{
    uint64_t inputBuffer;        // float[sampleCount * inputDim]
    uint64_t outputBuffer;       // float[sampleCount * encodedDim]
    uint64_t hashTable;          // NFloat[numLevels * tableSize * featuresPerLevel]
    uint sampleCount;
    uint inputDim;
    uint numLevels;
    uint featuresPerLevel;
    uint tableSize;
    float coarsestResolution;
    float perLevelScale;
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

    // Load raw coord
    MLVec coord = MLVec.zero();
    for (uint j = 0; j < pc.inputDim; j++)
        coord.data[j] = NFloat(inputs[tid * pc.inputDim + j]);

    // Encode
    MLVec encoded = MLVec.zero();
    encodeForward(coord, table, hp, encoded);

    // Write encoded features
    uint encodedDim = pc.numLevels * pc.featuresPerLevel;
    for (uint j = 0; j < encodedDim; j++)
        outputs[tid * encodedDim + j] = float(encoded.data[j]);
}
