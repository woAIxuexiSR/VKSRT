// HashGrid backward kernel: dEncoded + raw input → atomic splat to table grad.
// Each thread processes one sample.
import mlvec;
import hashgrid;

struct HashGridBackwardPushConstants
{
    uint64_t dEncodedBuffer;     // float[sampleCount * encodedDim] (gradient from MLP)
    uint64_t inputBuffer;        // float[sampleCount * inputDim] (raw coords, for rehashing)
    uint64_t hashTableGrad;      // Atomic<NFloat>[numLevels * tableSize * featuresPerLevel]
    uint sampleCount;
    uint inputDim;
    uint numLevels;
    uint featuresPerLevel;
    uint tableSize;
    float coarsestResolution;
    float perLevelScale;
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

    // Load raw coord
    MLVec coord = MLVec.zero();
    for (uint j = 0; j < pc.inputDim; j++)
        coord.data[j] = NFloat(inputs[tid * pc.inputDim + j]);

    // Load dEncoded
    uint encodedDim = pc.numLevels * pc.featuresPerLevel;
    MLVec gradOutput = MLVec.zero();
    for (uint j = 0; j < encodedDim; j++)
        gradOutput.data[j] = NFloat(dEncoded[tid * encodedDim + j]);

    // Splat gradients
    encodeBackward(gradOutput, coord, tableGrad, hp);
}
