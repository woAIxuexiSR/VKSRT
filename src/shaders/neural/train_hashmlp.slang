// HashGrid + MLP training kernel.
// Forward:  raw coord -> hashgrid encode -> MLP
// Backward: dLoss -> MLP backward -> hashgrid encodeBackward (atomic table grads)
import mlvec;
import mlp;
import hashgrid;

static const int H = 64;
static const int MAX_HIDDEN = 8;

struct TrainHashMlpPushConstants
{
    uint64_t layerAddressBuffer; // MLP layer BDA table
    uint64_t inputBuffer;        // float[sampleCount * rawInputDim]
    uint64_t gtBuffer;           // float[sampleCount * outputSize]
    uint64_t lossBuffer;         // Atomic<float>
    uint64_t hashTable;          // NFloat*
    uint64_t hashTableGrad;      // Atomic<NFloat>*
    uint sampleCount;
    uint hiddenLayerCount;
    uint inputSize;              // == encoded dim (MLP input)
    uint outputSize;
    uint rawInputDim;            // hashgrid input dim
    uint numLevels;
    uint featuresPerLevel;
    uint tableSize;
    float coarsestResolution;
    float perLevelScale;
};

[[vk::push_constant]]
TrainHashMlpPushConstants pc;

FeedForwardLayer loadLayer(uint64_t* addrs, int layerIndex)
{
    int base = layerIndex * 4;
    FeedForwardLayer layer;
    layer.weights     = (NFloat*)addrs[base + 0];
    layer.weightsGrad = (Atomic<NFloat>*)addrs[base + 1];
    layer.biases      = (NFloat*)addrs[base + 2];
    layer.biasesGrad  = (Atomic<NFloat>*)addrs[base + 3];
    return layer;
}

HashGridParams loadHashParams()
{
    HashGridParams p;
    p.numLevels = int(pc.numLevels);
    p.featuresPerLevel = int(pc.featuresPerLevel);
    p.tableSize = int(pc.tableSize);
    p.inputDim = int(pc.rawInputDim);
    p.coarsestResolution = pc.coarsestResolution;
    p.perLevelScale = pc.perLevelScale;
    return p;
}

[numthreads(256, 1, 1)]
[shader("compute")]
void main(uint tid : SV_DispatchThreadID)
{
    if (tid >= pc.sampleCount)
        return;

    uint64_t* addrs = (uint64_t*)pc.layerAddressBuffer;
    float* inputs   = (float*)pc.inputBuffer;
    float* gts      = (float*)pc.gtBuffer;
    NFloat* table   = (NFloat*)pc.hashTable;
    Atomic<NFloat>* tableGrad = (Atomic<NFloat>*)pc.hashTableGrad;

    HashGridParams hp = loadHashParams();

    // --- Load raw coord ---
    MLVec coord = MLVec.zero();
    for (uint j = 0; j < pc.rawInputDim; j++)
        coord.data[j] = inputs[tid * pc.rawInputDim + j];

    MLVec gt = MLVec.zero();
    for (uint j = 0; j < pc.outputSize; j++)
        gt.data[j] = gts[tid * pc.outputSize + j];

    // --- Encode ---
    MLVec encoded = MLVec.zero();
    encodeForward(coord, table, hp, encoded);

    // --- MLP forward ---
    MLVec acts[MAX_HIDDEN + 2];
    acts[0] = encoded;

    var inLayer = loadLayer(addrs, 0);
    acts[1] = inLayer.eval(encoded, int(pc.inputSize), H);

    for (uint i = 0; i < pc.hiddenLayerCount; i++)
    {
        var hl = loadLayer(addrs, 1 + int(i));
        acts[2 + i] = hl.eval(acts[1 + i], H, H);
    }

    uint outIdx = 1 + pc.hiddenLayerCount;
    var outLayer = loadLayer(addrs, int(outIdx));
    MLVec pred = outLayer.evalLinear(acts[outIdx], H, int(pc.outputSize));

    // --- MSE loss + grad ---
    MLVec dLoss = MLVec.zero();
    float mse = 0.0;
    for (uint i = 0; i < pc.outputSize; i++)
    {
        float diff = pred.data[i] - gt.data[i];
        mse += diff * diff;
        dLoss.data[i] = 2.0 * diff;
    }
    Atomic<float>* lossPtr = (Atomic<float>*)pc.lossBuffer;
    lossPtr[0].add(mse);

    // --- MLP backward ---
    MLVec dH = outLayer.backwardLinear(acts[outIdx], dLoss, H, int(pc.outputSize));

    for (int i = int(pc.hiddenLayerCount) - 1; i >= 0; i--)
    {
        var hl = loadLayer(addrs, 1 + i);
        dH = hl.backward(acts[1 + i], acts[2 + i], dH, H, H);
    }

    MLVec dEncoded = inLayer.backward(acts[0], acts[1], dH, int(pc.inputSize), H);

    // --- HashGrid backward ---
    encodeBackward(dEncoded, coord, tableGrad, hp);
}
