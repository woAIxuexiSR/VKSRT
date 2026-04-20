// Generic MLP training kernel: inputSize -> 64 -> ... -> 64 -> outputSize
// All sizes are runtime values from push constants; hidden width is fixed at 64.
import mlvec;
import mlp;

static const int H = 64;           // hidden width (fixed)
static const int MAX_HIDDEN = 8;   // cap on hidden layers

struct TrainPushConstants
{
    uint64_t layerAddressBuffer; // BDA to uint64_t[4 * numLayers]
    uint64_t inputBuffer;        // float[sampleCount * inputSize]
    uint64_t gtBuffer;           // float[sampleCount * outputSize]
    uint64_t lossBuffer;         // Atomic<float>
    uint sampleCount;
    uint hiddenLayerCount;       // number of 64->64 hidden layers
    uint inputSize;              // <= 64
    uint outputSize;             // <= 64
};

[[vk::push_constant]]
TrainPushConstants pc;

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

[numthreads(256, 1, 1)]
[shader("compute")]
void main(uint tid : SV_DispatchThreadID)
{
    if (tid >= pc.sampleCount)
        return;

    uint64_t* addrs = (uint64_t*)pc.layerAddressBuffer;
    float* inputs   = (float*)pc.inputBuffer;
    float* gts      = (float*)pc.gtBuffer;

    // --- Load sample ---
    MLVec x = MLVec.zero();
    for (uint j = 0; j < pc.inputSize; j++)
        x.data[j] = inputs[tid * pc.inputSize + j];

    MLVec gt = MLVec.zero();
    for (uint j = 0; j < pc.outputSize; j++)
        gt.data[j] = gts[tid * pc.outputSize + j];

    // --- Forward ---
    // Layer index:   0 = input layer (inputSize -> 64)
    //                1..hiddenLayerCount = hidden (64 -> 64)
    //                1 + hiddenLayerCount = output layer (64 -> outputSize)
    //
    // acts[0]       = network input x
    // acts[1..k]    = post-activation of layer 0..k-1  (k = 1 + hiddenLayerCount)
    // acts[k]       = input to output layer
    MLVec acts[MAX_HIDDEN + 2];

    acts[0] = x;

    // Input layer (inputSize -> 64) with LeakyReLU
    var inLayer = loadLayer(addrs, 0);
    acts[1] = inLayer.eval(x, int(pc.inputSize), H);

    // Hidden layers (64 -> 64)
    for (uint i = 0; i < pc.hiddenLayerCount; i++)
    {
        var hl = loadLayer(addrs, 1 + int(i));
        acts[2 + i] = hl.eval(acts[1 + i], H, H);
    }

    // Output layer (64 -> outputSize) linear
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

    // --- Backward ---
    //   input to layer L = acts[L], post-activation of layer L = acts[L+1]
    MLVec dH = outLayer.backwardLinear(acts[outIdx], dLoss, H, int(pc.outputSize));

    for (int i = int(pc.hiddenLayerCount) - 1; i >= 0; i--)
    {
        var hl = loadLayer(addrs, 1 + i);
        dH = hl.backward(acts[1 + i], acts[2 + i], dH, H, H);
    }

    inLayer.backward(acts[0], acts[1], dH, int(pc.inputSize), H);
}
