// Generic MLP training kernel
// Architecture: input(IN→64) + N×hidden(64→64) + output(64→OUT)
// Currently instantiated for IN=2, OUT=3. Add more as needed.
import mlvec;
import mlp;
import adam;

static const int H = 64; // hidden width
static const int MAX_HIDDEN = 8;

// Per-layer BDA addresses stored as flat uint64_t[4]:
//   [0] weights, [1] weightsGrad, [2] biases, [3] biasesGrad

struct TrainPushConstants
{
    uint64_t layerAddressBuffer; // BDA to uint64_t[] (4 per layer)
    uint64_t inputBuffer;        // float[] flattened inputs
    uint64_t gtBuffer;           // float[] flattened ground truth
    uint64_t lossBuffer;         // BDA to Atomic<float> for loss accumulation
    uint   sampleCount;
    uint   hiddenLayerCount;     // number of 64->64 hidden layers
    uint   inputSize;            // for future: switch between instantiations
    uint   outputSize;           // for future: switch between instantiations
};

[[vk::push_constant]]
TrainPushConstants pc;

// --- Helper: load FeedForwardLayer from flat address array ---

FeedForwardLayer<In, Out> loadLayerFromAddrs<int In, int Out>(uint64_t* addrs, int layerIndex)
{
    int base = layerIndex * 4;
    FeedForwardLayer<In, Out> layer;
    layer.weights     = (NFloat*)addrs[base + 0];
    layer.weightsGrad = (Atomic<NFloat>*)addrs[base + 1];
    layer.biases      = (NFloat*)addrs[base + 2];
    layer.biasesGrad  = (Atomic<NFloat>*)addrs[base + 3];
    return layer;
}

// --- Output layer: linear (no activation) ---

struct OutputLayer_64_3
{
    NFloat* weights;
    Atomic<NFloat>* weightsGrad;
    NFloat* biases;
    Atomic<NFloat>* biasesGrad;

    // Linear forward (no activation)
    MLVec<3> eval(MLVec<H> input)
    {
        return matMulAdd<3>(input, weights, biases);
    }

    // Linear backward
    MLVec<H> backward(MLVec<H> input, MLVec<3> gradOutput)
    {
        outerProductAccumulate(gradOutput, input, weightsGrad);
        for (int i = 0; i < 3; i++)
            biasesGrad[i].add(gradOutput.data[i]);
        return matMulTransposed<H>(gradOutput, weights);
    }
}

OutputLayer_64_3 loadOutputLayerFromAddrs(uint64_t* addrs, int layerIndex)
{
    int base = layerIndex * 4;
    OutputLayer_64_3 layer;
    layer.weights     = (NFloat*)addrs[base + 0];
    layer.weightsGrad = (Atomic<NFloat>*)addrs[base + 1];
    layer.biases      = (NFloat*)addrs[base + 2];
    layer.biasesGrad  = (Atomic<NFloat>*)addrs[base + 3];
    return layer;
}

// --- Training kernel ---

[numthreads(256, 1, 1)]
[shader("compute")]
void main(uint tid : SV_DispatchThreadID)
{
    if (tid >= pc.sampleCount)
        return;

    uint64_t* layerAddrs = (uint64_t*)pc.layerAddressBuffer;
    float* inputs = (float*)pc.inputBuffer;
    float* gts = (float*)pc.gtBuffer;

    // Currently: inputSize=2, outputSize=3
    // Load input sample
    MLVec<2> x;
    x.data[0] = inputs[tid * 2 + 0];
    x.data[1] = inputs[tid * 2 + 1];

    MLVec<3> gt;
    gt.data[0] = gts[tid * 3 + 0];
    gt.data[1] = gts[tid * 3 + 1];
    gt.data[2] = gts[tid * 3 + 2];

    // --- Forward pass ---

    // Input layer: 2 -> 64
    var inLayer = loadLayerFromAddrs<2, H>(layerAddrs, 0);
    MLVec<H> h = inLayer.eval(x);

    // Hidden layers: 64 -> 64 (store inputs for backward)
    MLVec<H> hiddenInputs[MAX_HIDDEN];

    for (int i = 0; i < int(pc.hiddenLayerCount); i++)
    {
        hiddenInputs[i] = h;
        var hl = loadLayerFromAddrs<H, H>(layerAddrs, 1 + i);
        h = hl.eval(h);
    }

    // Output layer: 64 -> 3 (linear)
    MLVec<H> outputInput = h;
    var outLayer = loadOutputLayerFromAddrs(layerAddrs, 1 + int(pc.hiddenLayerCount));
    MLVec<3> pred = outLayer.eval(h);

    // --- MSE loss ---
    MLVec<3> diff;
    float mse = 0.0;
    for (int i = 0; i < 3; i++)
    {
        diff.data[i] = pred.data[i] - gt.data[i];
        mse += diff.data[i] * diff.data[i];
    }

    // Accumulate loss for readback
    Atomic<float>* lossPtr = (Atomic<float>*)pc.lossBuffer;
    lossPtr[0].add(mse);

    // MSE gradient: 2 * (pred - gt)
    MLVec<3> dLoss;
    for (int i = 0; i < 3; i++)
        dLoss.data[i] = 2.0 * diff.data[i];

    // --- Backward pass ---

    // Output layer backward
    MLVec<H> dH = outLayer.backward(outputInput, dLoss);

    // Hidden layers backward (reverse order)
    for (int i = int(pc.hiddenLayerCount) - 1; i >= 0; i--)
    {
        var hl = loadLayerFromAddrs<H, H>(layerAddrs, 1 + i);
        var pair = diffPair(hiddenInputs[i]);
        hl.evalBwd(pair, dH);
        dH = pair.d;
    }

    // Input layer backward
    var inputPair = diffPair(x);
    inLayer.evalBwd(inputPair, dH);
}
