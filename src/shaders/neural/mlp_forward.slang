// MLP forward kernel: inputSize -> 64 -> ... -> 64 -> outputSize
// Writes output to outputBuffer and all layer activations to activationsBuffer.
// Activations layout: acts[tid * actStride + k * H], k=0..numLayers
//   k=0: network input, k=1..hiddenLayerCount: post-activation, k=outIdx: output layer input
import mlvec;
import mlp;

static const int H = 64;

struct MlpForwardPushConstants
{
    uint64_t layerAddressBuffer;
    uint64_t inputBuffer;        // float[sampleCount * inputSize]
    uint64_t outputBuffer;       // float[sampleCount * outputSize]
    uint64_t activationsBuffer;  // float[sampleCount * actStride]
    uint sampleCount;
    uint hiddenLayerCount;
    uint inputSize;
    uint outputSize;
    uint actStride;              // (hiddenLayerCount + 2) * H
};

[[vk::push_constant]]
MlpForwardPushConstants pc;

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

void storeAct(float* acts, uint tid, uint slot, MLVec v, int size)
{
    uint base = tid * pc.actStride + slot * uint(H);
    for (int j = 0; j < size; j++)
        acts[base + uint(j)] = float(v.data[j]);
}

[numthreads(256, 1, 1)]
[shader("compute")]
void main(uint tid : SV_DispatchThreadID)
{
    if (tid >= pc.sampleCount)
        return;

    uint64_t* addrs = (uint64_t*)pc.layerAddressBuffer;
    float* inputs   = (float*)pc.inputBuffer;
    float* outputs  = (float*)pc.outputBuffer;
    float* acts     = (float*)pc.activationsBuffer;

    // Load input
    MLVec x = MLVec.zero();
    for (uint j = 0; j < pc.inputSize; j++)
        x.data[j] = inputs[tid * pc.inputSize + j];

    // Store acts[0] = input
    storeAct(acts, tid, 0, x, int(pc.inputSize));

    // Input layer (inputSize -> H) with LeakyReLU
    var inLayer = loadLayer(addrs, 0);
    MLVec h = inLayer.eval(x, int(pc.inputSize), H);
    storeAct(acts, tid, 1, h, H);

    // Hidden layers (H -> H)
    for (uint i = 0; i < pc.hiddenLayerCount; i++)
    {
        var hl = loadLayer(addrs, 1 + int(i));
        h = hl.eval(h, H, H);
        storeAct(acts, tid, 2 + i, h, H);
    }

    // Output layer (H -> outputSize) linear
    uint outIdx = 1 + pc.hiddenLayerCount;
    var outLayer = loadLayer(addrs, int(outIdx));
    MLVec pred = outLayer.evalLinear(h, H, int(pc.outputSize));

    // Write output
    for (uint j = 0; j < pc.outputSize; j++)
        outputs[tid * pc.outputSize + j] = float(pred.data[j]);
}
