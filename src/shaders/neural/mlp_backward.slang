// MLP backward kernel: reads activations + dOutput, writes dInput + weight gradients.
// Also computes MSE loss from output and ground truth.
import mlvec;
import mlp;

static const int H = 64;

struct MlpBackwardPushConstants
{
    uint64_t layerAddressBuffer;
    uint64_t activationsBuffer;  // float[sampleCount * actStride] (from forward)
    uint64_t outputBuffer;       // float[sampleCount * outputSize] (forward output, for loss)
    uint64_t gtBuffer;           // float[sampleCount * outputSize]
    uint64_t dInputBuffer;       // float[sampleCount * inputSize] (gradient w.r.t. MLP input)
    uint64_t lossBuffer;         // Atomic<float>
    uint sampleCount;
    uint hiddenLayerCount;
    uint inputSize;
    uint outputSize;
    uint actStride;
};

[[vk::push_constant]]
MlpBackwardPushConstants pc;

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

MLVec loadAct(float* acts, uint tid, uint slot, int size)
{
    MLVec v = MLVec.zero();
    uint base = tid * pc.actStride + slot * uint(H);
    for (int j = 0; j < size; j++)
        v.data[j] = NFloat(acts[base + uint(j)]);
    return v;
}

[numthreads(256, 1, 1)]
[shader("compute")]
void main(uint tid : SV_DispatchThreadID)
{
    if (tid >= pc.sampleCount)
        return;

    uint64_t* addrs  = (uint64_t*)pc.layerAddressBuffer;
    float* acts      = (float*)pc.activationsBuffer;
    float* outputs   = (float*)pc.outputBuffer;
    float* gts       = (float*)pc.gtBuffer;
    float* dInputs   = (float*)pc.dInputBuffer;

    // Load prediction and ground truth
    MLVec pred = MLVec.zero();
    MLVec gt = MLVec.zero();
    for (uint j = 0; j < pc.outputSize; j++)
    {
        pred.data[j] = NFloat(outputs[tid * pc.outputSize + j]);
        gt.data[j]   = NFloat(gts[tid * pc.outputSize + j]);
    }

    // MSE loss + grad
    MLVec dLoss = MLVec.zero();
    float mse = 0.0;
    for (uint i = 0; i < pc.outputSize; i++)
    {
        float diff = float(pred.data[i]) - float(gt.data[i]);
        mse += diff * diff;
        dLoss.data[i] = NFloat(2.0 * diff);
    }
    Atomic<float>* lossPtr = (Atomic<float>*)pc.lossBuffer;
    lossPtr[0].add(mse);

    // Backward through output layer
    uint outIdx = 1 + pc.hiddenLayerCount;
    MLVec actOutIn = loadAct(acts, tid, outIdx, H);
    var outLayer = loadLayer(addrs, int(outIdx));
    MLVec dH = outLayer.backwardLinear(actOutIn, dLoss, H, int(pc.outputSize));

    // Backward through hidden layers
    for (int i = int(pc.hiddenLayerCount) - 1; i >= 0; i--)
    {
        MLVec actIn  = loadAct(acts, tid, uint(1 + i), H);
        MLVec actOut = loadAct(acts, tid, uint(2 + i), H);
        var hl = loadLayer(addrs, 1 + i);
        dH = hl.backward(actIn, actOut, dH, H, H);
    }

    // Backward through input layer
    MLVec actInput = loadAct(acts, tid, 0, int(pc.inputSize));
    MLVec actPost  = loadAct(acts, tid, 1, H);
    var inLayer = loadLayer(addrs, 0);
    MLVec dInput = inLayer.backward(actInput, actPost, dH, int(pc.inputSize), H);

    // Write dInput
    for (uint j = 0; j < pc.inputSize; j++)
        dInputs[tid * pc.inputSize + j] = float(dInput.data[j]);
}
