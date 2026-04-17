// Per-pixel inference kernel: evaluate trained network at (u,v) and write RGB to output image
import mlvec;
import mlp;

static const int H = 64;
static const float PI = 3.14159265358979323846;

// Layer addresses stored as flat uint64_t[4] per layer:
//   [0] weights, [1] weightsGrad, [2] biases, [3] biasesGrad

struct InferencePushConstants
{
    uint64_t layerAddressBuffer;
    uint   width;
    uint   height;
    uint   hiddenLayerCount;
    uint   showGT;
};

[[vk::push_constant]]
InferencePushConstants pc;

[[vk::binding(0, 0)]]
RWTexture2D<float4> outputImage;

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

struct OutputLayer_64_3
{
    NFloat* weights;
    NFloat* biases;

    MLVec<3> eval(MLVec<H> input)
    {
        return matMulAdd<3>(input, weights, biases);
    }
}

OutputLayer_64_3 loadOutputLayerInfer(uint64_t* addrs, int layerIndex)
{
    int base = layerIndex * 4;
    OutputLayer_64_3 layer;
    layer.weights = (NFloat*)addrs[base + 0];
    layer.biases  = (NFloat*)addrs[base + 2];
    return layer;
}

[numthreads(16, 16, 1)]
[shader("compute")]
void main(uint3 dtid : SV_DispatchThreadID)
{
    if (dtid.x >= pc.width || dtid.y >= pc.height)
        return;

    float u = (float(dtid.x) + 0.5) / float(pc.width);
    float v = (float(dtid.y) + 0.5) / float(pc.height);

    float3 color;

    if (pc.showGT != 0)
    {
        color = float3(sin(PI * u), cos(PI * v), u * v);
    }
    else
    {
        uint64_t* layerAddrs = (uint64_t*)pc.layerAddressBuffer;

        // Input: (u,v) -> 64
        MLVec<2> x;
        x.data[0] = u;
        x.data[1] = v;

        var inLayer = loadLayerFromAddrs<2, H>(layerAddrs, 0);
        MLVec<H> h = inLayer.eval(x);

        // Hidden layers: 64 -> 64
        for (int i = 0; i < int(pc.hiddenLayerCount); i++)
        {
            var hl = loadLayerFromAddrs<H, H>(layerAddrs, 1 + i);
            h = hl.eval(h);
        }

        // Output: 64 -> 3 (linear)
        var outLayer = loadOutputLayerInfer(layerAddrs, 1 + int(pc.hiddenLayerCount));
        MLVec<3> result = outLayer.eval(h);

        color = float3(result.data[0], result.data[1], result.data[2]);
    }

    // Clamp to [0,1] for display
    color = clamp(color, float3(0.0), float3(1.0));
    outputImage[int2(dtid.x, dtid.y)] = float4(color, 1.0);
}
