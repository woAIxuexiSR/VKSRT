// Per-pixel MLP inference: evaluate network at (u,v) and write RGB to output image.
// Network shape (inputSize/outputSize/hiddenLayerCount) comes from push constants.
// Currently wired to interpret:
//   inputSize  == 2 -> feed (u, v)
//   outputSize <= 3 -> write to R,G,B (unused channels = 0)
import mlvec;
import mlp;

static const int H = 64;
static const float PI = 3.14159265358979323846;

struct InferencePushConstants
{
    uint64_t layerAddressBuffer;
    uint width;
    uint height;
    uint hiddenLayerCount;
    uint showGT;
    uint inputSize;
    uint outputSize;
};

[[vk::push_constant]]
InferencePushConstants pc;

[[vk::binding(0, 0)]]
RWTexture2D<float4> outputImage;

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
        uint64_t* addrs = (uint64_t*)pc.layerAddressBuffer;

        // Build input (inputSize values). For the 2D test pass we feed (u, v).
        MLVec x = MLVec.zero();
        if (pc.inputSize >= 1) x.data[0] = u;
        if (pc.inputSize >= 2) x.data[1] = v;

        // Input layer (inputSize -> 64)
        var inLayer = loadLayer(addrs, 0);
        MLVec h = inLayer.eval(x, int(pc.inputSize), H);

        // Hidden layers (64 -> 64)
        for (uint i = 0; i < pc.hiddenLayerCount; i++)
        {
            var hl = loadLayer(addrs, 1 + int(i));
            h = hl.eval(h, H, H);
        }

        // Output layer (64 -> outputSize) linear
        var outLayer = loadLayer(addrs, 1 + int(pc.hiddenLayerCount));
        MLVec result = outLayer.evalLinear(h, H, int(pc.outputSize));

        color = float3(0.0);
        if (pc.outputSize >= 1) color.x = result.data[0];
        if (pc.outputSize >= 2) color.y = result.data[1];
        if (pc.outputSize >= 3) color.z = result.data[2];
    }

    color = clamp(color, float3(0.0), float3(1.0));
    outputImage[int2(dtid.x, dtid.y)] = float4(color, 1.0);
}
