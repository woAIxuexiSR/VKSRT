// Per-pixel HashGrid+MLP inference: encode (u,v) via hash grid, run MLP, write RGB.
import mlvec;
import mlp;
import hashgrid;

static const int H = 64;
static const float PI = 3.14159265358979323846;

struct InferencePushConstants
{
    uint64_t layerAddressBuffer;
    uint64_t hashTable;
    uint width;
    uint height;
    uint hiddenLayerCount;
    uint showGT;
    uint inputSize;              // MLP inputSize = encodedDim
    uint outputSize;
    uint rawInputDim;
    uint numLevels;
    uint featuresPerLevel;
    uint tableSize;
    float coarsestResolution;
    float perLevelScale;
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
        NFloat* table   = (NFloat*)pc.hashTable;

        HashGridParams hp;
        hp.numLevels = int(pc.numLevels);
        hp.featuresPerLevel = int(pc.featuresPerLevel);
        hp.tableSize = int(pc.tableSize);
        hp.inputDim = int(pc.rawInputDim);
        hp.coarsestResolution = pc.coarsestResolution;
        hp.perLevelScale = pc.perLevelScale;

        MLVec coord = MLVec.zero();
        if (pc.rawInputDim >= 1) coord.data[0] = u;
        if (pc.rawInputDim >= 2) coord.data[1] = v;

        MLVec encoded = MLVec.zero();
        encodeForward(coord, table, hp, encoded);

        var inLayer = loadLayer(addrs, 0);
        MLVec h = inLayer.eval(encoded, int(pc.inputSize), H);

        for (uint i = 0; i < pc.hiddenLayerCount; i++)
        {
            var hl = loadLayer(addrs, 1 + int(i));
            h = hl.eval(h, H, H);
        }

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
