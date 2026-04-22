// Adam parameter update compute kernel
// Dispatched with one thread per parameter
import mlp;
import adam;

struct AdamPushConstants
{
    uint64_t adamStates;
    uint64_t params;
    uint64_t gradients;
    uint count;
    float learningRate;
};

[[vk::push_constant]]
AdamPushConstants pc;

// AdamState layout: float mean (0), float variance (4), int iteration (8) = 12 bytes
static const int ADAM_STATE_STRIDE = 3; // in units of float (12 bytes)

[numthreads(256, 1, 1)]
[shader("compute")]
void main(uint tid : SV_DispatchThreadID)
{
    if (tid >= pc.count)
        return;

    // Access adam states as flat float*/int* arrays (avoid struct BDA pointer)
    float* stateFloats = (float*)pc.adamStates;
    int*   stateInts   = (int*)pc.adamStates;
    float* p = (float*)pc.params;
    float* g = (float*)pc.gradients;

    if (isnan(g[tid]))
    {
        g[tid] = 0.0;
        return;
    }

    // Read AdamState fields manually
    int base = int(tid) * ADAM_STATE_STRIDE;
    float mean     = stateFloats[base + 0];
    float variance = stateFloats[base + 1];
    int   iteration = stateInts[base + 2];

    // Adam step (inline to avoid struct BDA)
    iteration++;
    float grad = g[tid];

    if (isinf(grad))
        grad = grad > 0 ? 10000.0 : -10000.0;

    mean = 0.9 * mean + 0.1 * grad;
    variance = 0.999 * variance + 0.001 * grad * grad;

    float meanHat = mean / (1.0 - pow(0.9, float(iteration)));
    float varianceHat = variance / (1.0 - pow(0.999, float(iteration)));

    p[tid] -= pc.learningRate * meanHat / (sqrt(max(0.0, varianceHat) + 1e-7));
    g[tid] = 0.0;

    // Write back
    stateFloats[base + 0] = mean;
    stateFloats[base + 1] = variance;
    stateInts[base + 2] = iteration;
}
