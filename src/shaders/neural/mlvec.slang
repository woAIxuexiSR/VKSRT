// Fixed-capacity vector (MAX_WIDTH=64) with runtime-sized matmul helpers.
// All weight buffers are row-major: W[i,j] stored at weights[i * inSize + j]
// where i ∈ [0, outSize), j ∈ [0, inSize).
module mlvec;

public typealias NFloat = float;

public static const int MAX_WIDTH = 64;

public struct MLVec
{
    public NFloat data[MAX_WIDTH];

    public static MLVec zero()
    {
        MLVec result;
        [ForceUnroll]
        for (int i = 0; i < MAX_WIDTH; i++)
            result.data[i] = NFloat(0.0);
        return result;
    }
}

// Forward:  output[i] = bias[i] + sum_j weights[i*inSize+j] * input.data[j],  i ∈ [0, outSize)
public MLVec matMulAdd(MLVec input, NFloat* weights, NFloat* bias, int inSize, int outSize)
{
    MLVec result = MLVec.zero();
    for (int i = 0; i < outSize; i++)
    {
        NFloat r = bias[i];
        for (int j = 0; j < inSize; j++)
            r += weights[i * inSize + j] * input.data[j];
        result.data[i] = r;
    }
    return result;
}

// Backward to input:  dInput[j] = sum_i weights[i*inSize+j] * grad.data[i],  j ∈ [0, inSize)
public MLVec matMulTransposed(MLVec grad, NFloat* weights, int inSize, int outSize)
{
    MLVec result = MLVec.zero();
    for (int j = 0; j < inSize; j++)
    {
        NFloat r = NFloat(0.0);
        for (int i = 0; i < outSize; i++)
            r += weights[i * inSize + j] * grad.data[i];
        result.data[j] = r;
    }
    return result;
}

// Backward to weights:  W_grad[i,j] += grad[i] * input[j]  (atomic accumulate)
// Outer loop is ForceUnroll'd so grad.data[i] becomes a compile-time index.
// Intel Arc + slang miscompiles the kernel if grad is indexed by a runtime
// loop variable inside an atomic-store loop (hangs the GPU).
public void outerProductAccumulate(MLVec grad, MLVec input, Atomic<NFloat>* matrix,
                                    int outSize, int inSize)
{
    [ForceUnroll]
    for (int i = 0; i < MAX_WIDTH; i++)
    {
        if (i >= outSize) break;
        NFloat gi = grad.data[i];
        int rowBase = i * inSize;
        for (int j = 0; j < inSize; j++)
            matrix[rowBase + j].add(gi * input.data[j]);
    }
}
