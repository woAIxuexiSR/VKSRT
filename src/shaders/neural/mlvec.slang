// Differentiable fixed-size vector and linear algebra helpers for MLP
// Compile-time sized via generics; used by FeedForwardLayer<In,Out>
module mlvec;

public typealias NFloat = float;

public struct MLVec<int N> : IDifferentiable
{
    public NFloat data[N];

    [Differentiable]
    public NFloat[N] toArray()
    {
        return data;
    }

    [Differentiable]
    public static MLVec<N> fromArray(NFloat[N] values)
    {
        MLVec<N> result;
        [ForceUnroll]
        for (int i = 0; i < N; i++)
            result.data[i] = values[i];
        return result;
    }

    // Zero-initialize
    public static MLVec<N> zero()
    {
        MLVec<N> result;
        [ForceUnroll]
        for (int i = 0; i < N; i++)
            result.data[i] = NFloat(0.0);
        return result;
    }
}

// Forward: output = weights * input + bias (row-major)
public MLVec<OutputSize> matMulAdd<int OutputSize, int InputSize>(
    MLVec<InputSize> input, NFloat* weights, NFloat* bias)
{
    MLVec<OutputSize> result = {};
    for (int i = 0; i < OutputSize; i++)
    {
        NFloat r = bias[i];
        for (int j = 0; j < InputSize; j++)
            r += weights[i * InputSize + j] * input.data[j];
        result.data[i] = r;
    }
    return result;
}

// Backward: dL/dInput = weights^T * grad (transposed read)
public MLVec<OutputSize> matMulTransposed<int OutputSize, int InputSize>(
    MLVec<InputSize> input, NFloat* weights)
{
    MLVec<OutputSize> result = {};
    for (int i = 0; i < OutputSize; i++)
    {
        NFloat r = NFloat(0.0);
        for (int j = 0; j < InputSize; j++)
            r += weights[j * OutputSize + i] * input.data[j];
        result.data[i] = r;
    }
    return result;
}

// Backward: dL/dWeights += grad * input^T (atomic accumulate)
public void outerProductAccumulate<int M, int N>(
    MLVec<M> v0, MLVec<N> v1, Atomic<NFloat>* matrix)
{
    for (int i = 0; i < M; i++)
        for (int j = 0; j < N; j++)
            matrix[i * N + j].add(v0.data[i] * v1.data[j]);
}
