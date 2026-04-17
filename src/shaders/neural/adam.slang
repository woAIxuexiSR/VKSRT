// Adam optimizer for neural network training
module adam;

import mlp;

public struct AdamState
{
    public float mean;
    public float variance;
    public int iteration;
}

public struct AdamOptimizer
{
    public static const float beta1 = 0.9;
    public static const float beta2 = 0.999;
    public static const float epsilon = 1e-7;
    public static const float learningRate = 0.01;

    public static void step(inout AdamState state, inout float param, inout float grad)
    {
        state.iteration++;

        if (isinf(grad))
            grad = grad > 0 ? 10000.0 : -10000.0;

        state.mean = beta1 * state.mean + (1.0 - beta1) * grad;
        state.variance = beta2 * state.variance + (1.0 - beta2) * grad * grad;

        float meanHat = state.mean / (1.0 - pow(beta1, float(state.iteration)));
        float varianceHat = state.variance / (1.0 - pow(beta2, float(state.iteration)));

        param -= learningRate * meanHat / (sqrt(max(0.0, varianceHat) + epsilon));
        grad = 0.0;
    }
}
