// Feed-forward layer with runtime sizes and manual forward/backward.
// Activation: LeakyReLU(0.01). Output layer should use evalLinear/backwardLinear.
module mlp;

import mlvec;

public struct FeedForwardLayer
{
    public NFloat* weights;
    public Atomic<NFloat>* weightsGrad;
    public NFloat* biases;
    public Atomic<NFloat>* biasesGrad;

    // Forward with LeakyReLU
    public MLVec eval(MLVec input, int inSize, int outSize)
    {
        MLVec output = matMulAdd(input, weights, biases, inSize, outSize);
        for (int i = 0; i < outSize; i++)
            if (output.data[i] < NFloat(0.0))
                output.data[i] *= NFloat(0.01);
        return output;
    }

    // Linear forward (output layer)
    public MLVec evalLinear(MLVec input, int inSize, int outSize)
    {
        return matMulAdd(input, weights, biases, inSize, outSize);
    }

    // Backward through LeakyReLU + linear.
    //   input: cached forward input to this layer
    //   postAct: cached post-activation (layer output); for LeakyReLU the sign
    //            is identical to pre-activation so it is fine as the mask source
    //   gradOutput: dL/dOutput (post-activation); mutated in place to dL/dPreAct
    // returns dL/dInput
    public MLVec backward(MLVec input, MLVec postAct, MLVec gradOutput,
                           int inSize, int outSize)
    {
        for (int i = 0; i < outSize; i++)
            if (postAct.data[i] < NFloat(0.0))
                gradOutput.data[i] *= NFloat(0.01);

        outerProductAccumulate(gradOutput, input, weightsGrad, outSize, inSize);
        for (int i = 0; i < outSize; i++)
            biasesGrad[i].add(gradOutput.data[i]);

        return matMulTransposed(gradOutput, weights, inSize, outSize);
    }

    // Linear backward (output layer, no activation)
    public MLVec backwardLinear(MLVec input, MLVec gradOutput, int inSize, int outSize)
    {
        outerProductAccumulate(gradOutput, input, weightsGrad, outSize, inSize);
        for (int i = 0; i < outSize; i++)
            biasesGrad[i].add(gradOutput.data[i]);
        return matMulTransposed(gradOutput, weights, inSize, outSize);
    }
}
