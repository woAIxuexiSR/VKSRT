// MLP layer with manually-written backward derivative
// Uses BDA pointers for weight/gradient buffer access
module mlp;

import mlvec;

public struct FeedForwardLayer<int InputSize, int OutputSize>
{
    public NFloat* weights;
    public Atomic<NFloat>* weightsGrad;
    public NFloat* biases;
    public Atomic<NFloat>* biasesGrad;

    [BackwardDerivative(evalBwd)]
    public MLVec<OutputSize> eval(MLVec<InputSize> input)
    {
        var output = matMulAdd<OutputSize>(input, weights, biases);
        // LeakyReLU activation
        for (int i = 0; i < OutputSize; i++)
            if (output.data[i] < 0.0)
                output.data[i] *= 0.01;
        return output;
    }

    public void evalBwd(
        inout DifferentialPair<MLVec<InputSize>> input,
        MLVec<OutputSize> resultGrad)
    {
        let fwd = eval(input.p);

        // Back-prop through activation
        for (int i = 0; i < OutputSize; i++)
            if (fwd.data[i] < 0.0)
                resultGrad.data[i] *= 0.01;

        // dL/dWeights += resultGrad * input^T
        outerProductAccumulate(resultGrad, input.p, weightsGrad);

        // dL/dBiases += resultGrad
        for (int i = 0; i < OutputSize; i++)
            biasesGrad[i].add(resultGrad.data[i]);

        // dL/dInput = weights^T * resultGrad
        let dInput = matMulTransposed<InputSize>(resultGrad, weights);
        input = {input.p, dInput};
    }
}
