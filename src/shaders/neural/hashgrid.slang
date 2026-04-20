// Multi-resolution hash grid encoding (NRC / Instant-NGP style).
// Hand-written forward + backward (no Slang autodiff).
//
// IMPORTANT: [ForceUnroll] loops must NOT use early break/continue when the
// loop body touches inout MLVec or BDA pointers — Slang generates an invalid
// ConvertUToPtr(0) → Function-storage pointer on the "break" branch.
// Use an if-guard around the body instead.
//
// Table layout: float[numLevels * tableSize * featuresPerLevel], level-major.
module hashgrid;

import mlvec;

public static const int MAX_LEVELS = 8;
public static const int MAX_INPUT_DIM = 3;
public static const int MAX_FEATURES_PER_LEVEL = 4;

static const uint PRIMES[3] = { 1u, 2654435761u, 805459861u };

public struct HashGridParams
{
    public int numLevels;
    public int featuresPerLevel;
    public int tableSize;
    public int inputDim;
    public float coarsestResolution;
    public float perLevelScale;
}

public static const int MAX_CORNERS = 8;

uint hashCorner(int c0, int c1, int c2, int dim, uint tableMask)
{
    uint h = uint(c0) * PRIMES[0];
    if (dim > 1) h ^= uint(c1) * PRIMES[1];
    if (dim > 2) h ^= uint(c2) * PRIMES[2];
    return h & tableMask;
}

public void encodeForward(MLVec coord, NFloat* table, HashGridParams p,
                          inout MLVec result)
{
    uint tableMask = uint(p.tableSize - 1);

    NFloat c0 = p.inputDim > 0 ? coord.data[0] : NFloat(0.0);
    NFloat c1 = p.inputDim > 1 ? coord.data[1] : NFloat(0.0);
    NFloat c2 = p.inputDim > 2 ? coord.data[2] : NFloat(0.0);

    float levelScale = p.coarsestResolution;

    [ForceUnroll]
    for (int l = 0; l < MAX_LEVELS; l++)
    {
        if (l < p.numLevels)
        {
            float res = levelScale;
            float s0 = float(c0) * res;
            float s1 = float(c1) * res;
            float s2 = float(c2) * res;

            int lo0 = int(floor(s0)); float f0 = s0 - float(lo0);
            int lo1 = int(floor(s1)); float f1 = s1 - float(lo1);
            int lo2 = int(floor(s2)); float f2 = s2 - float(lo2);

            uint levelBase = uint(l) * uint(p.tableSize) * uint(p.featuresPerLevel);

            int numCorners = 1 << p.inputDim;
            for (int c = 0; c < numCorners; c++)
            {
                int b0 = (c >> 0) & 1;
                int b1 = (c >> 1) & 1;
                int b2 = (c >> 2) & 1;

                float w = (b0 == 1) ? f0 : (1.0 - f0);
                if (p.inputDim > 1) w *= (b1 == 1) ? f1 : (1.0 - f1);
                if (p.inputDim > 2) w *= (b2 == 1) ? f2 : (1.0 - f2);

                uint slot = hashCorner(lo0 + b0, lo1 + b1, lo2 + b2, p.inputDim, tableMask);
                uint addr = levelBase + slot * uint(p.featuresPerLevel);

                int outBase = l * p.featuresPerLevel;
                for (int ff = 0; ff < p.featuresPerLevel; ff++)
                    result.data[outBase + ff] += NFloat(w) * table[addr + uint(ff)];
            }
        }
        levelScale *= p.perLevelScale;
    }
}

public void encodeBackward(MLVec gradOutput, MLVec coord,
                           Atomic<NFloat>* tableGrad, HashGridParams p)
{
    uint tableMask = uint(p.tableSize - 1);

    NFloat c0 = p.inputDim > 0 ? coord.data[0] : NFloat(0.0);
    NFloat c1 = p.inputDim > 1 ? coord.data[1] : NFloat(0.0);
    NFloat c2 = p.inputDim > 2 ? coord.data[2] : NFloat(0.0);

    float levelScale = p.coarsestResolution;

    [ForceUnroll]
    for (int l = 0; l < MAX_LEVELS; l++)
    {
        if (l < p.numLevels)
        {
            float res = levelScale;
            float s0 = float(c0) * res;
            float s1 = float(c1) * res;
            float s2 = float(c2) * res;

            int lo0 = int(floor(s0)); float f0 = s0 - float(lo0);
            int lo1 = int(floor(s1)); float f1 = s1 - float(lo1);
            int lo2 = int(floor(s2)); float f2 = s2 - float(lo2);

            uint levelBase = uint(l) * uint(p.tableSize) * uint(p.featuresPerLevel);

            int outBase = l * p.featuresPerLevel;

            int numCorners = 1 << p.inputDim;
            for (int c = 0; c < numCorners; c++)
            {
                int b0 = (c >> 0) & 1;
                int b1 = (c >> 1) & 1;
                int b2 = (c >> 2) & 1;

                float w = (b0 == 1) ? f0 : (1.0 - f0);
                if (p.inputDim > 1) w *= (b1 == 1) ? f1 : (1.0 - f1);
                if (p.inputDim > 2) w *= (b2 == 1) ? f2 : (1.0 - f2);

                uint slot = hashCorner(lo0 + b0, lo1 + b1, lo2 + b2, p.inputDim, tableMask);
                uint addr = levelBase + slot * uint(p.featuresPerLevel);
                NFloat nw = NFloat(w);

                for (int ff = 0; ff < p.featuresPerLevel; ff++)
                    tableGrad[addr + uint(ff)].add(gradOutput.data[outBase + ff] * nw);
            }
        }
        levelScale *= p.perLevelScale;
    }
}
