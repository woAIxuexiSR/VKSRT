// Shared RNG module: Wang hash based PRNG
module rng;

public uint wangHash(uint seed)
{
    seed = (seed ^ 61u) ^ (seed >> 16u);
    seed *= 9u;
    seed = seed ^ (seed >> 4u);
    seed *= 0x27d4eb2du;
    seed = seed ^ (seed >> 15u);
    return seed;
}

public float randFloat(inout uint seed)
{
    seed = wangHash(seed);
    return float(seed) / 4294967296.0;
}

public float2 randFloat2(inout uint seed)
{
    return float2(randFloat(seed), randFloat(seed));
}

public float3 randFloat3(inout uint seed)
{
    return float3(randFloat(seed), randFloat(seed), randFloat(seed));
}
