// Shared camera helpers for ray-gen style passes that need to spawn primary rays.
module camera;

import rng;

// Camera uniform laid out to match Camera::getViewInverse() / getProjInverse() on the C++ side.
// Bindings are per-pass; the struct itself is shared so the 5+ copies in RT/compute shaders
// don't drift out of sync.
public struct CameraProperties
{
    public float4x4 viewInverse;
    public float4x4 projInverse;
};

// Per-pixel RNG seed. Keep the magic constants here so all passes share the same sequence
// spacing and we only have one place to retune if needed.
public uint seedFromPixel(uint2 pixel, uint frameIndex)
{
    return wangHash(uint(pixel.x * 1973u + pixel.y * 9277u + frameIndex * 26699u) | 1u);
}

// Generate a primary ray through (pixel + jitter) on a [launchSize] screen.
// Caller feeds in the RNG seed so they stay in control of its lifetime and side effects.
public void generatePrimaryRay(
    CameraProperties cam,
    uint2 pixel,
    uint2 launchSize,
    inout uint seed,
    out float3 rayOrigin,
    out float3 rayDir)
{
    float2 jitter = randFloat2(seed);
    float2 pixelCenter = float2(pixel) + jitter;
    float2 inUV = pixelCenter / float2(launchSize);
    float2 d = inUV * 2.0 - 1.0;

    float4 origin = mul(cam.viewInverse, float4(0, 0, 0, 1));
    float4 target = mul(cam.projInverse, float4(d.x, d.y, 1, 1));
    float4 direction = mul(cam.viewInverse, float4(normalize(target.xyz), 0));

    rayOrigin = origin.xyz;
    rayDir = normalize(direction.xyz);
}
