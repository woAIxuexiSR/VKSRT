// Shared material definition module
// Layout must match C++ Material struct in src/scene/material.h (64 bytes)
module material;

public static const int MAT_LAMBERTIAN = 0;
public static const int MAT_METAL      = 1;
public static const int MAT_DIELECTRIC = 2;
public static const int MAT_EMISSIVE   = 3;

public struct Material
{
    public float4 baseColor;       // rgb = albedo, a = alpha
    public float4 emission;        // rgb = emission color, a = emission strength
    public float metallic;
    public float roughness;        // 0.0 = mirror, 1.0 = diffuse
    public float ior;              // index of refraction (glass: 1.5)
    public int type;               // MAT_LAMBERTIAN / MAT_METAL / MAT_DIELECTRIC / MAT_EMISSIVE
    public int diffuseTexIndex;    // -1 = no texture
    public int _pad0;
    public int _pad1;
    public int _pad2;

    public bool canTransmit() { return type == MAT_DIELECTRIC; }
};
