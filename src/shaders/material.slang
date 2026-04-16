// Shared material definition module
// Layout must match C++ Material struct in src/scene/material.h (64 bytes)
module material;

public static const int MAT_LAMBERTIAN = 0;
public static const int MAT_METAL      = 1;
public static const int MAT_DIELECTRIC = 2;
public static const int MAT_EMISSIVE   = 3;

public static const float DELTA_ROUGHNESS_THRESHOLD = 0.001;

public static const int STYLE_NONE       = 0;
public static const int STYLE_TOON       = 1;
public static const int STYLE_EIGHT_BIT  = 2;
public static const int STYLE_SATURATION = 3;
public static const int STYLE_HALFTONE   = 4;
public static const int STYLE_PALETTE    = 5;

public struct Material
{
    public float4 baseColor;       // rgb = albedo, a = alpha
    public float4 emission;        // rgb = emission color, a = emission strength
    public float metallic;
    public float roughness;        // 0.0 = mirror, 1.0 = diffuse
    public float ior;              // index of refraction (glass: 1.5)
    public int type;               // MAT_LAMBERTIAN / MAT_METAL / MAT_DIELECTRIC / MAT_EMISSIVE
    public int diffuseTexIndex;    // -1 = no texture
    public int styleType;          // 0 = STYLE_NONE
    public float styleParam0;      // parameter for stylization function g
    public int _pad2;

    public bool canTransmit() { return type == MAT_DIELECTRIC; }
    public bool isEmissive()  { return type == MAT_EMISSIVE; }

    public bool isDeltaBRDF()
    {
        if (type == MAT_DIELECTRIC) return true; // always delta
        if (type == MAT_METAL) return roughness < DELTA_ROUGHNESS_THRESHOLD;
        return false;
    }
};

public struct LightTriangle
{
    public float area;
    public int indexOffset;
    public int vertexOffset;
    public int matIndex;
};
