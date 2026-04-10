// Shared math utility functions
module math_utils;

public static const float PI = 3.14159265358979323846;

// Cosine-weighted hemisphere sampling (z-up)
public float3 cosineSampleHemisphere(float2 u)
{
    float phi = 2.0 * PI * u.x;
    float cosTheta = sqrt(1.0 - u.y);
    float sinTheta = sqrt(u.y);
    return float3(cos(phi) * sinTheta, sin(phi) * sinTheta, cosTheta);
}

// Uniform sampling on a unit sphere
public float3 randomOnSphere(float2 u)
{
    float z = 1.0 - 2.0 * u.x;
    float r = sqrt(max(0.0, 1.0 - z * z));
    float phi = 2.0 * PI * u.y;
    return float3(r * cos(phi), r * sin(phi), z);
}

// Schlick's Fresnel approximation
public float schlickFresnel(float cosTheta, float etaRatio)
{
    float r0 = (1.0 - etaRatio) / (1.0 + etaRatio);
    r0 = r0 * r0;
    float x = 1.0 - cosTheta;
    return r0 + (1.0 - r0) * x * x * x * x * x;
}

// Uniformly sample a point on a triangle given two random numbers in [0,1)
public float3 samplePointOnTriangle(float3 v0, float3 v1, float3 v2, float2 u)
{
    float su0 = sqrt(u.x);
    float b0 = 1.0 - su0;
    float b1 = u.y * su0;
    return b0 * v0 + b1 * v1 + (1.0 - b0 - b1) * v2;
}

// Power heuristic for MIS (balance heuristic with power=2)
public float powerHeuristic(float pdfA, float pdfB)
{
    float a2 = pdfA * pdfA;
    float b2 = pdfB * pdfB;
    return a2 / (a2 + b2);
}
