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

// ============================================================
// Instance transform utilities
// ============================================================

// Transform object-space position to world space using a 3x4 row-major matrix
public float3 transformPosition(float3x4 objectToWorld, float3 objectPos)
{
    return mul(objectToWorld, float4(objectPos, 1.0));
}

// Transform object-space normal to world space using inverse-transpose
// worldToObject = inverse of objectToWorld (3x4 row-major)
public float3 transformNormal(float3x4 worldToObject, float3 objectNormal)
{
    // normal transform = transpose of the 3x3 part of worldToObject
    float3x3 normalMatrix = float3x3(
        float3(worldToObject[0][0], worldToObject[1][0], worldToObject[2][0]),
        float3(worldToObject[0][1], worldToObject[1][1], worldToObject[2][1]),
        float3(worldToObject[0][2], worldToObject[1][2], worldToObject[2][2]));
    return normalize(mul(normalMatrix, objectNormal));
}

// ============================================================
// GGX Microfacet utilities
// ============================================================

// GGX (Trowbridge-Reitz) Normal Distribution Function
// alpha = roughness (NOT pre-squared; squared internally)
public float D_GGX(float NdotH, float alpha)
{
    if (NdotH <= 0.0) return 0.0;
    float a2 = alpha * alpha;
    float denom = NdotH * NdotH * (a2 - 1.0) + 1.0;
    return a2 / (PI * denom * denom);
}

// Smith G1 masking function for GGX
public float G1_SmithGGX(float NdotX, float alpha)
{
    if (NdotX <= 0.0) return 0.0;
    float a2 = alpha * alpha;
    return 2.0 * NdotX / (NdotX + sqrt(a2 + (1.0 - a2) * NdotX * NdotX));
}

// Smith separable G2 = G1(V) * G1(L)
public float G_SmithGGX(float NdotV, float NdotL, float alpha)
{
    return G1_SmithGGX(NdotV, alpha) * G1_SmithGGX(NdotL, alpha);
}

// GGX VNDF importance sampling (Heitz 2018, simplified spherical cap variant)
// Ve: view direction in local space (z-up), must have Ve.z > 0
// alpha: roughness (squared internally by D_GGX/G1; here used as stretch factor)
// u: float2 uniform random in [0,1)
// Returns: sampled half-vector (microfacet normal) in local space
public float3 sampleGGXVNDF(float3 Ve, float alpha, float2 u)
{
    // Stretch the view vector to the hemisphere configuration
    float3 Vh = normalize(float3(alpha * Ve.x, alpha * Ve.y, Ve.z));

    // Build orthonormal basis around Vh
    float lensq = Vh.x * Vh.x + Vh.y * Vh.y;
    float3 T1 = lensq > 0.0 ? float3(-Vh.y, Vh.x, 0.0) / sqrt(lensq) : float3(1.0, 0.0, 0.0);
    float3 T2 = cross(Vh, T1);

    // Sample a point on the hemisphere (spherical cap method)
    float r = sqrt(u.x);
    float phi = 2.0 * PI * u.y;
    float t1 = r * cos(phi);
    float t2 = r * sin(phi);
    float s = 0.5 * (1.0 + Vh.z);
    t2 = (1.0 - s) * sqrt(1.0 - t1 * t1) + s * t2;

    // Compute the normal on the hemisphere
    float3 Nh = t1 * T1 + t2 * T2 + sqrt(max(0.0, 1.0 - t1 * t1 - t2 * t2)) * Vh;

    // Unstretch
    return normalize(float3(alpha * Nh.x, alpha * Nh.y, max(0.0, Nh.z)));
}

// PDF of VNDF sampling for the reflected direction wi
// Ve: view/outgoing direction in local space (z-up)
// H: half-vector (microfacet normal) in local space
// alpha: roughness (same as passed to D_GGX / sampleGGXVNDF)
// Returns: pdf in solid angle measure of the reflected direction
public float ggxReflectionPdf(float3 Ve, float3 H, float alpha)
{
    float NdotH = H.z;
    float VdotH = max(dot(Ve, H), 0.0);
    if (VdotH <= 0.0 || NdotH <= 0.0) return 0.0;

    // VNDF pdf for half-vector: D(H) * G1(V) * VdotH / Ve.z
    float D = D_GGX(NdotH, alpha);
    float G1 = G1_SmithGGX(Ve.z, alpha);
    float pdfH = D * G1 * VdotH / Ve.z;

    // Jacobian: dH/dwi for reflection = 1 / (4 * VdotH)
    return pdfH / (4.0 * VdotH);
}

// Vector Fresnel (Schlick) — for metallic materials where F0 = baseColor
public float3 fresnelSchlickVec3(float cosTheta, float3 F0)
{
    float x = 1.0 - cosTheta;
    float x5 = x * x * x * x * x;
    return F0 + (float3(1.0) - F0) * x5;
}
