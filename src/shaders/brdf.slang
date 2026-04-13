// BRDF interface: eval / sample / pdf per material type
// All BRDF math operates in local space where normal = (0,0,1)
module brdf;

import material;
import onb;
import math_utils;

// Return value for BRDF sampling
public struct BRDFSample
{
    public float3 wi;         // sampled incoming direction (local space)
    public float3 value;      // BRDF value f(wi, wo)
    public float  pdf;        // probability density
    public bool   valid;      // false if sampling failed
    public bool   isDelta;    // true for delta distributions (Metal, Dielectric)
};

// --- BRDF evaluation: f(wi, wo) ---
public float3 brdfEval(float3 wi, float3 wo, Material mat)
{
    switch (mat.type)
    {
    case MAT_LAMBERTIAN:
        return mat.baseColor.rgb / PI;
    case MAT_METAL:
        return float3(0.0);
    case MAT_DIELECTRIC:
        return float3(0.0);
    case MAT_EMISSIVE:
        return float3(0.0);
    default:
        return float3(0.0);
    }
}

// --- Importance sampling: generate wi given wo ---
public BRDFSample brdfSample(float3 wo, Material mat, float2 rng)
{
    BRDFSample s;
    s.valid = true;
    s.isDelta = false;
    s.wi = float3(0.0);
    s.value = float3(0.0);
    s.pdf = 1.0;

    switch (mat.type)
    {
    case MAT_LAMBERTIAN:
    {
        float3 localDir = cosineSampleHemisphere(rng);
        s.wi = localDir;
        s.pdf = localDir.z / PI;
        s.value = mat.baseColor.rgb / PI;
        s.isDelta = false;
        break;
    }
    case MAT_METAL:
    {
        float3 reflected = reflectLocal(wo);
        if (mat.roughness > 0.0)
        {
            float3 offset = mat.roughness * randomOnSphere(rng);
            reflected = normalize(reflected + offset);
        }
        s.wi = reflected;
        s.pdf = 1.0;
        s.value = mat.baseColor.rgb;
        s.isDelta = true;
        if (s.wi.z <= 0.0) s.valid = false;
        break;
    }
    case MAT_DIELECTRIC:
    {
        float cosTheta = min(abs(wo.z), 1.0);
        float etaRatio = wo.z > 0.0 ? (1.0 / mat.ior) : mat.ior;
        float refl = schlickFresnel(cosTheta, etaRatio);

        if (rng.x < refl)
        {
            s.wi = reflectLocal(wo);
        }
        else
        {
            s.wi = refractLocal(wo, etaRatio);
            if (length(s.wi) < 0.001)
                s.wi = reflectLocal(wo);
        }
        s.pdf = 1.0;
        s.value = float3(1.0);
        s.isDelta = true;
        break;
    }
    case MAT_EMISSIVE:
    {
        s.valid = false;
        break;
    }
    }
    return s;
}

// --- PDF evaluation: p(wi | wo) ---
public float brdfPdf(float3 wi, float3 wo, Material mat)
{
    switch (mat.type)
    {
    case MAT_LAMBERTIAN:
        return max(wi.z, 0.0) / PI;
    case MAT_METAL:
        return 0.0;
    case MAT_DIELECTRIC:
        return 0.0;
    case MAT_EMISSIVE:
        return 0.0;
    default:
        return 0.0;
    }
}

// --- Local-space helpers ---

float3 reflectLocal(float3 wo)
{
    return float3(-wo.x, -wo.y, wo.z);
}

float3 refractLocal(float3 wo, float etaRatio)
{
    float cosI = abs(wo.z);
    float sin2T = etaRatio * etaRatio * (1.0 - cosI * cosI);
    if (sin2T > 1.0) return float3(0.0);
    float cosT = sqrt(1.0 - sin2T);
    float signZ = (wo.z > 0.0) ? -1.0 : 1.0;
    return float3(-etaRatio * wo.x, -etaRatio * wo.y, signZ * cosT);
}
