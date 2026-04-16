// BRDF interface: eval / sample / pdf per material type
// GGX microfacet model for Metal, diffuse + specular for Lambertian, delta for Dielectric
// All BRDF math operates in local space where normal = (0,0,1)
module brdf;

import material;
import math_utils;

public struct BRDFSample
{
    public float3 wi;
    public float3 value;
    public float  pdf;
    public bool   valid;
    public bool   isDelta;
};

// --- Local-space helpers ---

float3 reflectLocal(float3 wo)
{
    return float3(-wo.x, -wo.y, wo.z);
}

float3 reflectAbout(float3 wo, float3 wh)
{
    return 2.0 * dot(wo, wh) * wh - wo;
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

float dielectricF0(float ior)
{
    float r = (1.0 - ior) / (1.0 + ior);
    return r * r;
}

// ============================================================
// BRDF evaluation: f(wi, wo)
// ============================================================

public float3 brdfEval(float3 wi, float3 wo, Material mat)
{
    if (wo.z <= 0.0) return float3(0.0);

    float alpha = max(mat.roughness, 0.001);

    switch (mat.type)
    {
    case MAT_LAMBERTIAN:
    {
        if (wi.z <= 0.0) return float3(0.0);

        float3 wh = normalize(wi + wo);
        if (wh.z <= 0.0) return float3(0.0);

        float F0 = dielectricF0(mat.ior);
        float F = F0 + (1.0 - F0) * pow(1.0 - max(dot(wo, wh), 0.0), 5.0);

        float3 diffuse = (1.0 - F) * mat.baseColor.rgb / PI;

        float D = D_GGX(wh.z, alpha);
        float G = G_SmithGGX(wo.z, wi.z, alpha);
        float spec = F * D * G / (4.0 * wo.z * wi.z);

        return diffuse + float3(spec);
    }
    case MAT_METAL:
    {
        if (mat.isDeltaBRDF()) return float3(0.0);
        if (wi.z <= 0.0) return float3(0.0);

        float3 wh = normalize(wi + wo);
        if (wh.z <= 0.0) return float3(0.0);

        float3 F = fresnelSchlickVec3(max(dot(wo, wh), 0.0), mat.baseColor.rgb);
        float D = D_GGX(wh.z, alpha);
        float G = G_SmithGGX(wo.z, wi.z, alpha);

        return F * D * G / (4.0 * wo.z * wi.z);
    }
    default:
        return float3(0.0);
    }
}

// ============================================================
// Importance sampling: generate wi given wo
// ============================================================

public BRDFSample brdfSample(float3 wo, Material mat, float3 rng)
{
    BRDFSample s;
    s.valid = true;
    s.isDelta = false;
    s.wi = float3(0.0);
    s.value = float3(0.0);
    s.pdf = 1.0;

    float alpha = max(mat.roughness, 0.001);

    switch (mat.type)
    {
    case MAT_LAMBERTIAN:
    {
        float F0 = dielectricF0(mat.ior);
        float F = F0 + (1.0 - F0) * pow(1.0 - wo.z, 5.0);

        if (rng.z < F)
        {
            float3 wh = sampleGGXVNDF(wo, alpha, rng.xy);
            s.wi = reflectAbout(wo, wh);
            if (s.wi.z <= 0.0) { s.valid = false; break; }
        }
        else
        {
            s.wi = cosineSampleHemisphere(rng.xy);
        }
        s.pdf = brdfPdf(s.wi, wo, mat);
        s.value = brdfEval(s.wi, wo, mat);
        break;
    }
    case MAT_METAL:
    {
        if (mat.isDeltaBRDF())
        {
            s.wi = reflectLocal(wo);
            s.pdf = 1.0;
            s.value = mat.baseColor.rgb;
            s.isDelta = true;
            if (s.wi.z <= 0.0) s.valid = false;
        }
        else
        {
            float3 wh = sampleGGXVNDF(wo, alpha, rng.xy);
            s.wi = reflectAbout(wo, wh);
            if (s.wi.z <= 0.0) { s.valid = false; break; }
            s.pdf = ggxReflectionPdf(wo, wh, alpha);
            s.value = brdfEval(s.wi, wo, mat);
        }
        break;
    }
    case MAT_DIELECTRIC:
    {
        float cosTheta = min(abs(wo.z), 1.0);
        float etaRatio = wo.z > 0.0 ? (1.0 / mat.ior) : mat.ior;
        float refl = schlickFresnel(cosTheta, etaRatio);

        if (rng.x < refl)
            s.wi = reflectLocal(wo);
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
        s.valid = false;
        break;
    }
    return s;
}

// ============================================================
// PDF evaluation: p(wi | wo)
// ============================================================

public float brdfPdf(float3 wi, float3 wo, Material mat)
{
    float alpha = max(mat.roughness, 0.001);

    switch (mat.type)
    {
    case MAT_LAMBERTIAN:
    {
        if (wi.z <= 0.0 || wo.z <= 0.0) return 0.0;

        float F0 = dielectricF0(mat.ior);
        float F = F0 + (1.0 - F0) * pow(1.0 - wo.z, 5.0);

        float3 wh = normalize(wi + wo);
        return (1.0 - F) * (wi.z / PI) + F * ggxReflectionPdf(wo, wh, alpha);
    }
    case MAT_METAL:
    {
        if (mat.isDeltaBRDF()) return 0.0;
        if (wi.z <= 0.0 || wo.z <= 0.0) return 0.0;

        float3 wh = normalize(wi + wo);
        return ggxReflectionPdf(wo, wh, alpha);
    }
    default:
        return 0.0;
    }
}
