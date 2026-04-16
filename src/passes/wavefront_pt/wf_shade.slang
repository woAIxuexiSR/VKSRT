// Wavefront PT: Shade kernel
// Material eval, BRDF sampling, NEE preparation, compaction, G-buffer write
// 1D indirect dispatch from counterBuffer[0..2]

import wavefront_structs;
import material;
import brdf;
import onb;
import rng;
import math_utils;

struct PushConstants
{
    int frameIndex;
    int maxDepth;
    int rrDepth;
    int lightCount;
    int useNEE;
    int useMIS;
    int screenWidth;
    int screenHeight;
    int totalPaths;
    int bounceIndex;
    int queueOffset;
    int _pad;
};

[[vk::push_constant]]
PushConstants pc;

[[vk::binding(0)]]
RWStructuredBuffer<PathState> pathStates;

[[vk::binding(1)]]
RWStructuredBuffer<RayData> rays;

[[vk::binding(2)]]
RWStructuredBuffer<HitInfo> hitInfos;

[[vk::binding(3)]]
RWStructuredBuffer<ShadowRay> shadowRays;

[[vk::binding(4)]]
RWStructuredBuffer<uint> counters;

[[vk::binding(5)]]
RWStructuredBuffer<uint> queue;

[[vk::binding(9)]]
StructuredBuffer<float> vertexData;

[[vk::binding(10)]]
StructuredBuffer<uint> indices;

[[vk::binding(11)]]
StructuredBuffer<Material> materials;

[[vk::binding(14)]]
StructuredBuffer<LightTriangle> lights;

// G-buffer
[[vk::binding(16)]]
[[vk::image_format("rgba32f")]]
RWTexture2D<float4> gbufferNormal;

[[vk::binding(17)]]
[[vk::image_format("rgba32f")]]
RWTexture2D<float4> gbufferPosition;

[[vk::binding(18)]]
[[vk::image_format("rgba32f")]]
RWTexture2D<float4> gbufferAlbedo;

float3 getVertex(uint index)
{
    return float3(vertexData[index * 3], vertexData[index * 3 + 1], vertexData[index * 3 + 2]);
}

uint3 getTriIndices(int indexOffset)
{
    return uint3(
        indices[indexOffset * 3 + 0],
        indices[indexOffset * 3 + 1],
        indices[indexOffset * 3 + 2]);
}

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    uint threadIdx = dispatchThreadID.x;

    // Read path index from current queue
    uint pathIdx = queue[pc.queueOffset + threadIdx];
    if (pathIdx >= uint(pc.totalPaths)) return;

    PathState path = pathStates[pathIdx];
    if (path.alive == 0) return;

    HitInfo hit = hitInfos[pathIdx];
    RayData ray = rays[pathIdx];

    int2 pixel = int2(path.pixelIndex % pc.screenWidth, path.pixelIndex / pc.screenWidth);

    // --- Miss ---
    if (hit.isHit == 0)
    {
        path.alive = 0;
        if (path.depth == 0)
        {
            gbufferNormal[pixel] = float4(0.0);
            gbufferPosition[pixel] = float4(0.0);
            gbufferAlbedo[pixel] = float4(0.0);
        }
        pathStates[pathIdx] = path;
        return;
    }

    Material mat = materials[hit.matIndex];

    // --- G-buffer write (depth 0 only) ---
    if (path.depth == 0)
    {
        float3 N = normalize(hit.hitNormal);
        if (dot(N, ray.direction) > 0.0) N = -N;
        gbufferNormal[pixel] = float4(N, 1.0);
        gbufferPosition[pixel] = float4(hit.hitPos, 1.0);
        gbufferAlbedo[pixel] = float4(mat.baseColor.rgb, 1.0);
    }

    // --- Emissive hit: add emission with MIS weight ---
    if (mat.isEmissive())
    {
        float3 emission = mat.emission.rgb * mat.emission.a;
        float misWeight = 1.0;

        if (path.depth > 0 && pc.useNEE != 0 && pc.lightCount > 0)
        {
            if (path.prevBsdfPdf <= 0.0)
            {
                misWeight = 1.0; // Delta BRDF: full weight
            }
            else if (pc.useMIS != 0)
            {
                float3 hitNormal = normalize(hit.hitNormal);
                float dist2 = dot(ray.origin - hit.hitPos, ray.origin - hit.hitPos);
                float cosLight = abs(dot(hitNormal, normalize(ray.origin - hit.hitPos)));
                float totalArea = 0.0;
                for (int li = 0; li < pc.lightCount; li++)
                    totalArea += lights[li].area;
                if (cosLight > 1e-4 && totalArea > 0.0)
                {
                    float pdfLight = dist2 / (totalArea * cosLight);
                    misWeight = powerHeuristic(path.prevBsdfPdf, pdfLight);
                }
                else
                    misWeight = 0.0;
            }
            else
            {
                misWeight = 0.0; // NEE on but MIS off
            }
        }

        path.L += path.throughput * emission * misWeight;
        path.alive = 0;
        pathStates[pathIdx] = path;
        return;
    }

    // --- Normal flipping ---
    float3 shadingN = hit.hitNormal;
    if (!mat.canTransmit())
    {
        if (dot(shadingN, ray.direction) > 0.0)
            shadingN = -shadingN;
    }

    // --- NEE: Direct light sampling ---
    if (pc.useNEE != 0 && pc.lightCount > 0 && !mat.isDeltaBRDF())
    {
        int lightIdx = int(randFloat(path.seed) * float(pc.lightCount));
        lightIdx = min(lightIdx, pc.lightCount - 1);

        LightTriangle lt = lights[lightIdx];
        uint3 triIdx = getTriIndices(lt.indexOffset);
        float3 lv0 = getVertex(triIdx.x + lt.vertexOffset);
        float3 lv1 = getVertex(triIdx.y + lt.vertexOffset);
        float3 lv2 = getVertex(triIdx.z + lt.vertexOffset);

        float3 lightPoint = samplePointOnTriangle(lv0, lv1, lv2, randFloat2(path.seed));
        float3 lightNormal = normalize(cross(lv1 - lv0, lv2 - lv0));

        float3 toLight = lightPoint - hit.hitPos;
        float dist = length(toLight);
        float3 lightDir = toLight / dist;

        float cosHit = dot(shadingN, lightDir);
        float cosLight = abs(dot(lightNormal, -lightDir));

        if (cosHit > 1e-4 && cosLight > 1e-4)
        {
            Material lightMat = materials[lt.matIndex];
            float3 lightEmission = lightMat.emission.rgb * lightMat.emission.a;

            float pdfLight = (dist * dist) / (lt.area * cosLight * float(pc.lightCount));

            ONB neeBasis = buildONB(shadingN);
            float3 wi = worldToLocal(neeBasis, lightDir);
            float3 wo_nee = worldToLocal(neeBasis, -ray.direction);
            float3 bsdfVal = brdfEval(wi, wo_nee, mat);

            float w = 1.0;
            if (pc.useMIS != 0)
            {
                float pdfBsdf = brdfPdf(wi, wo_nee, mat);
                w = powerHeuristic(pdfLight, pdfBsdf);
            }

            float3 contribution = path.throughput * bsdfVal * lightEmission * cosHit * w / pdfLight;

            // Enqueue shadow ray
            uint shadowIdx;
            InterlockedAdd(counters[7], 1u, shadowIdx);

            ShadowRay sr;
            sr.origin = hit.hitPos;
            sr.tMax = dist - 0.001;
            sr.direction = lightDir;
            sr.pathIndex = int(pathIdx);
            sr.contribution = contribution;
            sr._pad = 0.0;
            shadowRays[shadowIdx] = sr;
        }
    }

    // --- BRDF importance sampling for next bounce ---
    ONB basis = buildONB(shadingN);
    float3 wo = worldToLocal(basis, -ray.direction);

    BRDFSample s = brdfSample(wo, mat, randFloat3(path.seed));

    if (!s.valid || s.pdf < 1e-6)
    {
        path.alive = 0;
        pathStates[pathIdx] = path;
        return;
    }

    // Track BRDF pdf for MIS on next emissive hit
    path.prevBsdfPdf = s.isDelta ? 0.0 : s.pdf;

    // Update throughput
    if (s.isDelta)
        path.throughput *= s.value;
    else
        path.throughput *= s.value * abs(s.wi.z) / s.pdf;

    // Set up next ray
    rays[pathIdx].origin = hit.hitPos;
    rays[pathIdx].direction = normalize(localToWorld(basis, s.wi));
    rays[pathIdx].tMin = 0.001;
    rays[pathIdx].tMax = 10000.0;

    path.depth++;

    // --- Russian Roulette ---
    if (path.depth >= pc.rrDepth)
    {
        float p = max(max(path.throughput.r, max(path.throughput.g, path.throughput.b)), 0.05);
        if (randFloat(path.seed) > p)
        {
            path.alive = 0;
            pathStates[pathIdx] = path;
            return;
        }
        path.throughput /= p;
    }

    // --- Compaction: write to next queue ---
    uint nextOffset = (pc.queueOffset == 0) ? uint(pc.totalPaths) : 0u;
    uint nextIdx;
    InterlockedAdd(counters[6], 1u, nextIdx);
    queue[nextOffset + nextIdx] = pathIdx;

    pathStates[pathIdx] = path;
}
