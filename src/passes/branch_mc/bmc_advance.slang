// Branch MC: Advance kernel — process vertices at current depth, spawn children
// 1D dispatch: [depthVertexCount/256, 1, 1]
// For each vertex at current depth:
//   - Read hit info, material
//   - Handle emissive hit (store local emission, MIS weight in vtx.mis)
//   - Write G-buffer (depth 0 only)
//   - Determine branching factor via actualBranchingDepth (not absolute depth)
//   - Draw geometric distribution extra samples for debiasing
//   - Spawn nInnerSamples children: NEE (local, no parent throughput) + BRDF sample each
//   - Child stores edge throughput only (f*cos/pdf), NOT accumulated
//   - No Russian Roulette

import bmc_structs;
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
    int currentDepth;
    int depthVertexStart;
    int depthVertexCount;
    int useDebiasing;
    float debiasR;
    int innerSamples0;
    int innerSamples1;
    int innerSamples2;
    int innerSamples3;
    int maxVertices;
    int _pad0;
    int _pad1;
};

[[vk::push_constant]]
PushConstants pc;

[[vk::binding(0)]]
RWStructuredBuffer<BMCVertex> vertices;

[[vk::binding(1)]]
RWStructuredBuffer<RayData> rays;

[[vk::binding(2)]]
RWStructuredBuffer<HitInfo> hitInfos;

[[vk::binding(3)]]
RWStructuredBuffer<ShadowRay> shadowRays;

[[vk::binding(4)]]
RWStructuredBuffer<uint> counters;

[[vk::binding(5)]]
RWStructuredBuffer<int2> depthRanges;

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

// Debias buffers
[[vk::binding(19)]]
RWStructuredBuffer<float4> debiasDirectBuffer;

[[vk::binding(21)]]
RWStructuredBuffer<uint> debiasCounter;

// Acceleration structure for inline shadow tracing (debiasing mode)
[[vk::binding(8)]]
RaytracingAccelerationStructure topLevelAS;

float3 getVertexPos(uint index)
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

int getInnerSamples(int branchingDepth)
{
    if (branchingDepth == 0) return pc.innerSamples0;
    if (branchingDepth == 1) return pc.innerSamples1;
    if (branchingDepth == 2) return pc.innerSamples2;
    return pc.innerSamples3;
}

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    uint threadIdx = dispatchThreadID.x;
    if (threadIdx >= uint(pc.depthVertexCount)) return;

    int vtxIdx = pc.depthVertexStart + int(threadIdx);
    if (vtxIdx >= pc.maxVertices) return;

    BMCVertex vtx = vertices[vtxIdx];
    if (vtx.alive == 0) return;

    HitInfo hit = hitInfos[vtxIdx];
    int2 pixel = int2(vtx.pixelIndex % pc.screenWidth, vtx.pixelIndex / pc.screenWidth);

    // --- Miss ---
    if (hit.isHit == 0)
    {
        vtx.alive = 0;
        if (vtx.depth == 0)
        {
            gbufferNormal[pixel] = float4(0.0);
            gbufferPosition[pixel] = float4(0.0);
            gbufferAlbedo[pixel] = float4(0.0);
        }
        vertices[vtxIdx] = vtx;
        return;
    }

    Material mat = materials[hit.matIndex];
    vtx.matIndex = hit.matIndex;
    vtx.hitPos = hit.hitPos;
    vtx.hitNormal = hit.hitNormal;
    vtx.styleType = mat.styleType;
    vtx.styleParam0 = mat.styleParam0;

    // --- G-buffer write (depth 0 only) ---
    if (vtx.depth == 0)
    {
        float3 N = normalize(hit.hitNormal);
        if (dot(N, vtx.wo) < 0.0) N = -N;
        gbufferNormal[pixel] = float4(N, 1.0);
        gbufferPosition[pixel] = float4(hit.hitPos, 1.0);
        gbufferAlbedo[pixel] = float4(mat.baseColor.rgb, 1.0);
    }

    // --- Emissive hit ---
    if (mat.type == MAT_EMISSIVE)
    {
        float3 emission = mat.emission.rgb * mat.emission.a;

        // Store LOCAL emission (no parent throughput)
        vtx.radiance = emission;

        // Compute and store MIS weight separately
        float misWeight = 1.0;
        if (vtx.depth > 0 && pc.useNEE != 0 && pc.lightCount > 0)
        {
            if (vtx.prevBsdfPdf <= 0.0)
            {
                misWeight = 1.0;
            }
            else if (pc.useMIS != 0)
            {
                float3 hitN = normalize(hit.hitNormal);
                float3 toParent = vertices[vtx.parent].hitPos - hit.hitPos;
                float dist2 = dot(toParent, toParent);
                float cosLight = abs(dot(hitN, normalize(toParent)));
                float totalArea = 0.0;
                for (int li = 0; li < pc.lightCount; li++)
                    totalArea += lights[li].area;
                if (cosLight > 1e-4 && totalArea > 0.0)
                {
                    float pdfLight = dist2 / (totalArea * cosLight);
                    misWeight = powerHeuristic(vtx.prevBsdfPdf, pdfLight);
                }
                else
                    misWeight = 0.0;
            }
            else
            {
                misWeight = 0.0;
            }
        }
        vtx.mis = misWeight;

        vtx.alive = 0;
        vertices[vtxIdx] = vtx;
        return;
    }

    // --- Normal flipping ---
    float3 shadingN = normalize(hit.hitNormal);
    if (!mat.canTransmit())
    {
        if (dot(shadingN, vtx.wo) < 0.0)
            shadingN = -shadingN;
    }

    // --- Determine branching factor ---
    // Use actualBranchingDepth (not absolute depth) to index inner samples
    bool needBranching = mat.roughness >= 0.01 || (mat.type != MAT_METAL && mat.type != MAT_DIELECTRIC);
    // For specular surfaces (very low roughness metal/dielectric), force 1 sample
    if (mat.roughness < 0.01 && (mat.type == MAT_METAL || mat.type == MAT_DIELECTRIC))
        needBranching = false;

    int baseSamples;
    int debiasingSamples = 0;
    int actualBranchingDepth = vtx.actualBranchingDepth;

    if (needBranching)
    {
        baseSamples = getInnerSamples(actualBranchingDepth);

        // Geometric distribution extra samples for debiasing
        if (pc.useDebiasing != 0)
        {
            debiasingSamples = 0;
            // Draw from Geometric(r): keep going until success
            for (int k = 0; k < 64; k++) // safety cap
            {
                debiasingSamples++;
                if (randFloat(vtx.seed) < pc.debiasR)
                    break;
            }
        }

        actualBranchingDepth++; // increment for children
    }
    else
    {
        baseSamples = 1;
    }

    int nSamples = baseSamples + debiasingSamples;
    vtx.nInnerSamples = nSamples;
    vtx.debiasingSamples = debiasingSamples;

    // Don't spawn children if we've reached max depth
    if (vtx.depth + 1 > pc.maxDepth)
    {
        vtx.alive = 0;
        vtx.nInnerSamples = 0;
        vertices[vtxIdx] = vtx;
        return;
    }

    // Allocate debias buffer entries only for vertices with stylization
    // STYLE_NONE vertices use shadow rays for NEE (same as debiasing OFF path)
    int debiasIdx = -1;
    bool useDebiasBuf = pc.useDebiasing != 0 && nSamples > 0 && vtx.styleType != STYLE_NONE;
    if (useDebiasBuf)
    {
        uint allocIdx;
        InterlockedAdd(debiasCounter[0], uint(nSamples), allocIdx);
        debiasIdx = int(allocIdx);
    }
    vtx.debiasRadianceIdx = debiasIdx;

    // --- Spawn nSamples children ---
    int prevChildIdx = -1;
    ONB basis = buildONB(shadingN);
    float3 wo_local = worldToLocal(basis, vtx.wo);

    for (int i = 0; i < nSamples; i++)
    {
        // --- NEE for this child ---
        if (pc.useNEE != 0 && pc.lightCount > 0 && mat.type == MAT_LAMBERTIAN)
        {
            int lightIdx = int(randFloat(vtx.seed) * float(pc.lightCount));
            lightIdx = min(lightIdx, pc.lightCount - 1);

            LightTriangle lt = lights[lightIdx];
            uint3 triIdx = getTriIndices(lt.indexOffset);
            float3 lv0 = getVertexPos(triIdx.x + lt.vertexOffset);
            float3 lv1 = getVertexPos(triIdx.y + lt.vertexOffset);
            float3 lv2 = getVertexPos(triIdx.z + lt.vertexOffset);

            float3 lightPoint = samplePointOnTriangle(lv0, lv1, lv2, randFloat2(vtx.seed));
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

                float3 wi = worldToLocal(basis, lightDir);
                float3 bsdfVal = brdfEval(wi, wo_local, mat);

                float w = 1.0;
                if (pc.useMIS != 0)
                {
                    float pdfBsdf = brdfPdf(wi, wo_local, mat);
                    w = powerHeuristic(pdfLight, pdfBsdf);
                }

                // LOCAL direct contribution (no parent throughput)
                float3 localDirect = bsdfVal * lightEmission * cosHit * w / pdfLight;

                if (useDebiasBuf && debiasIdx >= 0)
                {
                    // Debiasing ON: store per-child direct in debias buffer
                    // Shadow ray carries the full local direct; on hit it adds to vtx.radiance
                    // But we write the debias buffer entry here — shadow kernel will only
                    // add to radiance if unoccluded. We need to handle this carefully:
                    // Write zero to debias buffer now; shadow kernel writes the actual value.
                    // Actually, reference traces shadow inline. We emit shadow ray with localDirect,
                    // and store localDirect in debias buffer. If occluded, shadow kernel won't add
                    // to radiance, but debias buffer still has the value. We need to zero it on occlusion.
                    // Simpler: store localDirect in debias buffer, emit shadow ray that on success
                    // sets the debias buffer entry (and adds to radiance for non-debias path).
                    // But shadow kernel doesn't know the debias index...
                    //
                    // Approach: Store localDirect in debias buffer optimistically.
                    // Shadow ray adds to radiance on unoccluded.
                    // In propagate, use debias buffer entries for debiasing math.
                    // Problem: if occluded, debias buffer has wrong value.
                    //
                    // Reference traces shadow inline (CPU/CUDA). On GPU we decouple.
                    // Solution: shadow ray carries debias index and zeros the entry on occlusion.
                    // But our ShadowRay struct only has pathIndex...
                    //
                    // Alternative solution: Don't use shadow rays for debias mode.
                    // Instead, store localDirect in debias buffer, and shadow ray contribution = 0.
                    // Shadow kernel adds 0 to radiance (no-op).
                    // In propagate, read debias buffer for the direct values.
                    // But occluded entries shouldn't contribute!
                    //
                    // Best approach matching reference: The reference traces occlusion INLINE
                    // in advance. We can do the same with ray query.
                    // For debiasing mode: trace shadow inline, write result to debias buffer.
                    // For non-debiasing mode: emit shadow ray as before.

                    // Inline shadow trace for debiasing
                    RayDesc shadowDesc;
                    shadowDesc.Origin = hit.hitPos;
                    shadowDesc.TMin = 0.001;
                    shadowDesc.Direction = lightDir;
                    shadowDesc.TMax = dist - 0.001;

                    RayQuery<RAY_FLAG_FORCE_OPAQUE | RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH> rq;
                    rq.TraceRayInline(topLevelAS,
                                      RAY_FLAG_FORCE_OPAQUE | RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH,
                                      0xff, shadowDesc);
                    rq.Proceed();

                    if (rq.CommittedStatus() == COMMITTED_NOTHING)
                    {
                        // Unoccluded
                        debiasDirectBuffer[debiasIdx + i] = float4(localDirect, 0.0);
                    }
                    else
                    {
                        // Occluded
                        debiasDirectBuffer[debiasIdx + i] = float4(0.0);
                    }
                }
                else
                {
                    // Debiasing OFF: emit shadow ray with localDirect / nSamples
                    float3 contribution = localDirect / float(nSamples);

                    uint shadowIdx;
                    InterlockedAdd(counters[7], 1u, shadowIdx);

                    ShadowRay sr;
                    sr.origin = hit.hitPos;
                    sr.tMax = dist - 0.001;
                    sr.direction = lightDir;
                    sr.pathIndex = vtxIdx;
                    sr.contribution = contribution;
                    sr._pad = 0.0;
                    shadowRays[shadowIdx] = sr;
                }
            }
            else
            {
                if (useDebiasBuf && debiasIdx >= 0)
                    debiasDirectBuffer[debiasIdx + i] = float4(0.0);
            }
        }
        else
        {
            if (useDebiasBuf && debiasIdx >= 0)
                debiasDirectBuffer[debiasIdx + i] = float4(0.0);
        }

        // --- BRDF sampling for child ---
        float2 rng2 = randFloat2(vtx.seed);
        BRDFSample s = brdfSample(wo_local, mat, rng2);

        if (!s.valid || s.pdf < 1e-6)
        {
            // Dead child — debias direct entry already set above
            continue;
        }

        // Edge throughput only (f*cos/pdf for this segment, NOT accumulated)
        float3 edgeThroughput;
        if (s.isDelta)
            edgeThroughput = s.value;
        else
            edgeThroughput = s.value * max(s.wi.z, 0.0) / s.pdf;

        // No Russian Roulette — paths terminate only at maxDepth

        // Allocate child vertex
        uint childGlobalIdx;
        InterlockedAdd(counters[6], 1u, childGlobalIdx);
        int childIdx = pc.depthVertexStart + pc.depthVertexCount + int(childGlobalIdx);
        if (childIdx >= pc.maxVertices)
            continue;

        // Link into tree
        if (prevChildIdx >= 0)
            vertices[prevChildIdx].nextSibling = childIdx;
        else
            vtx.firstChild = childIdx;
        prevChildIdx = childIdx;

        // Create child vertex
        BMCVertex child;
        child.throughput = edgeThroughput;
        child.seed = vtx.seed;
        child.radiance = float3(0.0);
        child.pixelIndex = vtx.pixelIndex;
        child.wo = -normalize(localToWorld(basis, s.wi));
        child.depth = vtx.depth + 1;
        child.matIndex = -1;
        child.firstChild = -1;
        child.nextSibling = -1;
        child.parent = vtxIdx;
        child.alive = 1;
        child.nInnerSamples = 0;
        child.childIndex = i;
        child.debiasRadianceIdx = -1;
        child.hitPos = float3(0.0);
        child.prevBsdfPdf = s.isDelta ? 0.0 : s.pdf;
        child.hitNormal = float3(0.0);
        child.styleType = 0;
        child.styleParam0 = 0.0;
        child.debiasingSamples = 0;
        child.actualBranchingDepth = actualBranchingDepth;
        child.mis = 1.0;
        vertices[childIdx] = child;

        // Generate ray for child
        RayData childRay;
        childRay.origin = hit.hitPos;
        childRay.direction = -child.wo;
        childRay.tMin = 0.001;
        childRay.tMax = 10000.0;
        rays[childIdx] = childRay;

        // Diversify seed for next child
        vtx.seed = wangHash(vtx.seed);
    }

    vtx.alive = 0;
    vertices[vtxIdx] = vtx;
}
