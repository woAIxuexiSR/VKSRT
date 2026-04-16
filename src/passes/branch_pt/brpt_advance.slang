// Branch PT: Advance megakernel — process vertices at current depth, spawn children
// Indirect 1D dispatch from counterBuffer[2..4]
// For each vertex at current depth:
//   - Read hit info from vertex fields (no separate hitInfoBuffer)
//   - Handle emissive hit (local emission + MIS weight)
//   - Write G-buffer (depth 0 only)
//   - Determine branching factor via actualBranchingDepth
//   - For each child: NEE (inline shadow trace) + BRDF sample + inline ray trace
//   - Child stores edge throughput only + embedded hit info

import brpt_structs;
import wavefront_structs;
import material;
import brdf;
import onb;
import rng;
import math_utils;
import stylize;

struct PushConstants
{
    int frameIndex;
    int maxDepth;
    int lightCount;
    int useNEE;
    int useMIS;
    int screenWidth;
    int screenHeight;
    int currentDepth;
    int useDebiasing;
    float debiasR;
    int innerSamples0;
    int innerSamples1;
    int innerSamples2;
    int innerSamples3;
    int maxVertices;
    int tilePixelOffset;
    int tilePixelCount;
    int rrDepth;
};

[[vk::push_constant]]
PushConstants pc;

[[vk::binding(0)]]
RWStructuredBuffer<BrPTVertex> vertices;

[[vk::binding(1)]]
RWStructuredBuffer<uint> counters; // [0]=vtxCounter, [1]=debiasCounter, [2,3,4]=indirect

[[vk::binding(2)]]
RWStructuredBuffer<int2> depthRanges;

[[vk::binding(5)]]
RaytracingAccelerationStructure topLevelAS;

[[vk::binding(6)]]
StructuredBuffer<float> vertexData;

[[vk::binding(7)]]
StructuredBuffer<uint> indices;

[[vk::binding(8)]]
StructuredBuffer<Material> materials;

[[vk::binding(9)]]
StructuredBuffer<float> normalData;

[[vk::binding(11)]]
StructuredBuffer<LightTriangle> lights;

[[vk::binding(12)]]
StructuredBuffer<float3x4> instanceTransforms;

[[vk::binding(13)]]
StructuredBuffer<MeshInfo> meshInfos;

// Debias buffers
[[vk::binding(14)]]
RWStructuredBuffer<float4> debiasDirectBuffer;

float3 getVertexPos(uint index)
{
    return float3(vertexData[index * 3], vertexData[index * 3 + 1], vertexData[index * 3 + 2]);
}

float3 getNormal(uint index)
{
    return float3(normalData[index * 3], normalData[index * 3 + 1], normalData[index * 3 + 2]);
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

// Inline ray trace helper — returns hit info embedded in vertex fields
struct InlineHitResult
{
    float3 hitPos;
    float3 hitNormal;
    int matIndex;
    int isHit;
};

InlineHitResult traceRay(float3 origin, float3 dir, float tMin, float tMax)
{
    RayDesc rayDesc;
    rayDesc.Origin = origin;
    rayDesc.TMin = tMin;
    rayDesc.Direction = dir;
    rayDesc.TMax = tMax;

    RayQuery<RAY_FLAG_FORCE_OPAQUE> rq;
    rq.TraceRayInline(topLevelAS, RAY_FLAG_FORCE_OPAQUE, 0xff, rayDesc);
    rq.Proceed();

    InlineHitResult result;
    if (rq.CommittedStatus() == COMMITTED_TRIANGLE_HIT)
    {
        uint instanceIdx = rq.CommittedInstanceID();
        uint primIdx = rq.CommittedPrimitiveIndex();
        float2 bary2 = rq.CommittedTriangleBarycentrics();
        float3 bary = float3(1.0 - bary2.x - bary2.y, bary2.x, bary2.y);

        MeshInfo mi = meshInfos[instanceIdx];
        uint i0 = indices[(mi.indexOffset + primIdx) * 3 + 0];
        uint i1 = indices[(mi.indexOffset + primIdx) * 3 + 1];
        uint i2 = indices[(mi.indexOffset + primIdx) * 3 + 2];

        float3 v0 = getVertexPos(i0 + mi.vertexOffset);
        float3 v1 = getVertexPos(i1 + mi.vertexOffset);
        float3 v2 = getVertexPos(i2 + mi.vertexOffset);

        float3 n0 = getNormal(i0 + mi.vertexOffset);
        float3 n1 = getNormal(i1 + mi.vertexOffset);
        float3 n2 = getNormal(i2 + mi.vertexOffset);

        float3 objPos = bary.x * v0 + bary.y * v1 + bary.z * v2;
        float3 objN = normalize(bary.x * n0 + bary.y * n1 + bary.z * n2);
        result.hitPos = transformPosition(rq.CommittedObjectToWorld3x4(), objPos);
        result.hitNormal = transformNormal(rq.CommittedWorldToObject3x4(), objN);
        result.matIndex = mi.matIndex;
        result.isHit = 1;
    }
    else
    {
        result.hitPos = float3(0.0);
        result.hitNormal = float3(0.0);
        result.matIndex = 0;
        result.isHit = 0;
    }
    return result;
}

bool traceShadow(float3 origin, float3 dir, float tMax)
{
    RayDesc rayDesc;
    rayDesc.Origin = origin;
    rayDesc.TMin = 0.001;
    rayDesc.Direction = dir;
    rayDesc.TMax = tMax;

    RayQuery<RAY_FLAG_FORCE_OPAQUE | RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH> rq;
    rq.TraceRayInline(topLevelAS,
                      RAY_FLAG_FORCE_OPAQUE | RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH,
                      0xff, rayDesc);
    rq.Proceed();

    return rq.CommittedStatus() == COMMITTED_NOTHING; // true = unoccluded
}

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    uint threadIdx = dispatchThreadID.x;

    // Read current depth range from depthRangeBuffer
    int2 range = depthRanges[pc.currentDepth];
    int depthStart = range.x;
    int depthCount = range.y;

    if (threadIdx >= uint(depthCount)) return;

    int vtxIdx = depthStart + int(threadIdx);
    if (vtxIdx >= pc.maxVertices) return;

    BrPTVertex vtx = vertices[vtxIdx];
    if (vtx.isHit == 0) return;

    int2 pixel = int2(vtx.pixelIndex % pc.screenWidth, vtx.pixelIndex / pc.screenWidth);
    Material mat = materials[vtx.matIndex];
    vtx.styleType = mat.styleType;
    vtx.styleParam0 = mat.styleParam0;

    // --- Emissive hit ---
    if (mat.isEmissive())
    {
        float3 emission = mat.emission.rgb * mat.emission.a;
        vtx.radiance = emission;

        // MIS weight for BSDF-sampled light hit
        float misWeight = 1.0;
        if (vtx.depth > 0 && pc.useNEE != 0 && pc.lightCount > 0)
        {
            if (vtx.prevBsdfPdf <= 0.0)
            {
                misWeight = 1.0;
            }
            else if (pc.useMIS != 0)
            {
                float3 hitN = normalize(vtx.hitNormal);
                float3 toParent = vertices[vtx.parent].hitPos - vtx.hitPos;
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
        vtx.isHit = 0; // terminal
        vertices[vtxIdx] = vtx;
        return;
    }

    // --- Normal flipping ---
    float3 shadingN = normalize(vtx.hitNormal);
    if (!mat.canTransmit())
    {
        if (dot(shadingN, vtx.wo) < 0.0)
            shadingN = -shadingN;
    }

    // --- Determine branching factor ---
    bool needBranching = !mat.isDeltaBRDF();

    int baseSamples;
    int debiasingSamples = 0;
    int actualBranchingDepth = vtx.actualBranchingDepth;

    if (needBranching)
    {
        baseSamples = getInnerSamples(actualBranchingDepth);

        if (pc.useDebiasing != 0)
        {
            debiasingSamples = 0;
            for (int k = 0; k < 64; k++)
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

    // Russian Roulette
    if (vtx.depth >= pc.rrDepth)
    {
        float survivalProb = min(max(vtx.throughput.r, max(vtx.throughput.g, vtx.throughput.b)), 0.95);
        if (randFloat(vtx.seed) >= survivalProb)
        {
            vtx.isHit = 0;
            vtx.nInnerSamples = 0;
            vertices[vtxIdx] = vtx;
            return;
        }
        vtx.throughput /= survivalProb;
    }

    // Don't spawn children beyond max depth
    if (vtx.depth + 1 > pc.maxDepth)
    {
        vtx.isHit = 0;
        vtx.nInnerSamples = 0;
        vertices[vtxIdx] = vtx;
        return;
    }

    // Allocate debias buffer entries (only for styled vertices)
    int debiasIdx = -1;
    bool useDebiasBuf = pc.useDebiasing != 0 && nSamples > 0 && vtx.styleType != STYLE_NONE;
    if (useDebiasBuf)
    {
        uint allocIdx;
        InterlockedAdd(counters[1], uint(nSamples), allocIdx);
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
        float3 localDirect = float3(0.0);

        bool isDelta = mat.isDeltaBRDF();
        if (pc.useNEE != 0 && pc.lightCount > 0 && !isDelta)
        {
            int lightIdx = int(randFloat(vtx.seed) * float(pc.lightCount));
            lightIdx = min(lightIdx, pc.lightCount - 1);

            LightTriangle lt = lights[lightIdx];
            uint3 triIdx = getTriIndices(lt.indexOffset);
            float3x4 lightXform = instanceTransforms[lt.instanceIndex];
            float3 lv0 = transformPosition(lightXform, getVertexPos(triIdx.x + lt.vertexOffset));
            float3 lv1 = transformPosition(lightXform, getVertexPos(triIdx.y + lt.vertexOffset));
            float3 lv2 = transformPosition(lightXform, getVertexPos(triIdx.z + lt.vertexOffset));

            float3 lightPoint = samplePointOnTriangle(lv0, lv1, lv2, randFloat2(vtx.seed));
            float3 lightNormal = normalize(cross(lv1 - lv0, lv2 - lv0));

            float3 toLight = lightPoint - vtx.hitPos;
            float dist = length(toLight);
            float3 lightDir = toLight / dist;

            float cosHit = dot(shadingN, lightDir);
            float cosLight = abs(dot(lightNormal, -lightDir));

            if (cosHit > 1e-4 && cosLight > 1e-4)
            {
                // Inline shadow trace
                if (traceShadow(vtx.hitPos, lightDir, dist - 0.001))
                {
                    Material lightMat = materials[lt.matIndex];
                    float3 lightEmission = lightMat.emission.rgb * lightMat.emission.a;

                    // Stylize light emission at source (light-end stylization)
                    lightEmission = applyStylize(lightEmission, lightMat.styleType, lightMat.styleParam0);

                    float pdfLight = (dist * dist) / (lt.area * cosLight * float(pc.lightCount));

                    float3 wi = worldToLocal(basis, lightDir);
                    float3 bsdfVal = brdfEval(wi, wo_local, mat);

                    float w = 1.0;
                    if (pc.useMIS != 0)
                    {
                        float pdfBsdf = brdfPdf(wi, wo_local, mat);
                        w = powerHeuristic(pdfLight, pdfBsdf);
                    }

                    localDirect = bsdfVal * lightEmission * cosHit * w / pdfLight;
                }
            }
        }

        // Write NEE direct to debias buffer or accumulate to vertex radiance
        if (useDebiasBuf && debiasIdx >= 0)
        {
            debiasDirectBuffer[debiasIdx + i] = float4(localDirect, 0.0);
        }
        else
        {
            // No debiasing: accumulate direct / nSamples to vertex radiance
            vtx.radiance += localDirect / float(nSamples);
        }

        // --- BRDF sampling for child ---
        BRDFSample s = brdfSample(wo_local, mat, randFloat3(vtx.seed));

        if (!s.valid || s.pdf < 1e-6)
        {
            vtx.seed = wangHash(vtx.seed);
            continue;
        }

        // Edge throughput only (f*cos/pdf for this segment)
        float3 edgeThroughput;
        if (s.isDelta)
            edgeThroughput = s.value;
        else
            edgeThroughput = s.value * abs(s.wi.z) / s.pdf;

        // Inline ray trace for child
        float3 childDir = normalize(localToWorld(basis, s.wi));
        InlineHitResult childHit = traceRay(vtx.hitPos, childDir, 0.001, 10000.0);

        // Allocate child vertex
        uint childGlobalIdx;
        InterlockedAdd(counters[0], 1u, childGlobalIdx);
        int childIdx = int(childGlobalIdx);
        if (childIdx >= pc.maxVertices)
        {
            vtx.seed = wangHash(vtx.seed);
            continue;
        }

        // Link into sibling list
        if (prevChildIdx >= 0)
            vertices[prevChildIdx].nextSibling = childIdx;
        else
            vtx.firstChild = childIdx;
        prevChildIdx = childIdx;

        // Create child vertex with embedded hit info
        BrPTVertex child;
        child.throughput = edgeThroughput;
        child.seed = vtx.seed;
        child.radiance = float3(0.0);
        child.pixelIndex = vtx.pixelIndex;
        child.wo = -childDir;
        child.depth = vtx.depth + 1;
        child.matIndex = childHit.matIndex;
        child.firstChild = -1;
        child.nextSibling = -1;
        child.parent = vtxIdx;
        child.isHit = childHit.isHit;
        child.nInnerSamples = 0;
        child.childIndex = i;
        child.debiasRadianceIdx = -1;
        child.hitPos = childHit.hitPos;
        child.prevBsdfPdf = s.isDelta ? 0.0 : s.pdf;
        child.hitNormal = childHit.hitNormal;
        child.styleType = 0;
        child.styleParam0 = 0.0;
        child.debiasingSamples = 0;
        child.actualBranchingDepth = actualBranchingDepth;
        child.mis = 1.0;
        vertices[childIdx] = child;

        // Diversify seed for next child
        vtx.seed = wangHash(vtx.seed);
    }

    vtx.isHit = 0; // done processing this vertex
    vertices[vtxIdx] = vtx;
}
