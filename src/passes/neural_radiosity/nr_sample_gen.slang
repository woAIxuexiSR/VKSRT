// Neural Radiosity training sample generator. Per thread (one sample in the batch):
//   1. Sample a non-emissive, non-delta surface point x area-weighted (binary search
//      the cumulative-area table).
//   2. Sample omega_o cosine-weighted on x's hemisphere. Write LHS features
//      [x, n_x, omega_o, albedo_x].
//   3. NEE at x: sample a scene light, shadow-trace, compute MIS-weighted contribution.
//   4. BSDF-sample an incoming direction omega_i, extend-trace one bounce to y.
//      - y missed:        leave RHS features zeroed, flags=0.
//      - y emissive:      precompute emission * throughput * powerHeuristic(pdfBsdf, pdfLight)
//                         into aux, set validEmissive.
//      - y non-emissive:  write y's 12-dim features (with omega_o=-omega_i) into
//                         rhsInputBuffer, stash throughput, set validNet. buildTarget
//                         will multiply the EMA network prediction at y by throughput.
//
// aux layout (12 floats per sample):
//   [ 0..2 ] neeContrib        (MIS-weighted direct-light contribution at x)
//   [ 3..5 ] throughputNet     (s.value * |s.wi.z| / s.pdf at x; multiplied into rhsPred)
//   [ 6..8 ] emissionTerm      (precomputed emission from y with MIS weight)
//   [  9  ] validNet           (1.0 if y is a non-emissive hit)
//   [ 10  ] validEmissive      (1.0 if y is emissive)
//   [ 11  ] _debug             (unused)

import material;
import brdf;
import onb;
import rng;
import math_utils;
import wavefront_structs;

struct SampleGenPC
{
    uint64_t lhsInputAddr;   // B * 12 floats
    uint64_t rhsInputAddr;   // B * S * 12 floats, indexed as (b * S + s) * 12
    uint64_t auxAddr;        // B * S * 12 floats, same (b, s) indexing
    uint batchSize;
    uint rhsSamples;
    uint frameIndex;
    uint surfaceTriCount;
    int lightCount;
    float totalSurfaceArea;
    uint _pad0;
};

[[vk::push_constant]]
SampleGenPC pc;

[[vk::binding(0)]]
RaytracingAccelerationStructure topLevelAS;

[[vk::binding(1)]]
StructuredBuffer<float> vertexData;

[[vk::binding(2)]]
StructuredBuffer<uint> indices;

[[vk::binding(3)]]
StructuredBuffer<Material> materials;

[[vk::binding(4)]]
StructuredBuffer<float> normalData;

// binding(5) = texcoords (unused, kept for scene-descriptor layout uniformity)

[[vk::binding(6)]]
StructuredBuffer<LightTriangle> lights;

[[vk::binding(7)]]
StructuredBuffer<float3x4> instanceTransforms;

[[vk::binding(8)]]
StructuredBuffer<MeshInfo> meshInfos;

[[vk::binding(9)]]
StructuredBuffer<SurfaceTriangle> surfaceTriangles;

// --- Geometry helpers ---

float3 getVertex(uint i)
{
    return float3(vertexData[i * 3], vertexData[i * 3 + 1], vertexData[i * 3 + 2]);
}

float3 getNormal(uint i)
{
    return float3(normalData[i * 3], normalData[i * 3 + 1], normalData[i * 3 + 2]);
}

uint3 getTriIndices(int indexOffset)
{
    return uint3(indices[indexOffset * 3 + 0],
                 indices[indexOffset * 3 + 1],
                 indices[indexOffset * 3 + 2]);
}

// --- Inline ray query helpers (mirrors brpt_advance.slang pattern) ---

struct InlineHit
{
    float3 hitPos;
    float3 hitNormal;
    int matIndex;
    int isHit;
};

InlineHit traceRay(float3 origin, float3 dir, float tMin, float tMax)
{
    RayDesc ray;
    ray.Origin = origin; ray.TMin = tMin;
    ray.Direction = dir; ray.TMax = tMax;

    RayQuery<RAY_FLAG_FORCE_OPAQUE> rq;
    rq.TraceRayInline(topLevelAS, RAY_FLAG_FORCE_OPAQUE, 0xff, ray);
    rq.Proceed();

    InlineHit r;
    if (rq.CommittedStatus() == COMMITTED_TRIANGLE_HIT)
    {
        uint instanceIdx = rq.CommittedInstanceID();
        uint primIdx = rq.CommittedPrimitiveIndex();
        float2 b2 = rq.CommittedTriangleBarycentrics();
        float3 bary = float3(1.0 - b2.x - b2.y, b2.x, b2.y);

        MeshInfo mi = meshInfos[instanceIdx];
        uint i0 = indices[(mi.indexOffset + primIdx) * 3 + 0];
        uint i1 = indices[(mi.indexOffset + primIdx) * 3 + 1];
        uint i2 = indices[(mi.indexOffset + primIdx) * 3 + 2];

        float3 v0 = getVertex(i0 + mi.vertexOffset);
        float3 v1 = getVertex(i1 + mi.vertexOffset);
        float3 v2 = getVertex(i2 + mi.vertexOffset);
        float3 n0 = getNormal(i0 + mi.vertexOffset);
        float3 n1 = getNormal(i1 + mi.vertexOffset);
        float3 n2 = getNormal(i2 + mi.vertexOffset);

        float3 objPos = bary.x * v0 + bary.y * v1 + bary.z * v2;
        float3 objN = normalize(bary.x * n0 + bary.y * n1 + bary.z * n2);
        r.hitPos = transformPosition(rq.CommittedObjectToWorld3x4(), objPos);
        r.hitNormal = transformNormal(rq.CommittedWorldToObject3x4(), objN);
        r.matIndex = mi.matIndex;
        r.isHit = 1;
    }
    else
    {
        r.hitPos = float3(0.0);
        r.hitNormal = float3(0.0);
        r.matIndex = 0;
        r.isHit = 0;
    }
    return r;
}

bool traceShadow(float3 origin, float3 dir, float tMax)
{
    RayDesc ray;
    ray.Origin = origin; ray.TMin = 0.001;
    ray.Direction = dir; ray.TMax = tMax;

    RayQuery<RAY_FLAG_FORCE_OPAQUE | RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH> rq;
    rq.TraceRayInline(topLevelAS,
                      RAY_FLAG_FORCE_OPAQUE | RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH,
                      0xff, ray);
    rq.Proceed();
    return rq.CommittedStatus() == COMMITTED_NOTHING;
}

// Binary-search the cumulative-area table for target = u * totalSurfaceArea.
int pickSurfaceTriangle(float u)
{
    float target = u * pc.totalSurfaceArea;
    int lo = 0;
    int hi = int(pc.surfaceTriCount) - 1;
    // Find first index with cumArea >= target
    while (lo < hi)
    {
        int mid = (lo + hi) >> 1;
        if (surfaceTriangles[mid].cumArea >= target) hi = mid;
        else                                         lo = mid + 1;
    }
    return lo;
}

// --- Buffer write helpers ---

void writeLhs(uint tid, float3 pos, float3 n, float3 wo, float3 alb)
{
    float* b = (float*)pc.lhsInputAddr;
    uint i = tid * 12;
    b[i + 0]  = pos.x; b[i + 1]  = pos.y; b[i + 2]  = pos.z;
    b[i + 3]  = n.x;   b[i + 4]  = n.y;   b[i + 5]  = n.z;
    b[i + 6]  = wo.x;  b[i + 7]  = wo.y;  b[i + 8]  = wo.z;
    b[i + 9]  = alb.x; b[i + 10] = alb.y; b[i + 11] = alb.z;
}

void writeRhs(uint slot, float3 pos, float3 n, float3 wo, float3 alb)
{
    float* b = (float*)pc.rhsInputAddr;
    uint i = slot * 12;
    b[i + 0]  = pos.x; b[i + 1]  = pos.y; b[i + 2]  = pos.z;
    b[i + 3]  = n.x;   b[i + 4]  = n.y;   b[i + 5]  = n.z;
    b[i + 6]  = wo.x;  b[i + 7]  = wo.y;  b[i + 8]  = wo.z;
    b[i + 9]  = alb.x; b[i + 10] = alb.y; b[i + 11] = alb.z;
}

void writeRhsZero(uint slot)
{
    float* b = (float*)pc.rhsInputAddr;
    uint i = slot * 12;
    [unroll] for (int k = 0; k < 12; k++) b[i + k] = 0.0;
}

void writeAux(uint slot, float3 nee, float3 tpNet, float3 emit, float vNet, float vEmit)
{
    float* a = (float*)pc.auxAddr;
    uint i = slot * 12;
    a[i + 0]  = nee.x;   a[i + 1]  = nee.y;   a[i + 2]  = nee.z;
    a[i + 3]  = tpNet.x; a[i + 4]  = tpNet.y; a[i + 5]  = tpNet.z;
    a[i + 6]  = emit.x;  a[i + 7]  = emit.y;  a[i + 8]  = emit.z;
    a[i + 9]  = vNet;    a[i + 10] = vEmit;   a[i + 11] = 0.0;
}

[numthreads(16, 16, 1)]
[shader("compute")]
void main(uint3 dtid : SV_DispatchThreadID)
{
    uint bIdx = dtid.x;
    uint sIdx = dtid.y;
    if (bIdx >= pc.batchSize || sIdx >= pc.rhsSamples) return;

    uint slot = bIdx * pc.rhsSamples + sIdx;    // RHS/aux element index

    if (pc.surfaceTriCount == 0 || pc.totalSurfaceArea <= 0.0)
    {
        if (sIdx == 0)
            writeLhs(bIdx, float3(0.0), float3(0, 0, 1), float3(0, 0, 1), float3(0.0));
        writeRhsZero(slot);
        writeAux(slot, float3(0.0), float3(0.0), float3(0.0), 0.0, 0.0);
        return;
    }

    // LHS seed depends only on (frameIndex, bIdx). Every sIdx-thread in a batch
    // derives the same x / n_x / omega_o, so their network predictions at y
    // average into a single gt[bIdx].
    uint lhsSeed = wangHash(pc.frameIndex * 2654435761u + bIdx * 1099087573u + 1u);
    // -- 1. Pick surface triangle area-weighted + sample barycentric point --
    int triIdx = pickSurfaceTriangle(randFloat(lhsSeed));
    SurfaceTriangle st = surfaceTriangles[triIdx];

    uint3 tri = getTriIndices(st.indexOffset);
    float3x4 xform = instanceTransforms[st.instanceIndex];

    float3 lv0 = transformPosition(xform, getVertex(tri.x + st.vertexOffset));
    float3 lv1 = transformPosition(xform, getVertex(tri.y + st.vertexOffset));
    float3 lv2 = transformPosition(xform, getVertex(tri.z + st.vertexOffset));

    // Barycentric sample + interpolated shading normal
    float2 bary2 = randFloat2(lhsSeed);
    float su0 = sqrt(bary2.x);
    float3 bary = float3(1.0 - su0, bary2.y * su0, 1.0 - (1.0 - su0) - bary2.y * su0);
    float3 x = bary.x * lv0 + bary.y * lv1 + bary.z * lv2;

    float3 n0 = getNormal(tri.x + st.vertexOffset);
    float3 n1 = getNormal(tri.y + st.vertexOffset);
    float3 n2 = getNormal(tri.z + st.vertexOffset);
    float3 objN = normalize(bary.x * n0 + bary.y * n1 + bary.z * n2);
    // For robustness use the geometric normal cross-product (works with any instanceTransform).
    float3 geomN = normalize(cross(lv1 - lv0, lv2 - lv0));
    float3 n_x = geomN;

    Material mat = materials[st.matIndex];
    float3 albedo_x = mat.baseColor.rgb;

    ONB basis = buildONB(n_x);

    // -- 2. Cosine-weighted omega_o on x's hemisphere --
    float2 u_wo = randFloat2(lhsSeed);
    float3 wo_local = cosineSampleHemisphere(u_wo);
    float3 wo_world = normalize(localToWorld(basis, wo_local));

    // LHS only needs to be written once per (bIdx). Let sIdx==0 do the work.
    if (sIdx == 0)
        writeLhs(bIdx, x, n_x, wo_world, albedo_x);

    // RHS seed additionally includes sIdx so each sub-sample's NEE + BSDF sampling
    // takes an independent path.
    uint seed = wangHash(lhsSeed ^ (sIdx * 2246822519u + 2654435761u));

    // -- 3. NEE at x (resampled per sub-sample) --
    float3 neeContrib = float3(0.0);
    if (pc.lightCount > 0)
    {
        int li = int(randFloat(seed) * float(pc.lightCount));
        li = min(li, pc.lightCount - 1);

        LightTriangle lt = lights[li];
        uint3 lidx = getTriIndices(lt.indexOffset);
        float3x4 lxf = instanceTransforms[lt.instanceIndex];
        float3 Lv0 = transformPosition(lxf, getVertex(lidx.x + lt.vertexOffset));
        float3 Lv1 = transformPosition(lxf, getVertex(lidx.y + lt.vertexOffset));
        float3 Lv2 = transformPosition(lxf, getVertex(lidx.z + lt.vertexOffset));

        float3 lPt = samplePointOnTriangle(Lv0, Lv1, Lv2, randFloat2(seed));
        float3 lN = normalize(cross(Lv1 - Lv0, Lv2 - Lv0));

        float3 toLight = lPt - x;
        float dist = length(toLight);
        float3 lDir = toLight / dist;

        float cosHit = dot(n_x, lDir);
        float cosLight = abs(dot(lN, -lDir));

        if (cosHit > COS_EPSILON && cosLight > COS_EPSILON)
        {
            if (traceShadow(x, lDir, dist - 0.001))
            {
                Material lMat = materials[lt.matIndex];
                float3 lE = lMat.emission.rgb * lMat.emission.a;

                float pdfLight = (dist * dist) / (lt.area * cosLight * float(pc.lightCount));

                float3 wi_local = worldToLocal(basis, lDir);
                float3 bsdfVal = brdfEval(wi_local, wo_local, mat);
                float pdfBsdf = brdfPdf(wi_local, wo_local, mat);

                float w = 1.0;
                if (pdfLight > PDF_EPSILON)
                    w = powerHeuristic(pdfLight, pdfBsdf);

                neeContrib = bsdfVal * lE * cosHit * w / pdfLight;
            }
        }
    }

    // -- 4. BSDF-sample at x. First-bounce pdf is saved for MIS with NEE; any subsequent
    //    delta bounces (mirror / glass) are chained into throughput until we land on a
    //    non-delta surface (the network query target) or an emitter. The network has a
    //    hard time fitting high-frequency delta responses, so we skip past them entirely.

    BRDFSample bs = brdfSample(wo_local, mat, randFloat3(seed));
    if (!bs.valid || bs.pdf < PDF_EPSILON)
    {
        writeRhsZero(slot);
        writeAux(slot, neeContrib, float3(0.0), float3(0.0), 0.0, 0.0);
        return;
    }

    // x itself is non-delta (filtered out of surfaceTriangles), so bs.isDelta=false here
    // and the first-bounce throughput is the usual brdf*|cos|/pdf.
    float3 throughput = bs.value * abs(bs.wi.z) / bs.pdf;
    float firstBouncePdf = bs.pdf;   // used in MIS with NEE if we ultimately hit an emitter

    float3 rayOrigin = x;
    float3 rayDir = normalize(localToWorld(basis, bs.wi));

    const int kMaxDeltaBounces = 8;
    bool finished = false;
    bool hitEmitter = false;
    bool missed = false;
    InlineHit hit;
    Material matY;

    [loop]
    for (int depth = 0; depth < kMaxDeltaBounces; depth++)
    {
        hit = traceRay(rayOrigin, rayDir, 0.001, 10000.0);
        if (hit.isHit == 0)
        {
            missed = true;
            finished = true;
            break;
        }

        matY = materials[hit.matIndex];

        if (matY.isEmissive())
        {
            hitEmitter = true;
            finished = true;
            break;
        }

        if (!matY.isDeltaBRDF())
        {
            finished = true;
            break;
        }

        // Delta surface: sample its BRDF, chain throughput, continue.
        float3 nLocal = hit.hitNormal;
        if (dot(nLocal, rayDir) > 0.0) nLocal = -nLocal;
        ONB yBasis = buildONB(nLocal);
        float3 yWo = worldToLocal(yBasis, -rayDir);
        BRDFSample ys = brdfSample(yWo, matY, randFloat3(seed));
        if (!ys.valid || ys.pdf < PDF_EPSILON)
        {
            missed = true;   // treat as black, drop sample
            finished = true;
            break;
        }
        // Delta BSDFs return value already pre-cancelled against |cos|/pdf; for
        // dielectrics value = 1, for delta metals value = baseColor. Just multiply.
        throughput *= ys.value;
        rayOrigin = hit.hitPos;
        rayDir = normalize(localToWorld(yBasis, ys.wi));
    }

    if (!finished || missed)
    {
        writeRhsZero(slot);
        writeAux(slot, neeContrib, float3(0.0), float3(0.0), 0.0, 0.0);
        return;
    }

    if (hitEmitter)
    {
        // MIS between our BSDF-sampled path and the NEE sampler. Only meaningful if
        // the *first* bounce was non-delta (it is — x is non-delta by construction).
        // Subsequent delta specular reflections don't change the MIS weight: NEE and
        // BSDF sampling at x are the two competing strategies at the LHS vertex.
        float3 toY = hit.hitPos - rayOrigin;
        float dist2 = dot(toY, toY);
        float3 nYflip = hit.hitNormal;
        if (dot(nYflip, -rayDir) < 0.0) nYflip = -nYflip;
        float cosLightY = abs(dot(nYflip, -rayDir));

        float totalArea = 0.0;
        for (int j = 0; j < pc.lightCount; j++)
            totalArea += lights[j].area;

        float pdfLightSA = 0.0;
        if (totalArea > 0.0 && cosLightY > COS_EPSILON)
            pdfLightSA = dist2 / (totalArea * cosLightY);

        float w = 1.0;
        if (pdfLightSA > PDF_EPSILON)
            w = powerHeuristic(firstBouncePdf, pdfLightSA);

        float3 emit = matY.emission.rgb * matY.emission.a;
        float3 emitTerm = throughput * emit * w;

        writeRhsZero(slot);
        writeAux(slot, neeContrib, float3(0.0), emitTerm, 0.0, 1.0);
        return;
    }

    // Landed on a non-emissive, non-delta surface — this is the network query point.
    float3 n_y = hit.hitNormal;
    if (dot(n_y, rayDir) > 0.0) n_y = -n_y;
    float3 albedo_y = matY.baseColor.rgb;

    writeRhs(slot, hit.hitPos, n_y, -rayDir, albedo_y);
    writeAux(slot, neeContrib, throughput, float3(0.0), 1.0, 0.0);
}
