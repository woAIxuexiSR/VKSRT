// ICV edge shader: one invocation processes right/down undirected pixel edges.
// It scatters Gaussian-weighted MIS control-variate estimates to both endpoints.

import material;
import brdf;
import onb;
import rng;
import math_utils;
import camera;

[raypayload]
struct RayPayload
{
    int isHit : read(caller) : write(caller, closesthit, miss);
    float3 hitPos : read(caller) : write(closesthit);
    float3 hitNormal : read(caller) : write(closesthit);
    int matIndex : read(caller) : write(closesthit);
};

struct PushConstants
{
    int frameIndex;
    int maxDepth;
    int rrDepth;
    int lightCount;
    int useNEE;
    int useMIS;
    float a;
    float gaussianSigma;
    int screenWidth;
    int screenHeight;
    int _pad0;
    int _pad1;
};

[[vk::push_constant]]
PushConstants pc;

[[vk::binding(0)]]
Texture2D<float4> anchorImage;

[[vk::binding(1)]]
RWTexture2D<float4> hEdgeLeft;

[[vk::binding(2)]]
RWTexture2D<float4> hEdgeRight;

[[vk::binding(3)]]
RWTexture2D<float4> vEdgeUp;

[[vk::binding(4)]]
RWTexture2D<float4> vEdgeDown;

[[vk::binding(5)]]
ConstantBuffer<CameraProperties> cam;

[[vk::binding(6)]]
RaytracingAccelerationStructure topLevelAS;

[[vk::binding(7)]]
StructuredBuffer<float> vertexData;

[[vk::binding(8)]]
StructuredBuffer<uint> indices;

[[vk::binding(9)]]
StructuredBuffer<Material> materials;

[[vk::binding(10)]]
StructuredBuffer<float> normalData;

// binding(11) = texcoords, bound for pipeline layout compatibility but unused.

[[vk::binding(12)]]
StructuredBuffer<LightTriangle> lights;

[[vk::binding(13)]]
StructuredBuffer<float3x4> instanceTransforms;

float3 getVertex(uint index)
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

struct HitGroupSBT
{
    int matIndex;
    int vertexOffset;
    int indexOffset;
};

[[vk::shader_record]]
ConstantBuffer<HitGroupSBT> sbt;

struct TraceResult
{
    float3 L;
};

TraceResult tracePath(uint2 pixel, uint2 launchSize, inout uint seed)
{
    float3 rayPos, rayDir;
    generatePrimaryRay(cam, pixel, launchSize, seed, rayPos, rayDir);

    TraceResult result;
    result.L = float3(0.0);

    float3 L = float3(0.0);
    float3 beta = float3(1.0);
    float prevBsdfPdf = 0.0;

    for (int depth = 0; depth < pc.maxDepth; depth++)
    {
        RayDesc ray;
        ray.Origin = rayPos;
        ray.Direction = rayDir;
        ray.TMin = 0.001;
        ray.TMax = 10000.0;

        RayPayload payload;
        payload.isHit = 0;
        TraceRay(topLevelAS, RAY_FLAG_FORCE_OPAQUE, 0xff, 0, 0, 0, ray, payload);

        if (payload.isHit == 0)
            break;

        Material mat = materials[payload.matIndex];

        if (mat.isEmissive())
        {
            float3 emission = mat.emission.rgb * mat.emission.a;
            float misWeight = 1.0;

            if (depth > 0 && pc.useNEE != 0 && pc.lightCount > 0)
            {
                if (prevBsdfPdf <= 0.0)
                {
                    misWeight = 1.0;
                }
                else if (pc.useMIS != 0)
                {
                    float3 hitNormal = normalize(payload.hitNormal);
                    float dist2 = dot(rayPos - payload.hitPos, rayPos - payload.hitPos);
                    float cosLight = abs(dot(hitNormal, normalize(rayPos - payload.hitPos)));
                    float totalArea = 0.0;
                    for (int li = 0; li < pc.lightCount; li++)
                        totalArea += lights[li].area;
                    if (cosLight > COS_EPSILON && totalArea > 0.0)
                    {
                        float pdfLight = dist2 / (totalArea * cosLight);
                        misWeight = powerHeuristic(prevBsdfPdf, pdfLight);
                    }
                    else
                        misWeight = 0.0;
                }
                else
                {
                    misWeight = 0.0;
                }
            }

            L += beta * emission * misWeight;
            break;
        }

        float3 shadingN = payload.hitNormal;
        if (!mat.canTransmit() && dot(shadingN, rayDir) > 0.0)
            shadingN = -shadingN;

        if (pc.useNEE != 0 && pc.lightCount > 0 && !mat.isDeltaBRDF())
        {
            int lightIdx = int(randFloat(seed) * float(pc.lightCount));
            lightIdx = min(lightIdx, pc.lightCount - 1);

            LightTriangle lt = lights[lightIdx];
            uint3 triIdx = getTriIndices(lt.indexOffset);
            float3x4 lightXform = instanceTransforms[lt.instanceIndex];
            float3 lv0 = transformPosition(lightXform, getVertex(triIdx.x + lt.vertexOffset));
            float3 lv1 = transformPosition(lightXform, getVertex(triIdx.y + lt.vertexOffset));
            float3 lv2 = transformPosition(lightXform, getVertex(triIdx.z + lt.vertexOffset));

            float3 lightPoint = samplePointOnTriangle(lv0, lv1, lv2, randFloat2(seed));
            float3 lightNormal = normalize(cross(lv1 - lv0, lv2 - lv0));

            float3 toLight = lightPoint - payload.hitPos;
            float dist = length(toLight);
            float3 lightDir = toLight / dist;

            float cosHit = dot(shadingN, lightDir);
            float cosLight = abs(dot(lightNormal, -lightDir));

            if (cosHit > COS_EPSILON && cosLight > COS_EPSILON)
            {
                RayDesc shadowRay;
                shadowRay.Origin = payload.hitPos;
                shadowRay.Direction = lightDir;
                shadowRay.TMin = 0.001;
                shadowRay.TMax = dist - 0.001;

                RayPayload shadowPayload;
                shadowPayload.isHit = 0;
                TraceRay(topLevelAS,
                         RAY_FLAG_FORCE_OPAQUE | RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH,
                         0xff, 0, 0, 0, shadowRay, shadowPayload);

                if (shadowPayload.isHit == 0)
                {
                    Material lightMat = materials[lt.matIndex];
                    float3 lightEmission = lightMat.emission.rgb * lightMat.emission.a;
                    float pdfLight = (dist * dist) / (lt.area * cosLight * float(pc.lightCount));

                    ONB neeBasis = buildONB(shadingN);
                    float3 wi = worldToLocal(neeBasis, lightDir);
                    float3 wo_nee = worldToLocal(neeBasis, -rayDir);
                    float3 bsdfVal = brdfEval(wi, wo_nee, mat);

                    float w = 1.0;
                    if (pc.useMIS != 0)
                    {
                        float pdfBsdf = brdfPdf(wi, wo_nee, mat);
                        w = powerHeuristic(pdfLight, pdfBsdf);
                    }

                    L += beta * bsdfVal * lightEmission * cosHit * w / pdfLight;
                }
            }
        }

        ONB basis = buildONB(shadingN);
        float3 wo = worldToLocal(basis, -rayDir);
        BRDFSample s = brdfSample(wo, mat, randFloat3(seed));

        if (!s.valid || s.pdf < PDF_EPSILON)
            break;

        prevBsdfPdf = s.isDelta ? 0.0 : s.pdf;

        if (s.isDelta)
            beta *= s.value;
        else
            beta *= s.value * abs(s.wi.z) / s.pdf;

        rayPos = payload.hitPos;
        rayDir = normalize(localToWorld(basis, s.wi));

        if (depth >= pc.rrDepth)
        {
            float p = max(max(beta.r, max(beta.g, beta.b)), 0.05);
            if (randFloat(seed) > p)
                break;
            beta /= p;
        }
    }

    result.L = L;
    return result;
}

float gaussianWeight(float2 delta)
{
    float sigma = max(pc.gaussianSigma, 1e-3);
    return exp(-dot(delta, delta) / (2.0 * sigma * sigma));
}

float3 sanitizeContribution(float3 value, float3 fallback)
{
    if (any(isnan(value)) || any(isinf(value)))
        return fallback;
    return value;
}

void processEdge(uint2 i, int2 offset, uint2 launchSize)
{
    int2 ip = int2(i);
    int2 jp = ip + offset;
    if (jp.x < 0 || jp.y < 0 || jp.x >= pc.screenWidth || jp.y >= pc.screenHeight)
        return;

    uint2 j = uint2(jp);
    float3 anchorI = anchorImage[ip].rgb;
    float3 anchorJ = anchorImage[jp].rgb;

    // Use symmetric pair weights. The anchor image alpha channel is not treated
    // as a PDF/selection-probability image for this estimator.
    const float wij = 0.5;
    const float wji = 0.5;

    uint seedI = seedFromPixel(i, uint(pc.frameIndex));
    uint seedJ = seedFromPixel(j, uint(pc.frameIndex));
    TraceResult shiftedJFromI = tracePath(j, launchSize, seedI);
    TraceResult shiftedIFromJ = tracePath(i, launchSize, seedJ);

    float3 estimateI =
        wij * (pc.a * anchorJ + anchorI - pc.a * shiftedJFromI.L) +
        wji * (pc.a * anchorJ + shiftedIFromJ.L - pc.a * anchorJ);

    float3 estimateJ =
        wij * (pc.a * anchorI + shiftedJFromI.L - pc.a * anchorI) +
        wji * (pc.a * anchorI + anchorJ - pc.a * shiftedIFromJ.L);

    estimateI = sanitizeContribution(estimateI, anchorI);
    estimateJ = sanitizeContribution(estimateJ, anchorJ);

    float weight = gaussianWeight(float2(offset));
    float4 outI = float4(estimateI, weight);
    float4 outJ = float4(estimateJ, weight);

    if (offset.x != 0)
    {
        hEdgeLeft[ip] = outI;
        hEdgeRight[ip] = outJ;
    }
    else
    {
        vEdgeUp[ip] = outI;
        vEdgeDown[ip] = outJ;
    }
}

[shader("raygeneration")]
void raygenMain()
{
    uint2 launchID = DispatchRaysIndex().xy;
    uint2 launchSize = DispatchRaysDimensions().xy;

    if (int(launchID.x) >= pc.screenWidth || int(launchID.y) >= pc.screenHeight)
        return;

    processEdge(launchID, int2(1, 0), launchSize);
    processEdge(launchID, int2(0, 1), launchSize);
}

[shader("miss")]
void missMain(inout RayPayload payload)
{
    payload.isHit = 0;
}

[shader("closesthit")]
void closestHitMain(inout RayPayload payload, BuiltInTriangleIntersectionAttributes attribs)
{
    float3 bary = float3(1.0 - attribs.barycentrics.x - attribs.barycentrics.y,
                         attribs.barycentrics.x, attribs.barycentrics.y);

    uint triIdx = PrimitiveIndex();
    uint i0 = indices[(sbt.indexOffset + triIdx) * 3 + 0];
    uint i1 = indices[(sbt.indexOffset + triIdx) * 3 + 1];
    uint i2 = indices[(sbt.indexOffset + triIdx) * 3 + 2];

    float3 v0 = getVertex(i0 + sbt.vertexOffset);
    float3 v1 = getVertex(i1 + sbt.vertexOffset);
    float3 v2 = getVertex(i2 + sbt.vertexOffset);
    float3 objPos = bary.x * v0 + bary.y * v1 + bary.z * v2;

    float3 n0 = getNormal(i0 + sbt.vertexOffset);
    float3 n1 = getNormal(i1 + sbt.vertexOffset);
    float3 n2 = getNormal(i2 + sbt.vertexOffset);
    float3 objN = normalize(bary.x * n0 + bary.y * n1 + bary.z * n2);

    payload.isHit = 1;
    payload.hitPos = transformPosition(ObjectToWorld3x4(), objPos);
    payload.hitNormal = transformNormal(WorldToObject3x4(), objN);
    payload.matIndex = sbt.matIndex;
}
