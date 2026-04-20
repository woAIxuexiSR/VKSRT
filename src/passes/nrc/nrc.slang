// Neural Radiance Cache: path tracing with NRC data collection
// raygen traces paths, collects training data for a random subset,
// and records query features at the NRC query depth for all pixels.

import material;
import brdf;
import onb;
import rng;
import math_utils;

[raypayload]
struct RayPayload
{
    int isHit : read(caller) : write(caller, closesthit, miss);
    float3 hitPos : read(caller) : write(closesthit);
    float3 hitNormal : read(caller) : write(closesthit);
    int matIndex : read(caller) : write(closesthit);
};

struct CameraProperties
{
    float4x4 viewInverse;
    float4x4 projInverse;
};

struct PushConstants
{
    int frameIndex;
    int maxDepth;
    int rrDepth;
    int lightCount;
    int useNEE;
    int useMIS;
    int nrcQueryDepth;
    int trainFraction;
    int screenWidth;
    int screenHeight;
    int useNRC;
    uint64_t trainInputAddr;
    uint64_t trainGTAddr;
    uint64_t trainCounterAddr;
    uint64_t queryInputAddr;
};

[[vk::push_constant]]
PushConstants pc;

[[vk::binding(0)]]
RWTexture2D<float4> image;

[[vk::binding(1)]]
ConstantBuffer<CameraProperties> cam;

[[vk::binding(2)]]
RaytracingAccelerationStructure topLevelAS;

[[vk::binding(3)]]
StructuredBuffer<float> vertexData;

[[vk::binding(4)]]
StructuredBuffer<uint> indices;

[[vk::binding(5)]]
StructuredBuffer<Material> materials;

[[vk::binding(6)]]
StructuredBuffer<float> normalData;

[[vk::binding(7)]]
StructuredBuffer<float2> texcoords;

[[vk::binding(8)]]
StructuredBuffer<LightTriangle> lights;

[[vk::binding(9)]]
StructuredBuffer<float3x4> instanceTransforms;

[[vk::binding(10)]]
RWTexture2D<float4> shortPathImage;

[[vk::binding(11)]]
RWTexture2D<float4> throughputImage;

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

void writeQueryFeatures(uint pixelIndex, float3 pos, float3 normal, float3 dir, float3 albedo)
{
    float* q = (float*)pc.queryInputAddr;
    uint base = pixelIndex * 12;
    q[base + 0] = pos.x;
    q[base + 1] = pos.y;
    q[base + 2] = pos.z;
    q[base + 3] = normal.x;
    q[base + 4] = normal.y;
    q[base + 5] = normal.z;
    q[base + 6] = dir.x;
    q[base + 7] = dir.y;
    q[base + 8] = dir.z;
    q[base + 9] = albedo.x;
    q[base + 10] = albedo.y;
    q[base + 11] = albedo.z;
}

[shader("raygeneration")]
void raygenMain()
{
    uint2 launchID = DispatchRaysIndex().xy;
    uint2 launchSize = DispatchRaysDimensions().xy;
    uint pixelIndex = launchID.y * launchSize.x + launchID.x;

    uint seed = wangHash(uint(launchID.x * 1973u + launchID.y * 9277u + uint(pc.frameIndex) * 26699u) | 1u);

    bool nrcEnabled = pc.useNRC != 0;

    // Training pixel selection (separate hash to avoid correlation)
    uint trainHash = wangHash(pixelIndex ^ uint(pc.frameIndex * 7919u));
    bool isTraining = nrcEnabled && (trainHash % uint(pc.trainFraction)) == 0;

    // Camera ray
    float2 jitter = randFloat2(seed);
    float2 pixelCenter = float2(launchID) + jitter;
    float2 inUV = pixelCenter / float2(launchSize);
    float2 d = inUV * 2.0 - 1.0;

    float4 origin = mul(cam.viewInverse, float4(0, 0, 0, 1));
    float4 target = mul(cam.projInverse, float4(d.x, d.y, 1, 1));
    float4 direction = mul(cam.viewInverse, float4(normalize(target.xyz), 0));

    float3 rayPos = origin.xyz;
    float3 rayDir = normalize(direction.xyz);

    float3 shortPathL = float3(0.0);
    float3 tailL = float3(0.0);
    float3 beta = float3(1.0);
    float3 tailBeta = float3(1.0);
    float3 queryBeta = float3(0.0);
    float prevBsdfPdf = 0.0;
    bool reachedQueryDepth = false;

    float3 queryPos = float3(0.0);
    float3 queryNormal = float3(0.0);
    float3 queryDir = float3(0.0);
    float3 queryAlbedo = float3(0.0);

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

        // Emissive hit
        if (mat.isEmissive())
        {
            float3 emission = mat.emission.rgb * mat.emission.a;
            float misWeight = 1.0;

            if (depth > 0 && pc.useNEE != 0 && pc.lightCount > 0)
            {
                if (prevBsdfPdf <= 0.0)
                    misWeight = 1.0;
                else if (pc.useMIS != 0)
                {
                    float3 hitNormal = normalize(payload.hitNormal);
                    float dist2 = dot(rayPos - payload.hitPos, rayPos - payload.hitPos);
                    float cosLight = abs(dot(hitNormal, normalize(rayPos - payload.hitPos)));
                    float totalArea = 0.0;
                    for (int li = 0; li < pc.lightCount; li++)
                        totalArea += lights[li].area;
                    if (cosLight > 1e-4 && totalArea > 0.0)
                    {
                        float pdfLight = dist2 / (totalArea * cosLight);
                        misWeight = powerHeuristic(prevBsdfPdf, pdfLight);
                    }
                    else
                        misWeight = 0.0;
                }
                else
                    misWeight = 0.0;
            }

            if (!reachedQueryDepth)
                shortPathL += beta * emission * misWeight;
            else
                tailL += tailBeta * emission * misWeight;
            break;
        }

        // At or beyond NRC query depth on a non-delta surface: query NRC
        if (nrcEnabled && depth >= pc.nrcQueryDepth && !reachedQueryDepth && !mat.isDeltaBRDF())
        {
            reachedQueryDepth = true;
            float3 N = normalize(payload.hitNormal);
            if (dot(N, rayDir) > 0.0) N = -N;

            queryPos = payload.hitPos;
            queryNormal = N;
            queryDir = rayDir;
            queryAlbedo = mat.baseColor.rgb;
            queryBeta = beta;
            tailBeta = float3(1.0);

            if (!isTraining)
                break;
        }

        // Shading normal
        float3 shadingN = payload.hitNormal;
        if (!mat.canTransmit())
        {
            if (dot(shadingN, rayDir) > 0.0)
                shadingN = -shadingN;
        }

        // NEE
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

            if (cosHit > 1e-4 && cosLight > 1e-4)
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

                    float3 neeContrib = bsdfVal * lightEmission * cosHit * w / pdfLight;

                    if (!reachedQueryDepth)
                        shortPathL += beta * neeContrib;
                    else
                        tailL += tailBeta * neeContrib;
                }
            }
        }

        // BRDF sampling
        ONB basis = buildONB(shadingN);
        float3 wo = worldToLocal(basis, -rayDir);

        BRDFSample s = brdfSample(wo, mat, randFloat3(seed));

        if (!s.valid || s.pdf < 1e-6)
            break;

        prevBsdfPdf = s.isDelta ? 0.0 : s.pdf;

        float3 brdfWeight;
        if (s.isDelta)
            brdfWeight = s.value;
        else
            brdfWeight = s.value * abs(s.wi.z) / s.pdf;

        beta *= brdfWeight;
        if (reachedQueryDepth)
            tailBeta *= brdfWeight;

        rayPos = payload.hitPos;
        rayDir = normalize(localToWorld(basis, s.wi));

        // Russian Roulette
        if (depth >= pc.rrDepth)
        {
            float p = max(max(beta.r, max(beta.g, beta.b)), 0.05);
            if (randFloat(seed) > p)
                break;
            beta /= p;
            if (reachedQueryDepth)
                tailBeta /= p;
        }
    }

    // Write per-pixel intermediate images
    shortPathImage[launchID] = float4(shortPathL, reachedQueryDepth ? 1.0 : 0.0);
    throughputImage[launchID] = float4(queryBeta, 0.0);

    // Write query features for inference
    if (reachedQueryDepth)
    {
        writeQueryFeatures(pixelIndex, queryPos, queryNormal, queryDir, queryAlbedo);
    }

    // Write training data
    if (isTraining && reachedQueryDepth)
    {
        uint* counter = (uint*)pc.trainCounterAddr;
        uint slot;
        InterlockedAdd(*counter, 1u, slot);

        uint trainBatchSize = uint(pc.screenWidth * pc.screenHeight) / uint(pc.trainFraction);
        if (slot < trainBatchSize)
        {
            float* tInput = (float*)pc.trainInputAddr;
            float* tGT = (float*)pc.trainGTAddr;

            uint fbase = slot * 12;
            tInput[fbase + 0] = queryPos.x;
            tInput[fbase + 1] = queryPos.y;
            tInput[fbase + 2] = queryPos.z;
            tInput[fbase + 3] = queryNormal.x;
            tInput[fbase + 4] = queryNormal.y;
            tInput[fbase + 5] = queryNormal.z;
            tInput[fbase + 6] = queryDir.x;
            tInput[fbase + 7] = queryDir.y;
            tInput[fbase + 8] = queryDir.z;
            tInput[fbase + 9] = queryAlbedo.x;
            tInput[fbase + 10] = queryAlbedo.y;
            tInput[fbase + 11] = queryAlbedo.z;

            uint gbase = slot * 3;
            tGT[gbase + 0] = tailL.x;
            tGT[gbase + 1] = tailL.y;
            tGT[gbase + 2] = tailL.z;
        }
    }
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
