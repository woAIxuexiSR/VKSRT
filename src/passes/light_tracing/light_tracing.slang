// Light Tracing shader: emit particles from light sources, splat contributions to camera image
// Single file with raygen, miss, closesthit entry points

import material;
import brdf;
import onb;
import rng;
import math_utils;

// --- Payload: closesthit returns geometry info, raygen does shading ---

[raypayload]
struct RayPayload
{
    int isHit : read(caller) : write(caller, closesthit, miss);
    float3 hitPos : read(caller) : write(closesthit);
    float3 hitNormal : read(caller) : write(closesthit);
    int matIndex : read(caller) : write(closesthit);
};

struct LTCameraProperties
{
    float4x4 viewProj;
    float4x4 viewInverse;
    int screenWidth;
    int screenHeight;
    float tanHalfFov;
    float _pad0;
};

struct PushConstants
{
    int frameIndex;
    int maxDepth;
    int rrDepth;
    int lightCount;
    int photonSize;
};

[[vk::push_constant]]
PushConstants pc;

[[vk::binding(0)]]
RaytracingAccelerationStructure topLevelAS;

[[vk::binding(1)]]
ConstantBuffer<LTCameraProperties> cam;

[[vk::binding(2)]]
StructuredBuffer<float> vertexData;

[[vk::binding(3)]]
StructuredBuffer<uint> indices;

[[vk::binding(4)]]
StructuredBuffer<Material> materials;

[[vk::binding(5)]]
StructuredBuffer<float> normalData;

[[vk::binding(6)]]
StructuredBuffer<float2> texcoords;

struct LightTriangle
{
    float area;
    int indexOffset;
    int vertexOffset;
    int matIndex;
};

[[vk::binding(7)]]
StructuredBuffer<LightTriangle> lights;

[[vk::binding(8)]]
[[vk::image_format("r32f")]]
RWTexture2D<float> splatR;

[[vk::binding(9)]]
[[vk::image_format("r32f")]]
RWTexture2D<float> splatG;

[[vk::binding(10)]]
[[vk::image_format("r32f")]]
RWTexture2D<float> splatB;

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

// --- Camera connection helpers ---

bool projectToScreen(float3 worldPos, out int2 pixel)
{
    pixel = int2(0, 0);

    float4 clipPos = mul(cam.viewProj, float4(worldPos, 1.0));
    if (clipPos.w <= 0.0) return false;

    float3 ndc = clipPos.xyz / clipPos.w;
    if (abs(ndc.x) > 1.0 || abs(ndc.y) > 1.0 || ndc.z < -1.0 || ndc.z > 1.0)
        return false;

    float2 uv = ndc.xy * 0.5 + 0.5;
    pixel = int2(uv * float2(cam.screenWidth, cam.screenHeight));

    if (pixel.x < 0 || pixel.x >= cam.screenWidth ||
        pixel.y < 0 || pixel.y >= cam.screenHeight)
        return false;

    return true;
}

void splatContribution(int2 pixel, float3 contribution)
{
    if (any(isnan(contribution)) || any(isinf(contribution))) return;
    contribution = max(contribution, float3(0.0));

    float origR, origG, origB;
    InterlockedAdd(splatR[pixel], contribution.r, origR);
    InterlockedAdd(splatG[pixel], contribution.g, origG);
    InterlockedAdd(splatB[pixel], contribution.b, origB);
}

// --- Ray Generation: light tracing with camera connection ---

[shader("raygeneration")]
void raygenMain()
{
    uint2 launchID = DispatchRaysIndex().xy;
    uint2 launchSize = DispatchRaysDimensions().xy;

    uint seed = wangHash(uint(launchID.x * 1973u + launchID.y * 9277u + uint(pc.frameIndex) * 26699u) | 1u);

    if (pc.lightCount == 0) return;

    // === 1. Light source sampling ===
    int lightIdx = int(randFloat(seed) * float(pc.lightCount));
    lightIdx = min(lightIdx, pc.lightCount - 1);

    LightTriangle lt = lights[lightIdx];
    uint3 triIdx = getTriIndices(lt.indexOffset);
    float3 lv0 = getVertex(triIdx.x + lt.vertexOffset);
    float3 lv1 = getVertex(triIdx.y + lt.vertexOffset);
    float3 lv2 = getVertex(triIdx.z + lt.vertexOffset);

    float3 lightPos = samplePointOnTriangle(lv0, lv1, lv2, randFloat2(seed));
    float3 lightNormal = normalize(cross(lv1 - lv0, lv2 - lv0));

    // Cosine-weighted hemisphere sampling from light normal
    ONB lightBasis = buildONB(lightNormal);
    float3 localDir = cosineSampleHemisphere(randFloat2(seed));
    float3 emitDir = normalize(localToWorld(lightBasis, localDir));

    Material lightMat = materials[lt.matIndex];
    float3 Le = lightMat.emission.rgb * lightMat.emission.a;

    // Initial throughput: Le * lightCount * area * PI
    // Derivation: Le / (pdf_light * pdf_area * pdf_dir)
    //   = Le / (1/lightCount * 1/area * cosTheta/PI)
    //   = Le * lightCount * area * PI / cosTheta
    // The cosTheta from Lambert emission cancels with 1/cosTheta
    float3 beta = Le * float(pc.lightCount) * lt.area * PI;

    float3 rayPos = lightPos;
    float3 rayDir = emitDir;

    float3 cameraPos = mul(cam.viewInverse, float4(0, 0, 0, 1)).xyz;
    float3 cameraForward = -normalize(mul(cam.viewInverse, float4(0, 0, 1, 0)).xyz);
    float N = float(pc.photonSize * pc.photonSize);

    // === 1b. Light source directly visible to camera ===
    {
        float3 toCamera = cameraPos - lightPos;
        float dist = length(toCamera);
        float3 toCameraDir = toCamera / dist;
        float cosLight = dot(lightNormal, toCameraDir);

        if (cosLight > 1e-4)
        {
            int2 pixel;
            if (projectToScreen(lightPos, pixel))
            {
                RayDesc shadowRay;
                shadowRay.Origin = lightPos;
                shadowRay.Direction = toCameraDir;
                shadowRay.TMin = 0.001;
                shadowRay.TMax = dist - 0.001;

                RayPayload shadowPayload;
                shadowPayload.isHit = 0;
                TraceRay(topLevelAS,
                    RAY_FLAG_FORCE_OPAQUE | RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH,
                    0xff, 0, 0, 0, shadowRay, shadowPayload);

                if (shadowPayload.isHit == 0)
                {
                    float cosCam = dot(cameraForward, -toCameraDir);
                    if (cosCam > 1e-4)
                    {
                        float aspect = float(cam.screenWidth) / float(cam.screenHeight);
                        float We = float(cam.screenWidth * cam.screenHeight)
                                 / (4.0 * cam.tanHalfFov * cam.tanHalfFov * aspect
                                    * cosCam * cosCam * cosCam);

                        // Le * lightCount * area * cosLight * We / (dist^2 * N)
                        // Note: beta includes an extra PI from cosine-weighted direction sampling,
                        // which doesn't apply to the direct visibility path, so divide it out.
                        float3 contribution = beta / PI * cosLight * We / (dist * dist * N);
                        splatContribution(pixel, contribution);
                    }
                }
            }
        }
    }

    // === 2. Bounce loop ===
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

        if (payload.isHit == 0) break;

        Material mat = materials[payload.matIndex];

        if (mat.type == MAT_EMISSIVE) break;

        float3 hitN = normalize(payload.hitNormal);
        if (dot(hitN, rayDir) > 0.0 && !mat.canTransmit())
            hitN = -hitN;

        // === BRDF sampling (needed for both camera connection and next bounce) ===
        ONB basis = buildONB(hitN);
        float3 wo = worldToLocal(basis, -rayDir);

        BRDFSample s = brdfSample(wo, mat, randFloat2(seed));
        if (!s.valid || s.pdf < 1e-6) break;

        // === Camera connection ===
        float3 toCamera = cameraPos - payload.hitPos;
        float dist = length(toCamera);
        float3 toCameraDir = toCamera / dist;
        float cosHit = dot(hitN, toCameraDir);

        float3 bsdfWeight = float3(0.0);
        if (cosHit > 1e-4)
        {
            if (!s.isDelta)
            {
                // Continuous BRDF: eval in camera direction directly
                float3 woCam = worldToLocal(basis, toCameraDir);
                bsdfWeight = brdfEval(wo, woCam, mat);
            }
            else
            {
                // Delta BRDF: approximate with cone around sampled direction
                float3 sampleDirWorld = normalize(localToWorld(basis, s.wi));
                float cosAngle = dot(toCameraDir, sampleDirWorld);
                float coneAngle = max(mat.roughness, 0.01);
                float threshold = cos(coneAngle * PI * 0.5);
                if (cosAngle > threshold)
                {
                    float t = (cosAngle - threshold) / (1.0 - threshold);
                    float coneSA = 2.0 * PI * (1.0 - threshold);
                    bsdfWeight = s.value * t / max(coneSA, 1e-6);
                }
            }
        }

        if (any(bsdfWeight > 0.0))
        {
            int2 pixel;
            if (projectToScreen(payload.hitPos, pixel))
            {
                RayDesc shadowRay;
                shadowRay.Origin = payload.hitPos;
                shadowRay.Direction = toCameraDir;
                shadowRay.TMin = 0.001;
                shadowRay.TMax = dist - 0.001;

                RayPayload shadowPayload;
                shadowPayload.isHit = 0;
                TraceRay(topLevelAS,
                    RAY_FLAG_FORCE_OPAQUE | RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH,
                    0xff, 0, 0, 0, shadowRay, shadowPayload);

                if (shadowPayload.isHit == 0)
                {
                    float cosCam = dot(cameraForward, -toCameraDir);
                    if (cosCam > 1e-4)
                    {
                        float aspect = float(cam.screenWidth) / float(cam.screenHeight);
                        float We = float(cam.screenWidth * cam.screenHeight)
                                 / (4.0 * cam.tanHalfFov * cam.tanHalfFov * aspect
                                    * cosCam * cosCam * cosCam);

                        float3 contribution = beta * bsdfWeight * cosHit * We / (dist * dist * N);
                        splatContribution(pixel, contribution);
                    }
                }
            }
        }

        // === Update throughput and direction for next bounce ===
        if (s.isDelta)
            beta *= s.value;
        else
            beta *= s.value * max(s.wi.z, 0.0) / s.pdf;

        rayPos = payload.hitPos;
        rayDir = normalize(localToWorld(basis, s.wi));

        // Russian Roulette
        if (depth >= pc.rrDepth)
        {
            float p = max(max(beta.r, max(beta.g, beta.b)), 0.05);
            if (randFloat(seed) > p) break;
            beta /= p;
        }
    }
}

// --- Miss ---

[shader("miss")]
void missMain(inout RayPayload payload)
{
    payload.isHit = 0;
}

// --- Closest Hit: return geometry info only ---

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
    float3 hitPos = bary.x * v0 + bary.y * v1 + bary.z * v2;

    float3 n0 = getNormal(i0 + sbt.vertexOffset);
    float3 n1 = getNormal(i1 + sbt.vertexOffset);
    float3 n2 = getNormal(i2 + sbt.vertexOffset);
    float3 N = normalize(bary.x * n0 + bary.y * n1 + bary.z * n2);

    payload.isHit = 1;
    payload.hitPos = hitPos;
    payload.hitNormal = N;
    payload.matIndex = sbt.matIndex;
}
