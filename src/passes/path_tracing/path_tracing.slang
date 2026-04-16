// Path tracing shader: iterative multi-bounce with BRDF sampling and Russian Roulette
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
RWTexture2D<float4> gbufferNormal;

[[vk::binding(11)]]
RWTexture2D<float4> gbufferPosition;

[[vk::binding(12)]]
RWTexture2D<float4> gbufferAlbedo;

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

// --- Ray Generation: iterative path tracing ---

[shader("raygeneration")]
void raygenMain()
{
    uint2 launchID = DispatchRaysIndex().xy;
    uint2 launchSize = DispatchRaysDimensions().xy;

    uint seed = wangHash(uint(launchID.x * 1973u + launchID.y * 9277u + uint(pc.frameIndex) * 26699u) | 1u);

    // Pixel jitter for anti-aliasing
    float2 jitter = randFloat2(seed);
    float2 pixelCenter = float2(launchID) + jitter;
    float2 inUV = pixelCenter / float2(launchSize);
    float2 d = inUV * 2.0 - 1.0;

    float4 origin = mul(cam.viewInverse, float4(0, 0, 0, 1));
    float4 target = mul(cam.projInverse, float4(d.x, d.y, 1, 1));
    float4 direction = mul(cam.viewInverse, float4(normalize(target.xyz), 0));

    float3 rayPos = origin.xyz;
    float3 rayDir = normalize(direction.xyz);

    float3 L = float3(0.0);
    float3 beta = float3(1.0);
    float prevBsdfPdf = 0.0; // tracks BRDF pdf for MIS when hitting emissive

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

        // Miss: terminate
        if (payload.isHit == 0)
        {
            if (depth == 0)
            {
                gbufferNormal[launchID] = float4(0.0);
                gbufferPosition[launchID] = float4(0.0);
                gbufferAlbedo[launchID] = float4(0.0);
            }
            break;
        }

        Material mat = materials[payload.matIndex];

        // Write G-buffer on first hit (alpha=1 marks valid geometry)
        if (depth == 0)
        {
            float3 N = normalize(payload.hitNormal);
            if (dot(N, rayDir) > 0.0) N = -N;
            gbufferNormal[launchID] = float4(N, 1.0);
            gbufferPosition[launchID] = float4(payload.hitPos, 1.0);
            gbufferAlbedo[launchID] = float4(mat.baseColor.rgb, 1.0);
        }

        // Hit emissive surface: add emission with MIS weight
        if (mat.isEmissive())
        {
            float3 emission = mat.emission.rgb * mat.emission.a;
            float misWeight = 1.0;

            if (depth > 0 && pc.useNEE != 0 && pc.lightCount > 0)
            {
                if (prevBsdfPdf <= 0.0)
                {
                    // Delta BRDF: NEE cannot sample this direction, full weight to BRDF path
                    misWeight = 1.0;
                }
                else if (pc.useMIS != 0)
                {
                    // Compute light sampling PDF for this hit
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
                {
                    // NEE on but MIS off: BRDF path should not count emissive hits
                    misWeight = 0.0;
                }
            }

            L += beta * emission * misWeight;
            break;
        }

        // Determine shading normal: flip to face ray unless material can transmit
        float3 shadingN = payload.hitNormal;
        if (!mat.canTransmit())
        {
            if (dot(shadingN, rayDir) > 0.0)
                shadingN = -shadingN;
        }

        // --- NEE: Direct light sampling ---
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

                    L += beta * bsdfVal * lightEmission * cosHit * w / pdfLight;
                }
            }
        }

        // --- BRDF importance sampling for next bounce ---
        ONB basis = buildONB(shadingN);
        float3 wo = worldToLocal(basis, -rayDir);

        BRDFSample s = brdfSample(wo, mat, randFloat3(seed));

        if (!s.valid || s.pdf < 1e-6)
            break;

        // Track BRDF pdf for MIS on next emissive hit
        prevBsdfPdf = s.isDelta ? 0.0 : s.pdf;

        // Update throughput
        if (s.isDelta)
            beta *= s.value;
        else
            beta *= s.value * abs(s.wi.z) / s.pdf;

        // Set up next ray
        rayPos = payload.hitPos;
        rayDir = normalize(localToWorld(basis, s.wi));

        // Russian Roulette
        if (depth >= pc.rrDepth)
        {
            float p = max(max(beta.r, max(beta.g, beta.b)), 0.05);
            if (randFloat(seed) > p)
                break;
            beta /= p;
        }
    }

    image[launchID] = float4(L, 1.0);
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

    // Interpolate object-space position and transform to world space
    float3 v0 = getVertex(i0 + sbt.vertexOffset);
    float3 v1 = getVertex(i1 + sbt.vertexOffset);
    float3 v2 = getVertex(i2 + sbt.vertexOffset);
    float3 objPos = bary.x * v0 + bary.y * v1 + bary.z * v2;

    // Interpolate object-space normal and transform to world space
    float3 n0 = getNormal(i0 + sbt.vertexOffset);
    float3 n1 = getNormal(i1 + sbt.vertexOffset);
    float3 n2 = getNormal(i2 + sbt.vertexOffset);
    float3 objN = normalize(bary.x * n0 + bary.y * n1 + bary.z * n2);

    payload.isHit = 1;
    payload.hitPos = transformPosition(ObjectToWorld3x4(), objPos);
    payload.hitNormal = transformNormal(WorldToObject3x4(), objN);
    payload.matIndex = sbt.matIndex;
}
