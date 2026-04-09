// Ray tracing shaders: raygen, miss, closest hit
// Compiled as single file with multiple entry points via slangc

[raypayload]
struct RayPayload
{
    int isHit : read(caller) : write(caller, closesthit, miss);
    float3 color : read(caller) : write(caller, closesthit, miss);
};

struct CameraProperties
{
    int frameIndex;
    int _pad0;
    int _pad1;
    int _pad2;
    float4x4 viewInverse;
    float4x4 projInverse;
};

struct Material
{
    float4 baseColor;
    float4 emission;
    float metallic;
    float roughness;
    float ior;
    int type;
    int diffuseTexIndex;
    int _pad0;
    int _pad1;
    int _pad2;
};

[[vk::binding(0)]]
RWTexture2D<float4> image;

[[vk::binding(1)]]
ConstantBuffer<CameraProperties> cam;

[[vk::binding(2)]]
RaytracingAccelerationStructure topLevelAS;

[[vk::binding(3)]]
StructuredBuffer<float3> vertices;

[[vk::binding(4)]]
StructuredBuffer<uint> indices;

[[vk::binding(5)]]
StructuredBuffer<Material> materials;

[[vk::binding(6)]]
StructuredBuffer<float3> normals;

[[vk::binding(7)]]
StructuredBuffer<float2> texcoords;

struct HitGroupSBT
{
    int matIndex;
    int vertexOffset;
    int indexOffset;
};

[[vk::shader_record]]
ConstantBuffer<HitGroupSBT> sbt;

// --- Ray Generation ---

[shader("raygeneration")]
void raygenMain()
{
    uint2 launchID = DispatchRaysIndex().xy;
    uint2 launchSize = DispatchRaysDimensions().xy;

    float2 pixelCenter = float2(launchID) + float2(0.5, 0.5);
    float2 inUV = pixelCenter / float2(launchSize);
    float2 d = inUV * 2.0 - 1.0;

    float4 origin = mul(cam.viewInverse, float4(0, 0, 0, 1));
    float4 target = mul(cam.projInverse, float4(d.x, d.y, 1, 1));
    float4 direction = mul(cam.viewInverse, float4(normalize(target.xyz), 0));

    RayDesc ray;
    ray.Origin = origin.xyz;
    ray.Direction = normalize(direction.xyz);
    ray.TMin = 0.001;
    ray.TMax = 10000.0;

    RayPayload payload;
    payload.isHit = 0;
    payload.color = float3(0.0, 0.0, 0.0);

    TraceRay(topLevelAS, RAY_FLAG_FORCE_OPAQUE, 0xff, 0, 0, 0, ray, payload);

    float3 result = payload.color;

    // Progressive accumulation
    float4 prevColor = float4(0.0, 0.0, 0.0, 0.0);
    if (cam.frameIndex > 0)
        prevColor = image[launchID];
    float3 finalColor = lerp(prevColor.rgb, result, 1.0 / float(cam.frameIndex + 1));
    image[launchID] = float4(finalColor, 1.0);
}

// --- Miss ---

[shader("miss")]
void missMain(inout RayPayload payload)
{
    payload.isHit = 0;
    payload.color = float3(0.2, 0.2, 0.2);
}

// --- Closest Hit ---

[shader("closesthit")]
void closestHitMain(inout RayPayload payload, BuiltInTriangleIntersectionAttributes attribs)
{
    Material mat = materials[sbt.matIndex];

    // Interpolate normal using barycentric coordinates
    float3 bary = float3(1.0 - attribs.barycentrics.x - attribs.barycentrics.y,
                         attribs.barycentrics.x, attribs.barycentrics.y);

    uint triIdx = PrimitiveIndex();
    uint i0 = indices[(sbt.indexOffset + triIdx) * 3 + 0];
    uint i1 = indices[(sbt.indexOffset + triIdx) * 3 + 1];
    uint i2 = indices[(sbt.indexOffset + triIdx) * 3 + 2];

    float3 n0 = normals[i0 + sbt.vertexOffset];
    float3 n1 = normals[i1 + sbt.vertexOffset];
    float3 n2 = normals[i2 + sbt.vertexOffset];
    float3 N = normalize(bary.x * n0 + bary.y * n1 + bary.z * n2);

    float3 albedo = mat.baseColor.rgb;

    // Emissive materials emit directly
    if (mat.type == 3)
    {
        payload.isHit = 1;
        payload.color = mat.emission.rgb * mat.emission.a;
        return;
    }

    // Basic diffuse shading with a simple directional light + ambient
    float3 lightDir = normalize(float3(0.5, 1.0, -0.3));
    float NdotL = max(dot(N, lightDir), 0.0);
    float3 diffuse = albedo * (0.3 + 0.7 * NdotL);

    payload.isHit = 1;
    payload.color = diffuse;
}
