// Shared struct definitions for wavefront path tracing kernels
// Layout must match C++ structs in wavefront_pt_pass.h
module wavefront_structs;

public struct PathState
{
    public float3 throughput;
    public uint   seed;
    public float3 L;
    public int    pixelIndex;
    public float  prevBsdfPdf;
    public int    depth;
    public int    alive;
    public int    _pad;
};

public struct RayData
{
    public float3 origin;
    public float  tMin;
    public float3 direction;
    public float  tMax;
};

public struct HitInfo
{
    public float3 hitPos;
    public int    matIndex;
    public float3 hitNormal;
    public int    isHit;
};

public struct ShadowRay
{
    public float3 origin;
    public float  tMax;
    public float3 direction;
    public int    pathIndex;
    public float3 contribution;
    public float  _pad;
};

public struct MeshInfo
{
    public int matIndex;
    public int vertexOffset;
    public int indexOffset;
};
