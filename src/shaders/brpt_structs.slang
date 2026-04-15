// Shared struct definitions for Branch PT kernels (tree-based branching approach)
// Layout must match C++ structs in branch_pt_pass.h
module brpt_structs;

// Import shared types from wavefront_structs (MeshInfo, etc.)
import wavefront_structs;

// BrPTVertex: 128 bytes per vertex in the path tree
public struct BrPTVertex
{
    public float3 throughput;        // 12  BSDF*cos/pdf for this edge
    public uint   seed;              // 4   RNG state
    public float3 radiance;          // 12  accumulated radiance at this vertex
    public int    pixelIndex;        // 4   pixel index in full screen
    public float3 wo;                // 12  outgoing direction (toward camera)
    public int    depth;             // 4
    public int    matIndex;          // 4   material index at this hit
    public int    firstChild;        // 4   head of child linked list (-1 = leaf)
    public int    nextSibling;       // 4   next sibling (-1 = end)
    public int    parent;            // 4   parent vertex index (-1 = root)
    public int    isHit;             // 4   1 = valid hit, 0 = miss
    public int    nInnerSamples;     // 4   branching factor at this vertex
    public int    childIndex;        // 4   which child am I (0..N-1) for debiasing
    public int    debiasRadianceIdx; // 4   index into debiasBuffer (-1 = none)
    public float3 hitPos;            // 12  world position of hit
    public float  prevBsdfPdf;       // 4   for MIS
    public float3 hitNormal;         // 12  interpolated normal
    public int    styleType;         // 4   cached from material
    public float  styleParam0;       // 4   cached from material
    public int    debiasingSamples;   // 4  extra samples from Geometric(r) distribution
    public int    actualBranchingDepth; // 4  branching depth (excludes specular bounces)
    public float  mis;               // 4  MIS weight for emission (applied by parent)
};
// Total: 128 bytes
