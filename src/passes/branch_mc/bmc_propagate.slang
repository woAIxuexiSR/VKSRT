// Branch MC: Backward propagation + stylize + jackknife debiasing
// 1D dispatch: [depthVertexCount/256, 1, 1]
// Processes vertices at current depth (backward, from deepest to shallowest):
//   - Leaf vertices: keep radiance as-is (local emission, no stylize)
//   - Non-leaf: traverse firstChild linked list, collect child.throughput * child.radiance * child.mis
//   - Write per-child indirect to debias buffer
//   - Apply stylize g(le + mean(direct + indirect))
//   - Jackknife debiasing if enabled

import bmc_structs;
import wavefront_structs;
import material;
import stylize;

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

[[vk::binding(19)]]
RWStructuredBuffer<float4> debiasDirectBuffer;

[[vk::binding(20)]]
RWStructuredBuffer<float4> debiasIndirectBuffer;

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    uint threadIdx = dispatchThreadID.x;
    if (threadIdx >= uint(pc.depthVertexCount)) return;

    int vtxIdx = pc.depthVertexStart + int(threadIdx);
    if (vtxIdx >= pc.maxVertices) return;

    BMCVertex vtx = vertices[vtxIdx];

    // --- Collect children: compute throughput * radiance * mis for each child ---
    // Leaf vertices (firstChild < 0) fall through with indirectRadiance=0,
    // so stylize(le + 0) = stylize(le) — matching reference behavior.
    int nChildren = 0;
    float3 indirectRadiance = float3(0.0);
    int debiasIdx = vtx.debiasRadianceIdx;

    // Zero out debias indirect buffer entries first
    if (debiasIdx >= 0)
    {
        for (int i = 0; i < vtx.nInnerSamples; i++)
            debiasIndirectBuffer[debiasIdx + i] = float4(0.0);
    }

    int childIdx = vtx.firstChild;
    while (childIdx >= 0 && childIdx < pc.maxVertices)
    {
        BMCVertex child = vertices[childIdx];
        // l = child.throughput * child.radiance * child.mis
        float3 l = child.throughput * child.radiance * child.mis;
        indirectRadiance += l;

        // Write per-child indirect to debias buffer at child's index
        if (debiasIdx >= 0 && child.childIndex < vtx.nInnerSamples)
            debiasIndirectBuffer[debiasIdx + child.childIndex] = float4(l, 0.0);

        nChildren++;
        childIdx = child.nextSibling;
    }

    // Local emission (stored by advance as local emission without throughput)
    float3 le = vtx.radiance;
    int N = vtx.nInnerSamples;
    if (N <= 0) N = max(nChildren, 1);

    // --- No stylization: linear combination ---
    // STYLE_NONE vertices always use shadow rays for NEE (never debias buffer),
    // so le already contains NEE direct lighting from shadow kernel.
    if (vtx.styleType == STYLE_NONE)
    {
        vtx.radiance = le + indirectRadiance / float(N);
        vertices[vtxIdx] = vtx;
        return;
    }

    // --- Debiasing OFF: biased stylize ---
    if (pc.useDebiasing == 0 || debiasIdx < 0)
    {
        float3 li = le + indirectRadiance / float(N);
        vtx.radiance = applyStylize(li, vtx.styleType, vtx.styleParam0);
        vertices[vtxIdx] = vtx;
        return;
    }

    // --- Jackknife debiasing ---
    int baseSamples = N - vtx.debiasingSamples;
    if (baseSamples < 1) baseSamples = 1;

    // Compute base estimator: use only first baseSamples entries
    float3 jkDirectSum = float3(0.0);
    float3 jkIndirectSum = float3(0.0);
    for (int i = 0; i < baseSamples; i++)
    {
        jkDirectSum += debiasDirectBuffer[debiasIdx + i].xyz;
        jkIndirectSum += debiasIndirectBuffer[debiasIdx + i].xyz;
    }
    float3 baseEstimator = applyStylize(
        (jkDirectSum + jkIndirectSum) / float(baseSamples) + le,
        vtx.styleType, vtx.styleParam0);

    // Extend to full estimator: add remaining debiasing samples
    for (int i = baseSamples; i < N; i++)
    {
        jkDirectSum += debiasDirectBuffer[debiasIdx + i].xyz;
        jkIndirectSum += debiasIndirectBuffer[debiasIdx + i].xyz;
    }
    float3 fullEstimator = applyStylize(
        (jkDirectSum + jkIndirectSum) / float(N) + le,
        vtx.styleType, vtx.styleParam0);

    // PMF of geometric distribution: pmf = r * (1 - r)^(K - 1)
    float r = pc.debiasR;
    int K = vtx.debiasingSamples;
    float pmf = r * pow(1.0 - r, float(max(K - 1, 0)));
    if (pmf < 1e-8) pmf = 1e-8; // safety clamp

    // Leave-one-out correction
    float3 outRadiance = baseEstimator;
    for (int removal = 0; removal < N; removal++)
    {
        // Sum all except removal
        float3 looDirectSum = float3(0.0);
        float3 looIndirectSum = float3(0.0);
        for (int j = 0; j < N; j++)
        {
            if (j == removal) continue;
            looDirectSum += debiasDirectBuffer[debiasIdx + j].xyz;
            looIndirectSum += debiasIndirectBuffer[debiasIdx + j].xyz;
        }
        float3 looEstimator = applyStylize(
            (looDirectSum + looIndirectSum) / float(N - 1) + le,
            vtx.styleType, vtx.styleParam0);

        float3 delta = (fullEstimator - looEstimator) / float(N) / pmf;
        outRadiance += delta;
    }

    vtx.radiance = outRadiance;
    vertices[vtxIdx] = vtx;
}
