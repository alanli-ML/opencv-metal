#ifndef OPENCV_METALIMGPROC_GRAPHCUT_KERNELS_HPP
#define OPENCV_METALIMGPROC_GRAPHCUT_KERNELS_HPP

// This file is auto-generated. Do not edit.
// It contains the Metal kernel source code for the GrabCut implementation.

namespace {

// -------------------------------------------------------------------------
// Common Metal helpers for atomic operations
// -------------------------------------------------------------------------
static const char* kGraphCutCommonSrc = R"(
#include <metal_stdlib>
using namespace metal;

#ifndef GRAPHCUT_COMMON_GUARD
#define GRAPHCUT_COMMON_GUARD

struct NodeDataAtom {
    atomic_uint excessBits; // float bits
    atomic_int  label;
};

// Bidirectional residual graph with 8 neighbours.
// Directions: 0:W, 1:NW, 2:N, 3:NE, 4:E, 5:SE, 6:S, 7:SW
// Note that e.g. capacity to W is stored at index 0, while capacity from W
// is stored at index 4 (i.e. W's capacity to E).
struct ResidualGraphAtom
{
    atomic_uint c[8];
};

struct TerminalFlow {
    atomic_uint to_source;
    atomic_uint to_sink;
};

inline float fload(const device atomic_uint* p) {
    return as_type<float>(atomic_load_explicit(p, memory_order_relaxed));
}

inline void fstore(device atomic_uint* p, float v) {
    atomic_store_explicit(p, as_type<uint>(v), memory_order_relaxed);
}

inline float fadd(device atomic_uint* p, float v) {
    return as_type<float>(atomic_fetch_add_explicit(p, as_type<uint>(v), memory_order_relaxed));
}

inline float fsub(device atomic_uint* p, float v) {
    return as_type<float>(atomic_fetch_sub_explicit(p, as_type<uint>(v), memory_order_relaxed));
}

#endif // GRAPHCUT_COMMON_GUARD

)";

// Kernel to build graph into atomic buffers
static const char* kGraphCutBuildAtomSrc = R"(
#include <metal_stdlib>
using namespace metal;

// Bring in common helpers via include string concatenation

kernel void buildGraphAtomKernel(texture2d<float, access::read> bgTerm [[texture(0)]],
                             texture2d<float, access::read> fgTerm [[texture(1)]],
                             texture2d<float, access::read> wLeft [[texture(2)]],
                             texture2d<float, access::read> wTop [[texture(4)]],
                             texture2d<float, access::read> wTL  [[texture(3)]],
                             texture2d<float, access::read> wTR  [[texture(5)]],
                             texture2d<uint, access::read> mask [[texture(6)]],
                             device NodeDataAtom* nodeAtom [[buffer(0)]],
                             device TerminalFlow* termBuf [[buffer(1)]],
                             device ResidualGraphAtom* resAtom [[buffer(2)]],
                             constant float& lambda [[buffer(3)]],
                             uint2 gid [[thread_position_in_grid]])
{
    uint width = bgTerm.get_width();
    uint height = bgTerm.get_height();

    if (gid.x >= width || gid.y >= height) return;

    uint idx = gid.y * width + gid.x;

    // Handle hard constraints from the mask
    uint maskVal = mask.read(gid).x;

    float unary_bg_cost = 0.0f; // Cost for this pixel being background
    float unary_fg_cost = 0.0f; // Cost for this pixel being foreground

    float bg_from_tex = bgTerm.read(gid).x;
    float fg_from_tex = fgTerm.read(gid).x;

    // GrabCut mask values: GC_BGD=0, GC_FGD=1, GC_PR_BGD=2, GC_PR_FGD=3
    if (maskVal == 0) { // Sure Background (GC_BGD)
        // Force background: no cost for BG, infinite cost for FG
        unary_bg_cost = 0.0f;
        unary_fg_cost = lambda;
    } else if (maskVal == 1) { // Sure Foreground (GC_FGD)
        // Force foreground: infinite cost for BG, no cost for FG
        unary_bg_cost = lambda;
        unary_fg_cost = 0.0f;
    } else { // Probable BG/FG (GC_PR_BGD=2, GC_PR_FGD=3)
        // Use GMM-derived costs
        unary_bg_cost = bg_from_tex;
        unary_fg_cost = fg_from_tex;
    }

    // New graph cut mapping: Source=Foreground, Sink=Background
    // - capacity from source to pixel = unary_bg_cost (penalty for pixel being BG)
    // - capacity from pixel to sink = unary_fg_cost (penalty for pixel being FG)
    // - initial excess = capacity from source = unary_bg_cost
    fstore(&nodeAtom[idx].excessBits, unary_bg_cost);
    atomic_store_explicit(&nodeAtom[idx].label, 0, memory_order_relaxed);

    // Terminal edge capacities
    fstore(&termBuf[idx].to_source, unary_bg_cost);
    fstore(&termBuf[idx].to_sink, unary_fg_cost);

    // Directions: 0:W, 1:NW, 2:N, 3:NE, 4:E, 5:SE, 6:S, 7:SW
    // Symmetrically initialize residual graph using the 4 provided weight maps.
    // e.g. cap(x,y -> E) == cap(x+1,y -> W) == wLeft(x+1,y)

    fstore(&resAtom[idx].c[0], wLeft.read(gid).x); // W
    fstore(&resAtom[idx].c[1], wTL.read(gid).x);   // NW
    fstore(&resAtom[idx].c[2], wTop.read(gid).x);  // N
    fstore(&resAtom[idx].c[3], wTR.read(gid).x);   // NE

    // For other directions, read from neighbor's weight texture coords
    fstore(&resAtom[idx].c[4], (gid.x < width - 1) ? wLeft.read(gid + uint2(1,0)).x : 0.f);  // E
    fstore(&resAtom[idx].c[5], (gid.x < width - 1 && gid.y < height - 1) ? wTL.read(gid + uint2(1,1)).x : 0.f);   // SE
    fstore(&resAtom[idx].c[6], (gid.y < height - 1) ? wTop.read(gid + uint2(0,1)).x : 0.f);   // S
    fstore(&resAtom[idx].c[7], (gid.x > 0 && gid.y < height - 1) ? wTR.read(gid + uint2(-1,1)).x : 0.f); // SW
}

)";

// New push-relabel algorithm kernels
static const char* kGraphCutGlobalRelabelInitSrc = R"(
#include <metal_stdlib>
using namespace metal;

kernel void globalRelabelInitKernel(device NodeDataAtom* nodeBuf [[buffer(0)]],
                                   device TerminalFlow* termBuf [[buffer(1)]],
                                   device uint* bfsQueue [[buffer(2)]],
                                   device atomic_uint* queueCount [[buffer(3)]],
                                   constant uint& totalNodes [[buffer(4)]],
                                   uint gid [[thread_position_in_grid]])
{
    if (gid >= totalNodes) return;

    // Initialize all nodes to unreachable height
    atomic_store_explicit(&nodeBuf[gid].label, totalNodes, memory_order_relaxed);

    // Find sink-connected nodes and initialize frontier
    if (fload(&termBuf[gid].to_sink) > 1e-6f) {
        atomic_store_explicit(&nodeBuf[gid].label, 1, memory_order_relaxed);
        uint pos = atomic_fetch_add_explicit(queueCount, 1u, memory_order_relaxed);
        bfsQueue[pos] = gid;
    }
}
)";

static const char* kGraphCutGlobalRelabelBfsTraverseSrc = R"(
#include <metal_stdlib>
using namespace metal;

kernel void globalRelabelBfsTraverseKernel(device NodeDataAtom* nodeBuf [[buffer(0)]],
                                          const device ResidualGraphAtom* resBuf [[buffer(1)]],
                                          const device uint* levelIn [[buffer(2)]],
                                          constant uint& levelInCount [[buffer(3)]],
                                          device atomic_uint* nextCount [[buffer(4)]],
                                          device uint* levelOut [[buffer(5)]],
                                          constant uint& width [[buffer(6)]],
                                          constant uint& height [[buffer(7)]],
                                          uint gid [[thread_position_in_grid]])
{
    if (gid >= levelInCount) return;
    uint idx = levelIn[gid];

    int myLabel = atomic_load_explicit(&nodeBuf[idx].label, memory_order_relaxed);
    if (myLabel <= 0) return;

    uint x = idx % width;
    uint y = idx / width;

    // Check 8-connected neighbors for backward edges (residual capacity from neighbor to current)
    int2 offsets[8] = { int2(-1,0), int2(-1,-1), int2(0,-1), int2(1,-1),
                       int2(1,0), int2(1,1), int2(0,1), int2(-1,1) };

    // Reverse direction indices: E->W, SE->NW, S->N, SW->NE, W->E, NW->SE, N->S, NE->SW
    int reverse_dir[8] = { 4, 5, 6, 7, 0, 1, 2, 3 };

    for (uint k = 0; k < 8; ++k) {
        int nx = int(x) + offsets[k].x;
        int ny = int(y) + offsets[k].y;
        if (nx < 0 || ny < 0 || nx >= int(width) || ny >= int(height)) continue;

        uint nIdx = ny * width + nx;

        // Check residual capacity from neighbor to current node.
        // This is the capacity of the *reverse* edge.
        float cap = fload(&resBuf[nIdx].c[reverse_dir[k]]);

        if (cap > 1e-6f) {
            int neighLabel = atomic_load_explicit(&nodeBuf[nIdx].label, memory_order_relaxed);
            if (neighLabel > myLabel + 1) { // Can be relabeled
                int newLabel = myLabel + 1;
                if (atomic_compare_exchange_weak_explicit(&nodeBuf[nIdx].label, &neighLabel, newLabel,
                                                        memory_order_relaxed, memory_order_relaxed)) {
                    uint pos = atomic_fetch_add_explicit(nextCount, 1u, memory_order_relaxed);
                    levelOut[pos] = nIdx;
                }
            }
        }
    }
}
)";

static const char* kGraphCutCreateActiveListSrc = R"(
#include <metal_stdlib>
using namespace metal;

kernel void createInitialActiveListKernel(const device NodeDataAtom* nodeBuf [[buffer(0)]],
                                         device uint* activeList [[buffer(1)]],
                                         device atomic_uint* activeCount [[buffer(2)]],
                                         constant uint& totalNodes [[buffer(3)]],
                                         uint gid [[thread_position_in_grid]])
{
    if (gid >= totalNodes) return;

    float excess = fload(&nodeBuf[gid].excessBits);
    if (excess > 1e-6f) {
        uint pos = atomic_fetch_add_explicit(activeCount, 1u, memory_order_relaxed);
        activeList[pos] = gid;
    }
}
)";

static const char* kGraphCutPushRelabelSrc = R"(
#include <metal_stdlib>
using namespace metal;

kernel void pushRelabelKernel(device NodeDataAtom* nodeBuf [[buffer(0)]],
                             device ResidualGraphAtom* resBuf [[buffer(1)]],
                             device TerminalFlow* termBuf [[buffer(2)]],
                             const device uint* activeListIn [[buffer(3)]],
                             constant uint& activeCount [[buffer(4)]],
                             device atomic_uint* nextCount [[buffer(5)]],
                             device uint* activeListOut [[buffer(6)]],
                             constant uint& width [[buffer(7)]],
                             constant uint& height [[buffer(8)]],
                             uint gid [[thread_position_in_grid]])
{
    if (gid >= activeCount) return;
    uint idx = activeListIn[gid];

    float excess = fload(&nodeBuf[idx].excessBits);
    if (excess <= 1e-6f) return; // Not active

    int myHeight = atomic_load_explicit(&nodeBuf[idx].label, memory_order_relaxed);
    if (myHeight >= int(width*height)) return; // Unreachable node

    uint x = idx % width;
    uint y = idx / width;

    // Directions: 0:W, 1:NW, 2:N, 3:NE, 4:E, 5:SE, 6:S, 7:SW
    int2 offsets[8] = { int2(-1,0), int2(-1,-1), int2(0,-1), int2(1,-1),
                       int2(1,0), int2(1,1), int2(0,1), int2(-1,1) };
    // Reverse direction indices: E->W, SE->NW, S->N, SW->NE, W->E, NW->SE, N->S, NE->SW
    int reverse_dir[8] = { 4, 5, 6, 7, 0, 1, 2, 3 };

    // Try to push to neighbors
    for (uint k = 0; k < 8 && excess > 1e-6f; ++k) {
        int nx = int(x) + offsets[k].x;
        int ny = int(y) + offsets[k].y;
        if (nx < 0 || ny < 0 || nx >= int(width) || ny >= int(height)) continue;

        uint nIdx = ny * width + nx;

        float cap = fload(&resBuf[idx].c[k]);
        if (cap <= 1e-6f) continue;

        int neighHeight = atomic_load_explicit(&nodeBuf[nIdx].label, memory_order_relaxed);

        // Push condition: height(u) = height(v) + 1
        if (myHeight == neighHeight + 1) {
            float delta = min(excess, cap);

            // Atomically update excesses
            fsub(&nodeBuf[idx].excessBits, delta);
            float oldNeighExcess = fadd(&nodeBuf[nIdx].excessBits, delta);

            // Atomically update residual capacities for forward and reverse edges
            fsub(&resBuf[idx].c[k], delta);
            fadd(&resBuf[nIdx].c[reverse_dir[k]], delta);

            excess -= delta;

            // If neighbor became active, add it to the next active list
            if (oldNeighExcess <= 1e-6f) {
                uint pos = atomic_fetch_add_explicit(nextCount, 1u, memory_order_relaxed);
                activeListOut[pos] = nIdx;
            }
        }
    }

    // Try to push to sink if excess remains
    if (excess > 1e-6f) {
        if (myHeight == 1) { // height(u) == height(sink) + 1
            float cap_to_sink = fload(&termBuf[idx].to_sink);
            if (cap_to_sink > 1e-6f) {
                float delta = min(excess, cap_to_sink);
                fsub(&nodeBuf[idx].excessBits, delta);
                fsub(&termBuf[idx].to_sink, delta);
                fadd(&termBuf[idx].to_source, delta); // Add reverse capacity
                excess -= delta;
            }
        }
    }

    // If still have excess, relabel and re-queue
    if (excess > 1e-6f) {
        int minHeight = INT_MAX;

        // Find minimum height among neighbors with positive residual capacity
        for (uint k = 0; k < 8; ++k) {
             int nx = int(x) + offsets[k].x;
             int ny = int(y) + offsets[k].y;
             if (nx < 0 || ny < 0 || nx >= int(width) || ny >= int(height)) continue;
             uint nIdx = ny * width + nx;

            if (fload(&resBuf[idx].c[k]) > 1e-6f) {
                minHeight = min(minHeight, atomic_load_explicit(&nodeBuf[nIdx].label, memory_order_relaxed));
            }
        }

        // Also consider the sink (height 0)
        if (fload(&termBuf[idx].to_sink) > 1e-6f) {
            minHeight = min(minHeight, 0);
        }

        // Relabel: set height = min neighbor height + 1
        if (minHeight < INT_MAX) {
            atomic_store_explicit(&nodeBuf[idx].label, minHeight + 1, memory_order_relaxed);
        }

        // Re-queue for next iteration
        uint pos = atomic_fetch_add_explicit(nextCount, 1u, memory_order_relaxed);
        activeListOut[pos] = idx;
    }
}
)";

// Phase 3 Optimization Kernels
static const char* kGraphCutBuildHeightHistogramSrc = R"(
#include <metal_stdlib>
using namespace metal;

kernel void buildHeightHistogramKernel(const device NodeDataAtom* nodeBuf [[buffer(0)]],
                                      device atomic_uint* histogram [[buffer(1)]],
                                      constant uint& totalNodes [[buffer(2)]],
                                      uint gid [[thread_position_in_grid]])
{
    if (gid >= totalNodes) return;
    int height = atomic_load_explicit(&nodeBuf[gid].label, memory_order_relaxed);
    if (height < totalNodes && height > 0) { // only count valid heights
        atomic_fetch_add_explicit(&histogram[height], 1u, memory_order_relaxed);
    }
}
)";

static const char* kGraphCutGapRelabelSrc = R"(
#include <metal_stdlib>
using namespace metal;

kernel void gapRelabelKernel(device NodeDataAtom* nodeBuf [[buffer(0)]],
                            constant uint& gapHeight [[buffer(1)]],
                            constant uint& totalNodes [[buffer(2)]],
                            uint gid [[thread_position_in_grid]])
{
    if (gid >= totalNodes) return;
    int height = atomic_load_explicit(&nodeBuf[gid].label, memory_order_relaxed);
    if (height > int(gapHeight)) {
        atomic_store_explicit(&nodeBuf[gid].label, totalNodes, memory_order_relaxed);
    }
}
)";

static const char* kGraphCutFinalCutBfsInitSrc = R"(
#include <metal_stdlib>
using namespace metal;

kernel void finalCutBfsInitKernel(const device NodeDataAtom* nodeBuf [[buffer(0)]],
                                 device int* reachableLabels [[buffer(1)]],
                                 device uint* bfsQueue [[buffer(2)]],
                                 device atomic_uint* queueCount [[buffer(3)]],
                                 constant uint& totalNodes [[buffer(4)]],
                                 uint gid [[thread_position_in_grid]])
{
    if (gid >= totalNodes) return;

    // Initialize all nodes as unreachable from source
    reachableLabels[gid] = 0;

    // Nodes unreachable from the sink (label >= totalNodes) are part of the source-side cut (background).
    // These form the initial set for the final BFS.
    int height = atomic_load_explicit(&nodeBuf[gid].label, memory_order_relaxed);
    if (height >= int(totalNodes)) {
        reachableLabels[gid] = 1; // Mark as reachable from source
        uint pos = atomic_fetch_add_explicit(queueCount, 1u, memory_order_relaxed);
        bfsQueue[pos] = gid;
    }
}
)";

static const char* kGraphCutFinalCutBfsTraverseSrc = R"(
#include <metal_stdlib>
using namespace metal;

kernel void finalCutBfsTraverseKernel(const device uint* levelIn [[buffer(0)]],
                                     constant uint& levelInCount [[buffer(1)]],
                                     const device ResidualGraphAtom* resBuf [[buffer(2)]],
                                     device int* reachableLabels [[buffer(3)]],
                                     device atomic_uint* nextCount [[buffer(4)]],
                                     device uint* levelOut [[buffer(5)]],
                                     constant uint& width [[buffer(6)]],
                                     constant uint& height [[buffer(7)]],
                                     uint gid [[thread_position_in_grid]])
{
    if (gid >= levelInCount) return;
    uint idx = levelIn[gid];

    uint x = idx % width;
    uint y = idx / width;

    // Check 8-connected neighbors for forward edges with positive residual capacity
    int2 offsets[8] = { int2(-1,0), int2(-1,-1), int2(0,-1), int2(1,-1),
                       int2(1,0), int2(1,1), int2(0,1), int2(-1,1) };

    for (uint k = 0; k < 8; ++k) {
        int nx = int(x) + offsets[k].x;
        int ny = int(y) + offsets[k].y;

        if (nx < 0 || nx >= int(width) || ny < 0 || ny >= int(height)) continue;

        uint nidx = ny * width + nx;
        if (reachableLabels[nidx] != 0) continue; // Already reachable

        // Check residual capacity from current to neighbor
        float residual = fload(&resBuf[idx].c[k]);

        // If there's positive residual capacity, neighbor is reachable
        if (residual > 1e-6f) {
            int expected = 0;
            if (atomic_compare_exchange_weak_explicit((device atomic_int*)&reachableLabels[nidx], &expected, 1,
                                                     memory_order_relaxed, memory_order_relaxed)) {
                uint pos = atomic_fetch_add_explicit(nextCount, 1u, memory_order_relaxed);
                levelOut[pos] = nidx;
            }
        }
    }
}
)";

static const char* kGraphCutFinalCutWriteMaskSrc = R"(
#include <metal_stdlib>
using namespace metal;

kernel void finalCutWriteMaskKernel(const device int* reachableLabels [[buffer(0)]],
                                   texture2d<uint, access::write> mask [[texture(1)]],
                                   texture2d<uint, access::read> initial_mask [[texture(2)]],
                                   constant uint& width [[buffer(3)]],
                                   uint2 gid [[thread_position_in_grid]])
{
    if (gid.x >= width || gid.y >= mask.get_height()) return;

    uint idx = gid.y * width + gid.x;
    uint initialMaskVal = initial_mask.read(gid).x;

    // GrabCut mask values: GC_BGD=0, GC_FGD=1, GC_PR_BGD=2, GC_PR_FGD=3
    uint finalMaskVal;

    if (initialMaskVal == 0u || initialMaskVal == 1u) { // Sure BGD or Sure FGD
        finalMaskVal = initialMaskVal;
    } else { // Probable BGD or Probable FGD
        // A node is foreground if it is reachable from the source terminal in the residual graph.
        if (reachableLabels[idx] != 0) { // Reachable from source -> Foreground
            finalMaskVal = 3u; // GC_PR_FGD
        } else { // Not reachable from source -> Background
            finalMaskVal = 2u; // GC_PR_BGD
        }
    }
    mask.write(finalMaskVal, gid);
}
)";

} // anonymous namespace

#endif // OPENCV_METALIMGPROC_GRAPHCUT_KERNELS_HPP