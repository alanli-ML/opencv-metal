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
    uint old_val = atomic_load_explicit(p, memory_order_relaxed);
    uint new_val;
    float old_float;
    do {
        old_float = as_type<float>(old_val);
        float new_float = old_float + v;
        new_val = as_type<uint>(new_float);
    } while (!atomic_compare_exchange_weak_explicit(p, &old_val, new_val,
                                                   memory_order_relaxed,
                                                   memory_order_relaxed));
    return old_float;
}

inline float fsub(device atomic_uint* p, float v) {
    uint old_val = atomic_load_explicit(p, memory_order_relaxed);
    uint new_val;
    float old_float;
    do {
        old_float = as_type<float>(old_val);
        float new_float = old_float - v;
        new_val = as_type<uint>(new_float);
    } while (!atomic_compare_exchange_weak_explicit(p, &old_val, new_val,
                                                   memory_order_relaxed,
                                                   memory_order_relaxed));
    return old_float;
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
                             device atomic_uint* capToSourceBuf [[buffer(4)]],
                             uint2 gid [[thread_position_in_grid]])
{
    uint width = bgTerm.get_width();
    uint height = bgTerm.get_height();

    if (gid.x >= width || gid.y >= height) return;

    uint idx = gid.y * width + gid.x;

    // Handle hard constraints from the mask
    uint maskVal = mask.read(gid).x;
    float bg_from_tex = bgTerm.read(gid).x;
    float fg_from_tex = fgTerm.read(gid).x;
    
    // CORRECTED LOGIC FOR HARD CONSTRAINTS
    float t_weight_to_sink = 0.0f;    // capacity to sink
    float t_weight_from_source = 0.0f; // capacity from source
    int initial_height = 0;
    uint totalNodes = width * height;
    
    // GrabCut mask values: GC_BGD=0, GC_FGD=1, GC_PR_BGD=2, GC_PR_FGD=3
    if (maskVal == 0u) { // Sure Background (GC_BGD)
        // Sure background: tied to sink with infinite capacity
        t_weight_from_source = 0.0f;
        t_weight_to_sink = lambda;
        initial_height = 0; // Part of the sink
    } else if (maskVal == 1u) { // Sure Foreground (GC_FGD)
        // Sure foreground: receives flow from source
        t_weight_from_source = lambda;
        t_weight_to_sink = 0.0f;
        initial_height = totalNodes; // Acts as source
    } else { // Probable BG/FG (GC_PR_BGD=2, GC_PR_FGD=3)
        // Use GMM-derived costs
        // In the min-cut framework, a node is assigned to the FG set (S-set) if the
        // edge to the T-sink is cut, and to the BG set (T-set) if the edge
        // from the S-source is cut.
        // - capacity(S->p) is the penalty for assigning p to BG. This is -log(P(p|BG)).
        // - capacity(p->T) is the penalty for assigning p to FG. This is -log(P(p|FG)).
        //
        // For push-relabel, initial excess flow is capacity(S->p). We want probable
        // FG pixels to have high excess flow. A probable FG pixel has a high -log(P(p|BG)).
        t_weight_from_source = bg_from_tex; // Penalty for BG assignment.
        t_weight_to_sink = fg_from_tex;     // Penalty for FG assignment.
        
        // Start probable nodes with excess at height 1 to enable pushing.
        initial_height = (t_weight_from_source > 1e-6f) ? 1 : 0;
    }
    
    // Assign to buffers based on corrected logic
    // Initial excess is the capacity from the source terminal
    fstore(&nodeAtom[idx].excessBits, t_weight_from_source);
    // Residual capacity of the reverse edge (pixel->source) is the initial capacity
    fstore(capToSourceBuf + idx, t_weight_from_source);
    // Initial height
    atomic_store_explicit(&nodeAtom[idx].label, initial_height, memory_order_relaxed);
    
    // Terminal edge capacities
    fstore(&termBuf[idx].to_source, 0.0f); // Not used for pushing, just for final cut BFS
    fstore(&termBuf[idx].to_sink, t_weight_to_sink);

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
                                   // Removed bfsQueue and queueCount
                                   constant uint& totalNodes [[buffer(2)]],
                                   uint gid [[thread_position_in_grid]])
{
    if (gid >= totalNodes) return;

    // Special case: if node has excess and is already at height >= totalNodes,
    // keep it there (it may need to push to source)
    float excess = fload(&nodeBuf[gid].excessBits);
    int currentHeight = atomic_load_explicit(&nodeBuf[gid].label, memory_order_relaxed);
    
    if (excess > 1e-6f && currentHeight >= (int)totalNodes) {
        // Keep current height for nodes with excess that might push to source
        return;
    }

    // Initialize all nodes to unreachable height
    atomic_store_explicit(&nodeBuf[gid].label, totalNodes, memory_order_relaxed);

    // Find sink-connected nodes and initialize frontier.
    // Seed BFS with nodes that have residual capacity TO sink.
    if (fload(&termBuf[gid].to_sink) > 1e-6f) {
        atomic_store_explicit(&nodeBuf[gid].label, 1, memory_order_relaxed);
        // No longer need to write to a queue
    }
}
)";

static const char* kGraphCutGlobalRelabelBfsTraverseSrc = R"(
#include <metal_stdlib>
using namespace metal;

kernel void globalRelabelBfsTraverseKernel(device NodeDataAtom* nodeBuf [[buffer(0)]],
                                          const device ResidualGraphAtom* resBuf [[buffer(1)]],
                                          constant uint& current_bfs_level [[buffer(2)]],
                                          constant uint& width [[buffer(3)]],
                                          constant uint& height [[buffer(4)]],
                                          constant uint& totalNodes [[buffer(5)]],
                                          uint gid [[thread_position_in_grid]])
{
    if (gid >= totalNodes) return;
    uint idx = gid;

    int myLabel = atomic_load_explicit(&nodeBuf[idx].label, memory_order_relaxed);
    
    // Only threads whose label matches the current BFS level do work.
    if (myLabel != (int)current_bfs_level) return;

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
            // If neighbor is unvisited (or can be reached by a shorter path)
            if (neighLabel > myLabel + 1) {
                int newLabel = myLabel + 1;
                // Try to claim this neighbor for the next level
                atomic_compare_exchange_weak_explicit(&nodeBuf[nIdx].label, &neighLabel, newLabel,
                                                        memory_order_relaxed,
                                                        memory_order_relaxed);
            }
        }
    }
}
)";

static const char* kGraphCutPushRelabelSrc = R"(
#include <metal_stdlib>
using namespace metal;

kernel void pushRelabelKernel(device NodeDataAtom* nodeBuf [[buffer(0)]],
                             device ResidualGraphAtom* resBuf [[buffer(1)]],
                             device TerminalFlow* termBuf [[buffer(2)]],
                             constant uint& width [[buffer(3)]],
                             constant uint& height [[buffer(4)]],
                             device atomic_uint* capToSourceBuf [[buffer(5)]],
                             constant uint& totalNodes [[buffer(6)]],
                             uint gid [[thread_position_in_grid]])
{
    if (gid >= totalNodes) return;
    uint idx = gid;

    float excess = fload(&nodeBuf[idx].excessBits);
    if (excess <= 1e-6f) return; // Not active

    int myHeight = atomic_load_explicit(&nodeBuf[idx].label, memory_order_relaxed);

    // Prioritize pushing to sink if possible
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

    // Push to source if possible
    if (excess > 1e-6f) {
        // Source height is totalNodes. Push if myHeight == totalNodes + 1.
        if (myHeight == (int)totalNodes + 1) {
            float cap_to_source = fload(capToSourceBuf + idx);
            if (cap_to_source > 1e-6f) {
                float delta = min(excess, cap_to_source);
                fsub(&nodeBuf[idx].excessBits, delta);
                fsub(capToSourceBuf + idx, delta);
                // Note: reverse capacity c(source, pixel) is not tracked, as it's not
                // read by any other part of the algorithm.
                excess -= delta;
            }
        }
    }

    uint x = idx % width;
    uint y = idx / width;

    // Directions: 0:W, 1:NW, 2:N, 3:NE, 4:E, 5:SE, 6:S, 7:SW
    int2 offsets[8] = { int2(-1,0), int2(-1,-1), int2(0,-1), int2(1,-1),
                       int2(1,0), int2(1,1), int2(0,1), int2(-1,1) };
    // Reverse direction indices: E->W, SE->NW, S->N, SW->NE, W->E, NW->SE, N->S, NE->SW
    int reverse_dir[8] = { 4, 5, 6, 7, 0, 1, 2, 3 };

    // Try to push to neighbors if excess remains
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
            fadd(&nodeBuf[nIdx].excessBits, delta);

            // Atomically update residual capacities for forward and reverse edges
            fsub(&resBuf[idx].c[k], delta);
            fadd(&resBuf[nIdx].c[reverse_dir[k]], delta);

            excess -= delta;
        }
    }

    // If still have excess, the node may need to be relabeled.
    if (excess > 1e-6f) {
        // Only relabel if the node is not a source node (height < totalNodes).
        if (myHeight < (int)totalNodes) {
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

            // Also consider the source (height totalNodes).
            if (fload(capToSourceBuf + idx) > 1e-6f) {
                minHeight = min(minHeight, int(totalNodes));
            }

            // Relabel: set height = min neighbor height + 1
            if (minHeight < INT_MAX) {
                // New height can be totalNodes+1 to push back to source
                int newHeight = minHeight + 1;
                atomic_store_explicit(&nodeBuf[idx].label, newHeight, memory_order_relaxed);
            }
        }
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

static const char* kGraphCutFindGapSrc = R"(
#include <metal_stdlib>
using namespace metal;

// This kernel is dispatched with a single thread.
// It scans the histogram to find the first gap.
kernel void findGapKernel(const device uint* histogram [[buffer(0)]],
                         device atomic_uint* gapHeightOut [[buffer(1)]],
                         constant uint& totalNodes [[buffer(2)]])
{
    // Find the first height h > 0 where histogram[h] is 0 but some h2 > h is not.
    for (uint h = 1; h < totalNodes - 1; ++h) {
        if (histogram[h] == 0) {
            // Found a potential gap, now confirm there's a node at a higher level.
            bool hasHigherNodes = false;
            for (uint h2 = h + 1; h2 < totalNodes; ++h2) {
                if (histogram[h2] > 0) {
                    hasHigherNodes = true;
                    break;
                }
            }
            if (hasHigherNodes) {
                // This is a valid gap. Store it and terminate.
                atomic_store_explicit(gapHeightOut, h, memory_order_relaxed);
                return;
            }
        }
    }
    // No gap found, ensure output is 0.
    atomic_store_explicit(gapHeightOut, 0u, memory_order_relaxed);
}
)";

static const char* kGraphCutGapRelabelSrc = R"(
#include <metal_stdlib>
using namespace metal;

kernel void gapRelabelKernel(device NodeDataAtom* nodeBuf [[buffer(0)]],
                            const device atomic_uint* gapHeightBuf [[buffer(1)]],
                            constant uint& totalNodes [[buffer(2)]],
                            uint gid [[thread_position_in_grid]])
{
    if (gid >= totalNodes) return;

    uint gapHeight = atomic_load_explicit(gapHeightBuf, memory_order_relaxed);
    if (gapHeight == 0) return; // No gap was found

    int height = atomic_load_explicit(&nodeBuf[gid].label, memory_order_relaxed);
    if (height > int(gapHeight)) {
        atomic_store_explicit(&nodeBuf[gid].label, totalNodes, memory_order_relaxed);
    }
}
)";

static const char* kGraphCutFinalCutBfsInit_SourceSet_Src = R"(
#include <metal_stdlib>
using namespace metal;

kernel void finalCutBfsInit_SourceSet_Kernel(const device NodeDataAtom* nodeBuf [[buffer(0)]],
                                             device int* reachableLabels [[buffer(1)]],
                                             // Removed bfsQueue and queueCount
                                             constant uint& totalNodes [[buffer(2)]],
                                             uint gid [[thread_position_in_grid]])
{
    if (gid >= totalNodes) return;

    // Initialize all nodes as unreachable from source (i.e. background)
    reachableLabels[gid] = 0;

    // In push-relabel, after convergence:
    // - Nodes at height >= totalNodes are connected to source
    // - Nodes at height 0 are connected to sink
    // - Nodes at intermediate heights form the cut
    int height = atomic_load_explicit(&nodeBuf[gid].label, memory_order_relaxed);
    
    // Seed BFS with nodes that are in the source set
    if (height >= (int)totalNodes) {
        reachableLabels[gid] = 1; // Mark as reachable from source (foreground) for level 1
        // No longer need to write to a queue
    }
}
)";

static const char* kGraphCutFinalCutBfsTraverse_SourceSet_Src = R"(
#include <metal_stdlib>
using namespace metal;

kernel void finalCutBfsTraverse_SourceSet_Kernel(const device ResidualGraphAtom* resBuf [[buffer(0)]],
                                                 device int* reachableLabels [[buffer(1)]],
                                                 constant uint& current_bfs_level [[buffer(2)]],
                                                 constant uint& width [[buffer(3)]],
                                                 constant uint& height [[buffer(4)]],
                                                 constant uint& totalNodes [[buffer(5)]],
                                                 uint gid [[thread_position_in_grid]])
{
    if (gid >= totalNodes) return;
    uint idx = gid;

    // Only process nodes on the current frontier
    if (reachableLabels[idx] != (int)current_bfs_level) return;

    uint x = idx % width;
    uint y = idx / width;

    // To find all nodes in the S-set (foreground), we perform a forward traversal
    // on the residual graph starting from nodes at source height. This finds all
    // nodes reachable from the source.
    int2 offsets[8] = { int2(-1,0), int2(-1,-1), int2(0,-1), int2(1,-1),
                       int2(1,0), int2(1,1), int2(0,1), int2(-1,1) };

    for (uint k = 0; k < 8; ++k) {
        int nx = int(x) + offsets[k].x;
        int ny = int(y) + offsets[k].y;

        if (nx < 0 || nx >= int(width) || ny < 0 || ny >= int(height)) continue;

        uint nidx = ny * width + nx;
        
        // Check residual capacity from current node to neighbor (forward edge).
        float residual = fload(&resBuf[idx].c[k]);

        if (residual > 1e-6f) {
            // If neighbor is unvisited (label 0), try to claim it for the next level
            int expected = 0;
            int new_label = current_bfs_level + 1;
            atomic_compare_exchange_weak_explicit((device atomic_int*)&reachableLabels[nidx], &expected, new_label,
                                                     memory_order_relaxed, memory_order_relaxed);
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
        // With source-set reachability:
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