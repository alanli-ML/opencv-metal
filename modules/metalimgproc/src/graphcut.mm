#include "precomp.hpp"

#ifdef HAVE_METAL

#include "graphcut_internal.hpp"
#include <stdexcept>
#include <simd/simd.h>
#include <cstring>
#include <algorithm>
#include "opencv2/core/hal/interface.h" // for CV_8UC1 definition

// -------------------------------------------------------------------------
// Common Metal helpers for upcoming atomic refactor (Slice 1)
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

namespace {

// Minimal MSL kernel to derive a provisional segmentation by comparing unary
// costs. This is *not* the push-relabel algorithm – it is just a placeholder so
// that the GPU path produces something meaningful while the full solver is
// under construction.
static const char* kGraphCutKernelsSrc = R"(
#include <metal_stdlib>
using namespace metal;

kernel void simpleSegmentationKernel(texture2d<float, access::read> bgTerm [[texture(0)]],
                                     texture2d<float, access::read> fgTerm [[texture(1)]],
                                     texture2d<uint,  access::write> mask  [[texture(2)]],
                                     uint2 gid [[thread_position_in_grid]])
{
    if (gid.x >= bgTerm.get_width() || gid.y >= bgTerm.get_height()) return;

    float bgCost = bgTerm.read(gid).x;
    float fgCost = fgTerm.read(gid).x;

    // If foreground cost lower → probable FG (3), else probable BG (2).
    uint label = (fgCost < bgCost) ? 3u : 2u;
    mask.write(label, gid);
}

)";

// Additional kernels for building the graph data structures
static const char* kGraphCutBuildSrc = R"(
#include <metal_stdlib>
using namespace metal;

struct NodeData {
    float excess;
    int   label;
};

kernel void buildGraphKernel(texture2d<float, access::read> bgTerm [[texture(0)]],
                             texture2d<float, access::read> fgTerm [[texture(1)]],
                             texture2d<float, access::read> wLeft [[texture(2)]],
                             texture2d<float, access::read> wTL [[texture(3)]],
                             texture2d<float, access::read> wTop [[texture(4)]],
                             texture2d<float, access::read> wTR [[texture(5)]],
                             device NodeData*       nodeBuf [[buffer(0)]],
                             device TerminalFlow*   termBuf [[buffer(1)]],
                             device float4*         resBuf  [[buffer(2)]],
                             uint2 gid [[thread_position_in_grid]])
{
    if (gid.x >= bgTerm.get_width() || gid.y >= bgTerm.get_height()) return;

    uint idx = gid.y * bgTerm.get_width() + gid.x;

    float bg = bgTerm.read(gid).x;
    float fg = fgTerm.read(gid).x;

    // Initial excess zero, labels 0 (unreached)
    nodeBuf[idx].excess = bg - fg; // positive excess indicates push to sink
    nodeBuf[idx].label  = 0;

    fstore(&termBuf[idx].to_source, bg);
    fstore(&termBuf[idx].to_sink, fg);

    // Store pairwise weights (only magnitude; direction handled when pushing flow)
    float4 res;
    res.x = wLeft.read(gid).x;  // capacity to left neighbor
    res.y = wTL.read(gid).x;    // to top-left
    res.z = wTop.read(gid).x;   // to top
    res.w = wTR.read(gid).x;    // to top-right
    resBuf[idx] = res;
}

)";

// Slice 2: Build graph into atomic buffers
static const char* kGraphCutBuildAtomSrc = R"(
#include <metal_stdlib>
using namespace metal;

// Bring in common helpers via include string concatenation

// DEBUG: Struct to hold values for inspection from the GPU
struct BuildGraphDebugInfo {
    float bg_cost;
    float fg_cost;
    uint mask_val;
    float unary_bg;
    float unary_fg;
};

kernel void buildGraphAtomKernel(texture2d<float, access::read> bgTerm [[texture(0)]],
                             texture2d<float, access::read> fgTerm [[texture(1)]],
                             texture2d<float, access::read> wLeft [[texture(2)]],
                             texture2d<float, access::read> wTop [[texture(4)]], // Note: Indices match buildGraph call
                             texture2d<float, access::read> wTL  [[texture(3)]],
                             texture2d<float, access::read> wTR  [[texture(5)]],
                             texture2d<uint, access::read> mask [[texture(6)]],
                             device NodeDataAtom* nodeAtom [[buffer(0)]],
                             device TerminalFlow* termBuf [[buffer(1)]],
                             device ResidualGraphAtom* resAtom [[buffer(2)]],
                             constant float& lambda [[buffer(3)]],
                             device BuildGraphDebugInfo* debugBuf [[buffer(4)]],
                             uint2 gid [[thread_position_in_grid]])
{
    uint width = bgTerm.get_width();
    uint height = bgTerm.get_height();
    
    // DEBUG: Store dimensions in first thread
    if (gid.x == 0 && gid.y == 0) {
        debugBuf[0].bg_cost = (float)width;  // Store width
        debugBuf[0].fg_cost = (float)height; // Store height
    }
    
    if (gid.x >= width || gid.y >= height) return;

    uint idx = gid.y * width + gid.x;

    // Handle hard constraints from the mask - force recompile v3
    uint maskVal = mask.read(gid).x;
    
    // DEBUG: Check if ANY threads execute by using thread ID 0,0 and 1,1
    if ((gid.x == 0 && gid.y == 0)) {
        debugBuf[0].mask_val = maskVal;
        debugBuf[0].unary_bg = (float)gid.x; // Should be 0
        debugBuf[0].unary_fg = (float)gid.y; // Should be 0
    }
    if ((gid.x == 1 && gid.y == 1)) {
        debugBuf[1].mask_val = maskVal;
        debugBuf[1].unary_bg = (float)gid.x; // Should be 1 
        debugBuf[1].unary_fg = (float)gid.y; // Should be 1
    }

    float unary_bg_cost = 0.0f; // Cost for this pixel being background
    float unary_fg_cost = 0.0f; // Cost for this pixel being foreground

    // DEBUG: Read texture values into local variables to inspect them
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

    // DEBUG: Write out values for a few sample pixels
    if ((gid.x == 102 && gid.y == 192) || (gid.x == 204 && gid.y == 192)) {
        int debug_idx = (gid.x == 102) ? 0 : 1;
        debugBuf[debug_idx].bg_cost = bg_from_tex;
        debugBuf[debug_idx].fg_cost = fg_from_tex;
        debugBuf[debug_idx].mask_val = maskVal;
        debugBuf[debug_idx].unary_bg = unary_bg_cost;
        debugBuf[debug_idx].unary_fg = unary_fg_cost;
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

// Slice 3: atomic BFS init kernel (sets label & frontier list)
static const char* kGraphCutBFSInitAtomSrc = R"(
#include <metal_stdlib>
using namespace metal;

kernel void bfsInitAtomKernel(const device NodeDataAtom* nodeBuf [[buffer(0)]],
                              device atomic_uint* levelCount [[buffer(1)]],
                              device uint*        levelList  [[buffer(2)]],
                              constant uint&       totalNodes [[buffer(3)]],
                              uint3 gid [[thread_position_in_grid]])
{
    uint idx = gid.x;
    if (idx >= totalNodes) return;

    int lbl = atomic_load_explicit(&nodeBuf[idx].label, memory_order_relaxed);
    if (lbl == 1) // reachable from sink set earlier by float BFS (or placeholder)
    {
        uint pos = atomic_fetch_add_explicit(levelCount, 1u, memory_order_relaxed);
        levelList[pos] = idx;
    }
}

)";

// Slice 4: atomic BFS traverse kernel
static const char* kGraphCutBFSTraverseAtomSrc = R"(
#include <metal_stdlib>
using namespace metal;

kernel void bfsTraverseAtomKernel(device NodeDataAtom* nodeBuf [[buffer(0)]],
                                  const device ResidualGraphAtom* resBuf [[buffer(1)]],
                                  device uint* levelIn [[buffer(2)]],
                                  constant uint& levelInCount [[buffer(3)]],
                                  device atomic_uint* nextCount [[buffer(4)]],
                                  device uint* levelOut [[buffer(5)]],
                                  constant uint& width [[buffer(6)]],
                                  constant uint& height [[buffer(7)]],
                                  uint gid [[thread_position_in_grid]])
{
    uint idxIn = gid.x;
    if (idxIn >= levelInCount) return;
    uint idx = levelIn[idxIn];

    int myLabel = atomic_load_explicit(&nodeBuf[idx].label, memory_order_relaxed);

    // Offsets for 8 neighbours
    int2 offsets[8] = { int2(-1,0), int2(-1,-1), int2(0,-1), int2(1,-1),
                       int2(1,0), int2(1,1), int2(0,1), int2(-1,1) };

    uint x = idx % width;
    uint y = idx / width;

    for(uint k=0; k < 8; ++k){
        int nx = int(x) + offsets[k].x;
        int ny = int(y) + offsets[k].y;
        if (nx < 0 || ny < 0 || nx >= int(width) || ny >= int(height)) continue;
        uint nIdx = ny * width + nx;

        // Load residual capacity from idx to neighbour
        float cap = fload(&resBuf[idx].c[k]);

        if(cap > 1e-6f){
            int neighLabel = atomic_load_explicit(&nodeBuf[nIdx].label, memory_order_relaxed);
            if(neighLabel==0){
                // set label atomically
                if(atomic_compare_exchange_weak_explicit(&nodeBuf[nIdx].label, &neighLabel, myLabel+1,
                                                        memory_order_relaxed, memory_order_relaxed)){
                    uint pos = atomic_fetch_add_explicit(nextCount, 1u, memory_order_relaxed);
                    levelOut[pos] = nIdx;
                }
            }
        }
    }
}

)";

// Slice 5 - Push kernel (atomic)
static const char* kGraphCutPushAtomSrc = R"(
#include <metal_stdlib>
using namespace metal;

kernel void pushAtomKernel(device NodeDataAtom* nodeBuf     [[buffer(0)]],
                          device ResidualGraphAtom* resBuf    [[buffer(1)]],
                          device uint*         activeList [[buffer(2)]],
                          constant uint&       activeCount [[buffer(3)]],
                          device atomic_uint*  nextCount  [[buffer(4)]],
                          device uint*         nextList   [[buffer(5)]],
                          device TerminalFlow* termBuf    [[buffer(6)]],
                          device atomic_uint*  excessFlag [[buffer(7)]],
                          uint tid [[thread_position_in_grid]])
{
    if (tid>=activeCount) return;
    uint idx = activeList[tid];

    float exc = fload(&nodeBuf[idx].excessBits);

    const float EPS = 1e-3f;

    if(exc > EPS){ // push to sink
        float cap = fload(&termBuf[idx].to_sink);
        float delta = (exc < cap) ? exc : cap;
        if(delta>EPS){
            exc = exc - delta;
            fsub(&termBuf[idx].to_sink, delta);
            fstore(&nodeBuf[idx].excessBits, exc);
        }
    } else if(exc < -EPS){ // pull from source (negative excess)
        float need = -exc;
        float cap = fload(&termBuf[idx].to_source);
        float delta = (need < cap) ? need : cap;
        if(delta>EPS){
            exc = exc + delta;
            fsub(&termBuf[idx].to_source, delta);
            fstore(&nodeBuf[idx].excessBits, exc);
        }
    }

    // if still active requeue
    if(fabs(exc) > EPS){
        uint pos = atomic_fetch_add_explicit(nextCount,1u, memory_order_relaxed);
        nextList[pos] = idx;
        atomic_store_explicit(excessFlag,1u,memory_order_relaxed);
    }
}
)";

// Slice 6 - simple relabel kernel: bump height of still active nodes
static const char* kGraphCutRelabelAtomSrc = R"(
#include <metal_stdlib>
using namespace metal;

kernel void relabelAtomKernel(device NodeDataAtom* nodeBuf [[buffer(0)]],
                              device uint* activeList      [[buffer(1)]],
                              constant uint& activeCount   [[buffer(2)]],
                              uint tid [[thread_position_in_grid]])
{
    if(tid>=activeCount) return;
    uint idx = activeList[tid];
    float exc = fload(&nodeBuf[idx].excessBits);
    if(fabs(exc) > 1e-3f){
        atomic_fetch_add_explicit(&nodeBuf[idx].label, 1, memory_order_relaxed);
    }
}
)";

// Kernel to initialize node labels and set excess based on terminal capacities
static const char* kGraphCutBFSInitSrc = R"(
#include <metal_stdlib>
using namespace metal;

struct NodeData { float excess; int label; };

kernel void bfsInitKernel(device NodeData* nodeBuf [[buffer(0)]],
                          device TerminalFlow* termBuf [[buffer(1)]],
                          device uint*         levelList [[buffer(2)]],
                          device atomic_uint*  levelCount [[buffer(3)]],
                          constant uint& totalNodes [[buffer(4)]],
                          uint3 gid [[thread_position_in_grid]])
{
    uint idx = gid.x;
    if (idx >= totalNodes) return;

    float toS = fload(&termBuf[idx].to_source);
    float toT = fload(&termBuf[idx].to_sink);
    int lbl = (toT < toS) ? 1 : 0; // reachable from sink initially
    nodeBuf[idx].label = lbl;

    if (lbl == 1)
    {
        uint pos = atomic_fetch_add_explicit(levelCount, 1u, memory_order_relaxed);
        levelList[pos] = idx;
    }
}

)";

// Kernel for BFS traversal using current frontier list
static const char* kGraphCutBFSTraverseSrc = R"(
#include <metal_stdlib>
using namespace metal;

struct NodeData { float excess; int label; };

kernel void bfsTraverseKernel(const device NodeData* nodeBuf [[buffer(0)]],
                              device atomic_uint* nextCount [[buffer(1)]],
                              device const uint*   levelIn  [[buffer(2)]],
                              constant uint&       levelInCount [[buffer(3)]],
                              device uint*         levelOut [[buffer(4)]],
                              const device float4* residualBuf [[buffer(5)]],
                              constant uint&       width [[buffer(6)]],
                              uint3 gid [[thread_position_in_grid]])
{
    uint idxIn = gid.x;
    if (idxIn >= levelInCount) return;

    uint nodeIdx = levelIn[idxIn];

    int nodeLabel = nodeBuf[nodeIdx].label;

    // Neighbour offsets in raster order
    int2 offsets[4] = { int2(-1,0), int2(-1,-1), int2(0,-1), int2(1,-1) };

    uint x = nodeIdx % width;
    uint y = nodeIdx / width;

    for (uint n = 0; n < 4; ++n)
    {
        int nx = (int)x + offsets[n].x;
        int ny = (int)y + offsets[n].y;
        if (nx < 0 || ny < 0) continue; // left or top boundary already handled by indexing
        uint nIdx = ny * width + nx;

        // Determine residual capacity from current node to neighbour based on direction
        float4 res = residualBuf[nodeIdx];
        float cap = 0.0f;
        if (n == 0) cap = res.x;      // left
        else if (n == 1) cap = res.y; // tl
        else if (n == 2) cap = res.z; // top
        else if (n == 3) cap = res.w; // tr

        if (cap > 0.0f)
        {
            // Atomically set label if it is still 0
            int prev = atomic_exchange_explicit((device atomic_int*)&nodeBuf[nIdx].label, nodeLabel + 1, memory_order_relaxed);
            if (prev == 0)
            {
                uint pos = atomic_fetch_add_explicit(nextCount, 1u, memory_order_relaxed);
                levelOut[pos] = nIdx;
            }
        }
    }
}

)";

// Kernel to write mask based on label > 0
static const char* kGraphCutLabelSegSrc = R"(
#include <metal_stdlib>
using namespace metal;

kernel void labelSegmentationKernel(const device NodeDataAtom* nodeBuf [[buffer(0)]],
                                    texture2d<uint, access::write> maskTex [[texture(0)]],
                                    constant uint& width [[buffer(1)]],
                                    uint2 gid [[thread_position_in_grid]])
{
    if (gid.x >= maskTex.get_width() || gid.y >= maskTex.get_height()) return;
    uint idx = gid.y * width + gid.x;
    int nodeLabel = atomic_load_explicit(&nodeBuf[idx].label, memory_order_relaxed);
    uint label = (nodeLabel > 0) ? 3u : 2u; // 3 probable FG, 2 probable BG
    maskTex.write(label, gid);
}
)";

// Placeholder pushKernel – currently just sets excessFlag=0
static const char* kGraphCutPushSrc = R"(
#include <metal_stdlib>
using namespace metal;

struct NodeData { float excess; int label; };

kernel void pushKernel(const device NodeData* nodeBuf [[buffer(0)]], device atomic_uint* flag [[buffer(1)]], constant uint& totalNodes [[buffer(2)]], uint3 gid [[thread_position_in_grid]])
{
    uint idx = gid.x + gid.y * get_thread_execution_width();
    if (idx >= totalNodes) return;
    if (nodeBuf[idx].excess > 1e-3f)
        atomic_store_explicit(flag, 1u, memory_order_relaxed);
}
)";

static const char* kGraphCutCheckExcessSrc = R"(
#include <metal_stdlib>
using namespace metal;

kernel void checkExcessKernel(const device atomic_uint* excessFlag [[buffer(0)]], device atomic_uint* outFlag [[buffer(1)]], uint3 gid [[thread_position_in_grid]])
{
    if (gid.x==0 && gid.y==0)
    {
        uint v = atomic_load_explicit(excessFlag, memory_order_relaxed);
        atomic_store_explicit(outFlag, v, memory_order_relaxed);
    }
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

static const char* kGraphCutSimpleFinalCutSrc = R"(
#include <metal_stdlib>
using namespace metal;

kernel void simpleFinalCutKernel(const device TerminalFlow* termBuf [[buffer(0)]],
                                 texture2d<uint, access::write> mask [[texture(1)]],
                                 constant uint& width [[buffer(2)]],
                                 uint2 gid [[thread_position_in_grid]])
{
    if (gid.x >= width || gid.y >= mask.get_height()) return;
    
    uint idx = gid.y * width + gid.x;
    
    // If there is still capacity to the sink, it's part of the foreground set.
    // In GrabCut, GC_BGD=0, GC_FGD=1. The final mask should be certain.
    uint maskValue = (fload(&termBuf[idx].to_sink) > 1e-6f) ? 1u : 0u;
    mask.write(maskValue, gid);
}
)";

} // anonymous namespace

using namespace cv::metal;

id<MTLComputePipelineState> MetalGraphCut::getSimpleSegmentationPipeline()
{
    static id<MTLComputePipelineState> s_pipeline = nil;
    if (!s_pipeline)
    {
        MetalContext& ctx = MetalContext::getInstance();
        id<MTLFunction> func = ctx.getMetalFunction(kGraphCutKernelsSrc, "simpleSegmentationKernel");
        if (!func)
            return nil;
        NSError* err = nil;
        s_pipeline = [ctx.device newComputePipelineStateWithFunction:func error:&err];
        if (!s_pipeline || err)
        {
            CV_Error(cv::Error::StsError, "Failed to create simpleSegmentationKernel pipeline");
        }
    }
    return s_pipeline;
}

id<MTLComputePipelineState> MetalGraphCut::getBuildGraphPipeline()
{
    static id<MTLComputePipelineState> s_pipeline = nil;
    if (!s_pipeline)
    {
        MetalContext& ctx = MetalContext::getInstance();
        std::string src = std::string(kGraphCutCommonSrc) + kGraphCutBuildSrc;
        id<MTLFunction> fn = ctx.getMetalFunction(src, "buildGraphKernel");
        if (!fn) return nil;
        NSError* err = nil;
        s_pipeline = [ctx.device newComputePipelineStateWithFunction:fn error:&err];
        if (!s_pipeline || err)
            CV_Error(cv::Error::StsError, "Failed to create buildGraphKernel pipeline");
    }
    return s_pipeline;
}

id<MTLComputePipelineState> getBuildGraphAtomPipeline()
{
    static id<MTLComputePipelineState> s_pipeline=nil;
    if(!s_pipeline){
        MetalContext& ctx=MetalContext::getInstance();
        std::string src=std::string(kGraphCutCommonSrc)+kGraphCutBuildAtomSrc;
        id<MTLFunction> fn=ctx.getMetalFunction(src,"buildGraphAtomKernel");
        if(!fn) return nil; NSError* err=nil;
        s_pipeline=[ctx.device newComputePipelineStateWithFunction:fn error:&err];
        if(!s_pipeline||err) CV_Error(cv::Error::StsError,"Failed to create buildGraphAtom pipeline");
    }
    return s_pipeline;
}

id<MTLComputePipelineState> getPushAtomPipeline()
{
    static id<MTLComputePipelineState> s_pipe = nil;
    if (!s_pipe)
    {
        MetalContext& ctx = MetalContext::getInstance();
        std::string src = std::string(kGraphCutCommonSrc) + kGraphCutPushAtomSrc;
        id<MTLFunction> fn = ctx.getMetalFunction(src, "pushAtomKernel");
        if (!fn) return nil;
        NSError* err = nil;
        s_pipe = [ctx.device newComputePipelineStateWithFunction:fn error:&err];
        if (!s_pipe || err)
            CV_Error(cv::Error::StsError, "Failed to create pushAtom pipeline");
    }
    return s_pipe;
}

id<MTLComputePipelineState> MetalGraphCut::getBFSInitPipeline()
{
    static id<MTLComputePipelineState> s_pipeline = nil;
    if (!s_pipeline)
    {
        MetalContext& ctx = MetalContext::getInstance();
        std::string src = std::string(kGraphCutCommonSrc) + kGraphCutBFSInitSrc;
        id<MTLFunction> fn = ctx.getMetalFunction(src, "bfsInitKernel");
        if (!fn) return nil;
        NSError* err = nil;
        s_pipeline = [ctx.device newComputePipelineStateWithFunction:fn error:&err];
        if (!s_pipeline || err)
            CV_Error(cv::Error::StsError, "Failed to create bfsInitKernel pipeline");
    }
    return s_pipeline;
}

id<MTLComputePipelineState> getBFSInitAtomPipeline()
{
    static id<MTLComputePipelineState> s_pipe=nil;
    if(!s_pipe){
        MetalContext& ctx = MetalContext::getInstance();
        std::string src = std::string(kGraphCutCommonSrc) + kGraphCutBFSInitAtomSrc;
        id<MTLFunction> fn = ctx.getMetalFunction(src, "bfsInitAtomKernel");
        if(!fn) return nil;
        NSError* err = nil;
        s_pipe = [ctx.device newComputePipelineStateWithFunction:fn error:&err];
        if(!s_pipe || err) CV_Error(cv::Error::StsError, "Failed to create bfsInitAtom pipeline");
    }
    return s_pipe;
}

id<MTLComputePipelineState> MetalGraphCut::getBFSTraversePipeline()
{
    static id<MTLComputePipelineState> s_pipeline = nil;
    if (!s_pipeline)
    {
        MetalContext& ctx = MetalContext::getInstance();
        id<MTLFunction> fn = ctx.getMetalFunction(kGraphCutBFSTraverseSrc, "bfsTraverseKernel");
        if (!fn) return nil;
        NSError* err = nil;
        s_pipeline = [ctx.device newComputePipelineStateWithFunction:fn error:&err];
        if (!s_pipeline || err)
            CV_Error(cv::Error::StsError, "Failed to create bfsTraverseKernel pipeline");
    }
    return s_pipeline;
}

id<MTLComputePipelineState> getBFSTraverseAtomPipeline()
{
    static id<MTLComputePipelineState> s_pipe = nil;
    if (!s_pipe)
    {
        MetalContext &ctx = MetalContext::getInstance();
        std::string src = std::string(kGraphCutCommonSrc) + kGraphCutBFSTraverseAtomSrc;
        id<MTLFunction> fn = ctx.getMetalFunction(src, "bfsTraverseAtomKernel");
        if (!fn) return nil;
        NSError *err = nil;
        s_pipe = [ctx.device newComputePipelineStateWithFunction:fn error:&err];
        if (!s_pipe || err)
        {
            CV_Error(cv::Error::StsError, "Failed to create bfsTraverseAtom pipeline");
        }
    }
    return s_pipe;
}

id<MTLComputePipelineState> MetalGraphCut::getLabelSegmentationPipeline()
{
    static id<MTLComputePipelineState> s_pipeline = nil;
    if (!s_pipeline)
    {
        MetalContext& ctx = MetalContext::getInstance();
        std::string src = std::string(kGraphCutCommonSrc) + kGraphCutLabelSegSrc;
        id<MTLFunction> fn = ctx.getMetalFunction(src, "labelSegmentationKernel");
        if (!fn) return nil;
        NSError* err = nil;
        s_pipeline = [ctx.device newComputePipelineStateWithFunction:fn error:&err];
        if (!s_pipeline || err)
            CV_Error(cv::Error::StsError, "Failed to create labelSegmentationKernel pipeline");
    }
    return s_pipeline;
}

id<MTLComputePipelineState> getPushPipeline()
{
    static id<MTLComputePipelineState> s_pipe=nil;
    if(!s_pipe){
        auto &ctx=MetalContext::getInstance();
        id<MTLFunction> fn=ctx.getMetalFunction(kGraphCutPushSrc,"pushKernel");
        if(!fn) return nil; NSError*err=nil;
        s_pipe=[ctx.device newComputePipelineStateWithFunction:fn error:&err];
    }
    return s_pipe;
}

id<MTLComputePipelineState> getCheckExcessPipeline()
{
    static id<MTLComputePipelineState> s_pipe=nil;
    if(!s_pipe){
        auto &ctx=MetalContext::getInstance();
        id<MTLFunction> fn=ctx.getMetalFunction(kGraphCutCheckExcessSrc,"checkExcessKernel");
        if(!fn) return nil; NSError*err=nil;
        s_pipe=[ctx.device newComputePipelineStateWithFunction:fn error:&err];
    }
    return s_pipe;
}

// getter for relabel pipeline
static id<MTLComputePipelineState> getRelabelAtomPipeline()
{
    static id<MTLComputePipelineState> s_pipe = nil;
    if (!s_pipe)
    {
        MetalContext &ctx = MetalContext::getInstance();
        std::string src = std::string(kGraphCutCommonSrc) + kGraphCutRelabelAtomSrc;
        id<MTLFunction> fn = ctx.getMetalFunction(src, "relabelAtomKernel");
        if (!fn) return nil;
        NSError *err = nil;
        s_pipe = [ctx.device newComputePipelineStateWithFunction:fn error:&err];
        if (!s_pipe || err)
        {
            CV_Error(cv::Error::StsError, "Failed to create relabelAtom pipeline");
        }
    }
    return s_pipe;
}

// New push-relabel pipeline getters
id<MTLComputePipelineState> MetalGraphCut::getGlobalRelabelInitPipeline()
{
    static id<MTLComputePipelineState> s_pipeline = nil;
    if (!s_pipeline)
    {
        MetalContext& ctx = MetalContext::getInstance();
        std::string src = std::string(kGraphCutCommonSrc) + kGraphCutGlobalRelabelInitSrc;
        id<MTLFunction> fn = ctx.getMetalFunction(src, "globalRelabelInitKernel");
        if (!fn) return nil;
        NSError* err = nil;
        s_pipeline = [ctx.device newComputePipelineStateWithFunction:fn error:&err];
        if (!s_pipeline || err)
            CV_Error(cv::Error::StsError, "Failed to create globalRelabelInitKernel pipeline");
    }
    return s_pipeline;
}

id<MTLComputePipelineState> MetalGraphCut::getGlobalRelabelBfsTraversePipeline()
{
    static id<MTLComputePipelineState> s_pipeline = nil;
    if (!s_pipeline)
    {
        MetalContext& ctx = MetalContext::getInstance();
        std::string src = std::string(kGraphCutCommonSrc) + kGraphCutGlobalRelabelBfsTraverseSrc;
        id<MTLFunction> fn = ctx.getMetalFunction(src, "globalRelabelBfsTraverseKernel");
        if (!fn) return nil;
        NSError* err = nil;
        s_pipeline = [ctx.device newComputePipelineStateWithFunction:fn error:&err];
        if (!s_pipeline || err)
            CV_Error(cv::Error::StsError, "Failed to create globalRelabelBfsTraverseKernel pipeline");
    }
    return s_pipeline;
}

id<MTLComputePipelineState> MetalGraphCut::getCreateInitialActiveListPipeline()
{
    static id<MTLComputePipelineState> s_pipeline = nil;
    if (!s_pipeline)
    {
        MetalContext& ctx = MetalContext::getInstance();
        std::string src = std::string(kGraphCutCommonSrc) + kGraphCutCreateActiveListSrc;
        id<MTLFunction> fn = ctx.getMetalFunction(src, "createInitialActiveListKernel");
        if (!fn) return nil;
        NSError* err = nil;
        s_pipeline = [ctx.device newComputePipelineStateWithFunction:fn error:&err];
        if (!s_pipeline || err)
            CV_Error(cv::Error::StsError, "Failed to create createInitialActiveListKernel pipeline");
    }
    return s_pipeline;
}

id<MTLComputePipelineState> MetalGraphCut::getPushRelabelPipeline()
{
    static id<MTLComputePipelineState> s_pipeline = nil;
    if (!s_pipeline)
    {
        MetalContext& ctx = MetalContext::getInstance();
        std::string src = std::string(kGraphCutCommonSrc) + kGraphCutPushRelabelSrc;
        id<MTLFunction> fn = ctx.getMetalFunction(src, "pushRelabelKernel");
        if (!fn) return nil;
        NSError* err = nil;
        s_pipeline = [ctx.device newComputePipelineStateWithFunction:fn error:&err];
        if (!s_pipeline || err)
            CV_Error(cv::Error::StsError, "Failed to create pushRelabelKernel pipeline");
    }
    return s_pipeline;
}

id<MTLComputePipelineState> MetalGraphCut::getBuildHeightHistogramPipeline()
{
    static id<MTLComputePipelineState> s_pipeline = nil;
    if (!s_pipeline)
    {
        MetalContext& ctx = MetalContext::getInstance();
        std::string src = std::string(kGraphCutCommonSrc) + kGraphCutBuildHeightHistogramSrc;
        id<MTLFunction> fn = ctx.getMetalFunction(src, "buildHeightHistogramKernel");
        if (!fn) return nil;
        NSError* err = nil;
        s_pipeline = [ctx.device newComputePipelineStateWithFunction:fn error:&err];
        if (!s_pipeline || err)
            CV_Error(cv::Error::StsError, "Failed to create buildHeightHistogramKernel pipeline");
    }
    return s_pipeline;
}

id<MTLComputePipelineState> MetalGraphCut::getGapRelabelPipeline()
{
    static id<MTLComputePipelineState> s_pipeline = nil;
    if (!s_pipeline)
    {
        MetalContext& ctx = MetalContext::getInstance();
        std::string src = std::string(kGraphCutCommonSrc) + kGraphCutGapRelabelSrc;
        id<MTLFunction> fn = ctx.getMetalFunction(src, "gapRelabelKernel");
        if (!fn) return nil;
        NSError* err = nil;
        s_pipeline = [ctx.device newComputePipelineStateWithFunction:fn error:&err];
        if (!s_pipeline || err)
            CV_Error(cv::Error::StsError, "Failed to create gapRelabelKernel pipeline");
    }
    return s_pipeline;
}

id<MTLComputePipelineState> MetalGraphCut::getFinalCutBfsInitPipeline()
{
    static id<MTLComputePipelineState> s_pipeline = nil;
    if (!s_pipeline)
    {
        MetalContext& ctx = MetalContext::getInstance();
        std::string src = std::string(kGraphCutCommonSrc) + kGraphCutFinalCutBfsInitSrc;
        id<MTLFunction> fn = ctx.getMetalFunction(src, "finalCutBfsInitKernel");
        if (!fn) return nil;
        NSError* err = nil;
        s_pipeline = [ctx.device newComputePipelineStateWithFunction:fn error:&err];
        if (!s_pipeline || err)
            CV_Error(cv::Error::StsError, "Failed to create finalCutBfsInitKernel pipeline");
    }
    return s_pipeline;
}

id<MTLComputePipelineState> MetalGraphCut::getFinalCutBfsTraversePipeline()
{
    static id<MTLComputePipelineState> s_pipeline = nil;
    if (!s_pipeline)
    {
        MetalContext& ctx = MetalContext::getInstance();
        std::string src = std::string(kGraphCutCommonSrc) + kGraphCutFinalCutBfsTraverseSrc;
        id<MTLFunction> fn = ctx.getMetalFunction(src, "finalCutBfsTraverseKernel");
        if (!fn) return nil;
        NSError* err = nil;
        s_pipeline = [ctx.device newComputePipelineStateWithFunction:fn error:&err];
        if (!s_pipeline || err)
            CV_Error(cv::Error::StsError, "Failed to create finalCutBfsTraverseKernel pipeline");
    }
    return s_pipeline;
}

id<MTLComputePipelineState> MetalGraphCut::getFinalCutWriteMaskPipeline()
{
    static id<MTLComputePipelineState> s_pipeline = nil;
    if (!s_pipeline)
    {
        MetalContext& ctx = MetalContext::getInstance();
        std::string src = std::string(kGraphCutCommonSrc) + kGraphCutFinalCutWriteMaskSrc;
        id<MTLFunction> fn = ctx.getMetalFunction(src, "finalCutWriteMaskKernel");
        if (!fn) return nil;
        NSError* err = nil;
        s_pipeline = [ctx.device newComputePipelineStateWithFunction:fn error:&err];
        if (!s_pipeline || err)
            CV_Error(cv::Error::StsError, "Failed to create finalCutWriteMaskKernel pipeline");
    }
    return s_pipeline;
}

id<MTLComputePipelineState> MetalGraphCut::getSimpleFinalCutPipeline()
{
    static id<MTLComputePipelineState> s_pipeline = nil;
    if (!s_pipeline)
    {
        MetalContext& ctx = MetalContext::getInstance();
        std::string src = std::string(kGraphCutCommonSrc) + kGraphCutSimpleFinalCutSrc;
        id<MTLFunction> fn = ctx.getMetalFunction(src, "simpleFinalCutKernel");
        if (!fn) return nil;
        NSError* err = nil;
        s_pipeline = [ctx.device newComputePipelineStateWithFunction:fn error:&err];
        if (!s_pipeline || err)
            CV_Error(cv::Error::StsError, "Failed to create simpleFinalCutKernel pipeline");
    }
    return s_pipeline;
}

namespace cv { namespace metal {

void MetalGraphCut::runGlobalRelabel()
{
    NSUInteger nodeCount = m_graphSize.width * m_graphSize.height;
    uint32_t totalNodes = (uint32_t)nodeCount;
    uint32_t width = (uint32_t)m_graphSize.width;
    uint32_t height = (uint32_t)m_graphSize.height;

    id<MTLComputePipelineState> globalRelabelInitPipe = getGlobalRelabelInitPipeline();
    CV_Assert(globalRelabelInitPipe);

    uint32_t zero = 0;
    memcpy([m_levelCount contents], &zero, sizeof(uint32_t));

    @autoreleasepool {
        id<MTLComputeCommandEncoder> enc = StreamAccessor::createComputeEncoder(m_stream);
        [enc setComputePipelineState:globalRelabelInitPipe];
        [enc setBuffer:m_nodeDataAtom offset:0 atIndex:0];
        [enc setBuffer:m_terminalFlow offset:0 atIndex:1];
        [enc setBuffer:m_activeList1 offset:0 atIndex:2];
        [enc setBuffer:m_levelCount offset:0 atIndex:3];
        [enc setBytes:&totalNodes length:sizeof(uint32_t) atIndex:4];
        MTLSize tg = MTLSizeMake(256, 1, 1);
        MTLSize grid = MTLSizeMake((totalNodes + tg.width - 1) / tg.width, 1, 1);
        [enc dispatchThreadgroups:grid threadsPerThreadgroup:tg];
        [enc endEncoding];
    }
    
    id<MTLComputePipelineState> globalRelabelBfsPipe = getGlobalRelabelBfsTraversePipeline();
    CV_Assert(globalRelabelBfsPipe);

    m_stream.syncCPU();
    uint32_t queueCount = *(uint32_t*)[m_levelCount contents];
    id<MTLBuffer> currentQueue = m_activeList1;
    id<MTLBuffer> nextQueue = m_activeList2;
    
    uint32_t bfsIter = 0;
    while (queueCount > 0 && bfsIter < (width + height)) { // Limit BFS iterations
        memcpy([m_levelCount contents], &zero, sizeof(uint32_t));
        
        @autoreleasepool {
            id<MTLComputeCommandEncoder> enc = StreamAccessor::createComputeEncoder(m_stream);
            [enc setComputePipelineState:globalRelabelBfsPipe];
            [enc setBuffer:m_nodeDataAtom offset:0 atIndex:0];
            [enc setBuffer:m_residualAtom offset:0 atIndex:1];
            [enc setBuffer:currentQueue offset:0 atIndex:2];
            [enc setBytes:&queueCount length:sizeof(uint32_t) atIndex:3];
            [enc setBuffer:m_levelCount offset:0 atIndex:4];
            [enc setBuffer:nextQueue offset:0 atIndex:5];
            [enc setBytes:&width length:sizeof(uint32_t) atIndex:6];
            [enc setBytes:&height length:sizeof(uint32_t) atIndex:7];
            MTLSize tg = MTLSizeMake(256, 1, 1);
            MTLSize grid = MTLSizeMake((queueCount + tg.width - 1) / tg.width, 1, 1);
            [enc dispatchThreadgroups:grid threadsPerThreadgroup:tg];
            [enc endEncoding];
        }
        
        m_stream.syncCPU();
        queueCount = *(uint32_t*)[m_levelCount contents];
        std::swap(currentQueue, nextQueue);
        ++bfsIter;
    }
}

void MetalGraphCut::createActiveList()
{
    NSUInteger nodeCount = m_graphSize.width * m_graphSize.height;
    uint32_t totalNodes = (uint32_t)nodeCount;

    id<MTLComputePipelineState> createActivePipe = getCreateInitialActiveListPipeline();
    CV_Assert(createActivePipe);

    uint32_t zero = 0;
    memcpy([m_levelCount contents], &zero, sizeof(uint32_t));

    @autoreleasepool {
        id<MTLComputeCommandEncoder> enc = StreamAccessor::createComputeEncoder(m_stream);
        [enc setComputePipelineState:createActivePipe];
        [enc setBuffer:m_nodeDataAtom offset:0 atIndex:0];
        [enc setBuffer:m_activeList1 offset:0 atIndex:1];
        [enc setBuffer:m_levelCount offset:0 atIndex:2];
        [enc setBytes:&totalNodes length:sizeof(uint32_t) atIndex:3];
        MTLSize tg = MTLSizeMake(256, 1, 1);
        MTLSize grid = MTLSizeMake((totalNodes + tg.width - 1) / tg.width, 1, 1);
        [enc dispatchThreadgroups:grid threadsPerThreadgroup:tg];
        [enc endEncoding];
    }
}

MetalGraphCut::MetalGraphCut(cv::Size sz, Stream& s)
    : m_stream(s), m_graphSize(sz)
{
    allocateGraphBuffers();
}

void MetalGraphCut::allocateGraphBuffers()
{
    MetalContext& ctx = MetalContext::getInstance();
    id<MTLDevice> dev = ctx.device;

    NSUInteger count = m_graphSize.width * m_graphSize.height;

    if (!m_nodeData)
        m_nodeData = [dev newBufferWithLength:count * sizeof(float) * 2
                                        options:MTLResourceStorageModePrivate];
    if (!m_terminalFlow)
        m_terminalFlow = [dev newBufferWithLength:count * sizeof(float) * 2
                                          options:MTLResourceStorageModeShared];
    if (!m_residualCap)
        m_residualCap = [dev newBufferWithLength:count * sizeof(simd_float4)
                                         options:MTLResourceStorageModePrivate];

    // Active node buffers (ping-pong buffers for push-relabel)
    if (!m_activeList1)
        m_activeList1 = [dev newBufferWithLength:count * sizeof(uint32_t)
                                    options:MTLResourceStorageModePrivate];
    if (!m_activeList2)
        m_activeList2 = [dev newBufferWithLength:count * sizeof(uint32_t)
                                    options:MTLResourceStorageModePrivate];
    if (!m_levelCount)
        m_levelCount = [dev newBufferWithLength:sizeof(uint32_t)
                                    options:MTLResourceStorageModeShared];

    if (!m_excessFlag)
        m_excessFlag = [dev newBufferWithLength:sizeof(uint32_t)
                                   options:MTLResourceStorageModeShared];
    
    // Height histogram for gap optimization (shared memory for CPU access)
    NSUInteger max_height = count; // worst case: all nodes at different heights
    if (!m_heightHistogram)
        m_heightHistogram = [dev newBufferWithLength:max_height * sizeof(uint32_t)
                                            options:MTLResourceStorageModeShared];
    
    // Final cut reachability labels
    if (!m_finalCutLabels)
        m_finalCutLabels = [dev newBufferWithLength:count * sizeof(int32_t)
                                            options:MTLResourceStorageModePrivate];

    // Slice 1: allocate atomic buffers (NodeDataAtom 8 bytes, ResidualGraphAtom 32 bytes)
    if (!m_nodeDataAtom)
        m_nodeDataAtom = [dev newBufferWithLength:count * 8
                                          options:MTLResourceStorageModeShared];
    if (!m_residualAtom)
        m_residualAtom = [dev newBufferWithLength:count * 32
                                          options:MTLResourceStorageModePrivate];
}

void MetalGraphCut::buildGraph(const MetalMat& unary_bg,
                               const MetalMat& unary_fg,
                               const MetalMat& pairwise_left,
                               const MetalMat& pairwise_top,
                               const MetalMat& pairwise_topleft,
                               const MetalMat& pairwise_topright,
                               const MetalMat& mask,
                               double lambda)
{
    // Store texture references for the provisional segmentation kernel.
    m_bgTermTex = unary_bg.texture();
    m_fgTermTex = unary_fg.texture();

    // Build graph capacities for push-relabel (partial – only terminal weights for now)
    id<MTLComputePipelineState> pipeline = getBuildGraphPipeline();
    CV_Assert(pipeline);

    @autoreleasepool {
        id<MTLComputeCommandEncoder> enc = StreamAccessor::createComputeEncoder(m_stream);
        [enc setComputePipelineState:pipeline];
        [enc setTexture:unary_bg.texture() atIndex:0];
        [enc setTexture:unary_fg.texture() atIndex:1];
        [enc setTexture:pairwise_left.texture() atIndex:2];
        [enc setTexture:pairwise_topleft.texture() atIndex:3];
        [enc setTexture:pairwise_top.texture() atIndex:4];
        [enc setTexture:pairwise_topright.texture() atIndex:5];
        [enc setTexture:mask.texture() atIndex:6];

        [enc setBuffer:m_nodeData offset:0 atIndex:0];
        [enc setBuffer:m_terminalFlow offset:0 atIndex:1];
        [enc setBuffer:m_residualCap offset:0 atIndex:2];
        
        float lambdaFloat = (float)lambda;
        [enc setBytes:&lambdaFloat length:sizeof(float) atIndex:3];

        MTLSize tg = MTLSizeMake(16,16,1);
        MTLSize grid = MTLSizeMake((m_graphSize.width  + tg.width  - 1)/tg.width,
                                   (m_graphSize.height + tg.height - 1)/tg.height,1);
        [enc dispatchThreadgroups:grid threadsPerThreadgroup:tg];
        [enc endEncoding];
    }

    // CRITICAL: Synchronize before atomic build to prevent race condition
    m_stream.syncCPU();

    // --- New atomic graph build ---
    id<MTLComputePipelineState> atomPipe = getBuildGraphAtomPipeline();
    printf("[DEBUG] atomPipe = %p\n", atomPipe);
    if(atomPipe){
        printf("[DEBUG] Atomic pipeline created successfully, dispatching kernel\n");
        // DEBUG: Create a temporary buffer to receive debug info from the kernel
        MetalContext& ctx = MetalContext::getInstance();
        id<MTLDevice> dev = ctx.device;
        struct BuildGraphDebugInfo {
            float bg_cost;
            float fg_cost;
            uint32_t mask_val;
            float unary_bg;
            float unary_fg;
        };
        id<MTLBuffer> debugBuf = [dev newBufferWithLength:sizeof(BuildGraphDebugInfo) * 2 options:MTLResourceStorageModeShared];

        // DEBUG: CPU-side validation of mask texture content
        {
            m_stream.syncCPU(); // Ensure any pending writes to mask are complete

            id<MTLTexture> tex = mask.texture();
            printf("[MetalGraphCut DEBUG] Mask texture pixel format: %lu, width: %lu, height: %lu\n",
                   (unsigned long)[tex pixelFormat],
                   (unsigned long)[tex width],
                   (unsigned long)[tex height]);

            NSUInteger bytesPerRow = tex.width * 1; // R8Uint is 1 byte per pixel
            NSUInteger bytesPerImage = bytesPerRow * tex.height;
            id<MTLBuffer> readbackBuf = [dev newBufferWithLength:bytesPerImage options:MTLResourceStorageModeShared];

            // Create a blit encoder to do the copy
            id<MTLBlitCommandEncoder> blitEnc = StreamAccessor::createBlitEncoder(m_stream);
            [blitEnc copyFromTexture:tex
                         sourceSlice:0
                         sourceLevel:0
                        sourceOrigin:MTLOriginMake(0,0,0)
                          sourceSize:MTLSizeMake(tex.width, tex.height, 1)
                            toBuffer:readbackBuf
                   destinationOffset:0
              destinationBytesPerRow:bytesPerRow
            destinationBytesPerImage:bytesPerImage];
            [blitEnc endEncoding];
            m_stream.syncCPU(); // wait for copy to complete

            // Now read from the buffer
            cv::Mat downloaded_mask(mask.size(), CV_8UC1, [readbackBuf contents], bytesPerRow);
            printf("[MetalGraphCut DEBUG] CPU-side mask validation at (102, 192): value=%d\n", downloaded_mask.at<uchar>(192, 102));
            printf("[MetalGraphCut DEBUG] CPU-side mask validation at (204, 192): value=%d\n", downloaded_mask.at<uchar>(192, 204));
        }

        @autoreleasepool{
            id<MTLComputeCommandEncoder> enc = StreamAccessor::createComputeEncoder(m_stream);
            [enc setComputePipelineState:atomPipe];
            [enc setTexture:unary_bg.texture() atIndex:0];
            [enc setTexture:unary_fg.texture() atIndex:1];
            [enc setTexture:pairwise_left.texture()      atIndex:2];
            [enc setTexture:pairwise_topleft.texture()   atIndex:3];
            [enc setTexture:pairwise_top.texture()       atIndex:4];
            [enc setTexture:pairwise_topright.texture()  atIndex:5];
            [enc setTexture:mask.texture()               atIndex:6];

            [enc setBuffer:m_nodeDataAtom offset:0 atIndex:0];
            [enc setBuffer:m_terminalFlow offset:0 atIndex:1];
            [enc setBuffer:m_residualAtom offset:0 atIndex:2];

            float lambdaFloat = (float)lambda;
            [enc setBytes:&lambdaFloat length:sizeof(float) atIndex:3];
            [enc setBuffer:debugBuf offset:0 atIndex:4]; // Pass debug buffer

            MTLSize tg=MTLSizeMake(16,16,1);
            MTLSize grid=MTLSizeMake((m_graphSize.width+15)/16,(m_graphSize.height+15)/16,1);
            printf("[DEBUG] Dispatching atomic kernel: grid=(%lu,%lu), tg=(%lu,%lu), graphSize=(%d,%d)\n",
                   grid.width, grid.height, tg.width, tg.height, m_graphSize.width, m_graphSize.height);
            [enc dispatchThreadgroups:grid threadsPerThreadgroup:tg];
            [enc endEncoding];
            printf("[DEBUG] Atomic kernel dispatch completed\n");
        }

        // DEBUG: Read and print the debug info
        m_stream.syncCPU();
        BuildGraphDebugInfo* debugData = (BuildGraphDebugInfo*)[debugBuf contents];
        printf("[MetalGraphCut DEBUG] Unary costs at (102,192): bg_cost=%.4f, fg_cost=%.4f, mask=%u, unary_bg=%.4f, unary_fg=%.4f\n",
               debugData[0].bg_cost, debugData[0].fg_cost, debugData[0].mask_val, debugData[0].unary_bg, debugData[0].unary_fg);
        printf("[MetalGraphCut DEBUG] Unary costs at (204,192): bg_cost=%.4f, fg_cost=%.4f, mask=%u, unary_bg=%.4f, unary_fg=%.4f\n",
               debugData[1].bg_cost, debugData[1].fg_cost, debugData[1].mask_val, debugData[1].unary_bg, debugData[1].unary_fg);
    }

    // Initialize atomic frontier list based on labels (currently labels are 0; no frontier). For now we just clear levelCount.
    uint32_t zero32=0; memcpy([m_levelCount contents], &zero32, sizeof(uint32_t));

    id<MTLComputePipelineState> bfsInitAtom = getBFSInitAtomPipeline();
    if(bfsInitAtom){
        @autoreleasepool{
            id<MTLComputeCommandEncoder> enc = StreamAccessor::createComputeEncoder(m_stream);
            [enc setComputePipelineState:bfsInitAtom];
            [enc setBuffer:m_nodeDataAtom offset:0 atIndex:0];
            [enc setBuffer:m_levelCount offset:0 atIndex:1];
            [enc setBuffer:m_activeList1 offset:0 atIndex:2];
            uint total=(uint)(m_graphSize.width*m_graphSize.height);
            [enc setBytes:&total length:sizeof(uint) atIndex:3];

            MTLSize tg=MTLSizeMake(256,1,1);
            MTLSize grid=MTLSizeMake((total+tg.width-1)/tg.width,1,1);
            [enc dispatchThreadgroups:grid threadsPerThreadgroup:tg];
            [enc endEncoding];
        }
    }
}

void MetalGraphCut::solve(int maxIterations)
{
    NSUInteger nodeCount = m_graphSize.width * m_graphSize.height;

    // --- DEBUG: Print initial graph stats ---
    m_stream.syncCPU(); // Ensure buildGraph is complete
    struct CppNodeDataAtom {
        uint32_t excessBits;
        int32_t  label;
    };
    const CppNodeDataAtom* nodeData = (const CppNodeDataAtom*)[m_nodeDataAtom contents];
    double totalExcess = 0;
    for (NSUInteger i = 0; i < nodeCount; ++i) {
        float excess;
        uint32_t excess_bits = nodeData[i].excessBits;
        memcpy(&excess, &excess_bits, sizeof(float));
        totalExcess += excess;
    }
    printf("[MetalGraphCut DEBUG] Total initial excess: %f\n", totalExcess);
    // --- END DEBUG ---

    uint32_t totalNodes = (uint32_t)nodeCount;
    uint32_t width = (uint32_t)m_graphSize.width;
    uint32_t height = (uint32_t)m_graphSize.height;
    
    // Phase 1: Global relabel initialization
    id<MTLComputePipelineState> globalRelabelInitPipe = getGlobalRelabelInitPipeline();
    if (!globalRelabelInitPipe) {
        CV_Error(cv::Error::StsError, "Failed to get global relabel init pipeline");
        return;
    }
    
    // Reset queue counter
    uint32_t zero = 0;
    memcpy([m_levelCount contents], &zero, sizeof(uint32_t));
    
    // Initialize heights and create initial BFS frontier
    @autoreleasepool {
        id<MTLComputeCommandEncoder> enc = StreamAccessor::createComputeEncoder(m_stream);
        [enc setComputePipelineState:globalRelabelInitPipe];
        [enc setBuffer:m_nodeDataAtom offset:0 atIndex:0];
        [enc setBuffer:m_terminalFlow offset:0 atIndex:1];
        [enc setBuffer:m_activeList1 offset:0 atIndex:2]; // BFS queue
        [enc setBuffer:m_levelCount offset:0 atIndex:3];   // Queue counter
        [enc setBytes:&totalNodes length:sizeof(uint32_t) atIndex:4];
        
        MTLSize tg = MTLSizeMake(256, 1, 1);
        MTLSize grid = MTLSizeMake((totalNodes + tg.width - 1) / tg.width, 1, 1);
        [enc dispatchThreadgroups:grid threadsPerThreadgroup:tg];
        [enc endEncoding];
    }
    
    // Phase 1: BFS traversal for global relabel
    id<MTLComputePipelineState> globalRelabelBfsPipe = getGlobalRelabelBfsTraversePipeline();
    if (!globalRelabelBfsPipe) {
        CV_Error(cv::Error::StsError, "Failed to get global relabel BFS pipeline");
        return;
    }
    
    m_stream.syncCPU();
    uint32_t queueCount = *(uint32_t*)[m_levelCount contents];
    id<MTLBuffer> currentQueue = m_activeList1;
    id<MTLBuffer> nextQueue = m_activeList2;
    
    uint32_t bfsIter = 0;
    while (queueCount > 0 && bfsIter < 1000) {
        // Reset next queue counter
        uint32_t zero = 0;
        memcpy([m_levelCount contents], &zero, sizeof(uint32_t));
        
        @autoreleasepool {
            id<MTLComputeCommandEncoder> enc = StreamAccessor::createComputeEncoder(m_stream);
            [enc setComputePipelineState:globalRelabelBfsPipe];
            [enc setBuffer:m_nodeDataAtom offset:0 atIndex:0];
            [enc setBuffer:m_residualAtom offset:0 atIndex:1];
            [enc setBuffer:currentQueue offset:0 atIndex:2];
            [enc setBytes:&queueCount length:sizeof(uint32_t) atIndex:3];
            [enc setBuffer:m_levelCount offset:0 atIndex:4]; // Next queue counter
            [enc setBuffer:nextQueue offset:0 atIndex:5];
            [enc setBytes:&width length:sizeof(uint32_t) atIndex:6];
            [enc setBytes:&height length:sizeof(uint32_t) atIndex:7];
            
            MTLSize tg = MTLSizeMake(256, 1, 1);
            MTLSize grid = MTLSizeMake((queueCount + tg.width - 1) / tg.width, 1, 1);
            [enc dispatchThreadgroups:grid threadsPerThreadgroup:tg];
            [enc endEncoding];
        }
        
        m_stream.syncCPU();
        queueCount = *(uint32_t*)[m_levelCount contents];
        std::swap(currentQueue, nextQueue);
        ++bfsIter;
    }
    
    // Phase 1: Create initial active list from nodes with excess flow
    id<MTLComputePipelineState> createActivePipe = getCreateInitialActiveListPipeline();
    if (!createActivePipe) {
        CV_Error(cv::Error::StsError, "Failed to get create active list pipeline");
        return;
    }
    
    // Reset active counter
    memcpy([m_levelCount contents], &zero, sizeof(uint32_t));
    
    @autoreleasepool {
        id<MTLComputeCommandEncoder> enc = StreamAccessor::createComputeEncoder(m_stream);
        [enc setComputePipelineState:createActivePipe];
        [enc setBuffer:m_nodeDataAtom offset:0 atIndex:0];
        [enc setBuffer:m_activeList1 offset:0 atIndex:1]; // Active list
        [enc setBuffer:m_levelCount offset:0 atIndex:2];   // Active counter
        [enc setBytes:&totalNodes length:sizeof(uint32_t) atIndex:3];
        
        MTLSize tg = MTLSizeMake(256, 1, 1);
        MTLSize grid = MTLSizeMake((totalNodes + tg.width - 1) / tg.width, 1, 1);
        [enc dispatchThreadgroups:grid threadsPerThreadgroup:tg];
        [enc endEncoding];
    }
    
    // Phase 2: Main push-relabel loop
    id<MTLComputePipelineState> pushRelabelPipe = getPushRelabelPipeline();
    if (!pushRelabelPipe) {
        CV_Error(cv::Error::StsError, "Failed to get push-relabel pipeline");
        return;
    }
    
    m_stream.syncCPU();
    uint32_t activeCount = *(uint32_t*)[m_levelCount contents];
    printf("[MetalGraphCut DEBUG] Initial active nodes: %u\n", activeCount);
    id<MTLBuffer> activeListIn = m_activeList1;
    id<MTLBuffer> activeListOut = m_activeList2;
    
    int iter = 0;
    int globalRelabelFreq = std::max(1, (int)sqrt(totalNodes)); // Global relabel frequency
    
    while (activeCount > 0 && iter < maxIterations) {
        // Reset output active counter
        uint32_t zero = 0;
        memcpy([m_levelCount contents], &zero, sizeof(uint32_t));
        
        @autoreleasepool {
            id<MTLComputeCommandEncoder> enc = StreamAccessor::createComputeEncoder(m_stream);
            [enc setComputePipelineState:pushRelabelPipe];
            [enc setBuffer:m_nodeDataAtom offset:0 atIndex:0];
            [enc setBuffer:m_residualAtom offset:0 atIndex:1];
            [enc setBuffer:m_terminalFlow offset:0 atIndex:2];
            [enc setBuffer:activeListIn offset:0 atIndex:3];
            [enc setBytes:&activeCount length:sizeof(uint32_t) atIndex:4];
            [enc setBuffer:m_levelCount offset:0 atIndex:5]; // Output active counter
            [enc setBuffer:activeListOut offset:0 atIndex:6];
            [enc setBytes:&width length:sizeof(uint32_t) atIndex:7];
            [enc setBytes:&height length:sizeof(uint32_t) atIndex:8];
            
            MTLSize tg = MTLSizeMake(256, 1, 1);
            MTLSize grid = MTLSizeMake((activeCount + tg.width - 1) / tg.width, 1, 1);
            [enc dispatchThreadgroups:grid threadsPerThreadgroup:tg];
            [enc endEncoding];
        }
        
        m_stream.syncCPU();
        activeCount = *(uint32_t*)[m_levelCount contents];
        std::swap(activeListIn, activeListOut);
        ++iter;
        
        // Phase 3: Periodic global relabel for better convergence
        if (iter > 0 && iter % globalRelabelFreq == 0 && activeCount > 0) {
            printf("Running global relabel at iteration %d (activeCount=%u)\n", iter, activeCount);
            runGlobalRelabel();
            createActiveList();
            m_stream.syncCPU();
            activeCount = *(uint32_t*)[m_levelCount contents];
            activeListIn = m_activeList1;
            activeListOut = m_activeList2;
        }
        
        // Phase 3: Gap heuristic optimization
        if (iter % 10 == 0 && activeCount > 0) { // Check gaps every 10 iterations
            id<MTLComputePipelineState> histogramPipe = getBuildHeightHistogramPipeline();
            id<MTLComputePipelineState> gapRelabelPipe = getGapRelabelPipeline();
            
            if (histogramPipe && gapRelabelPipe) {
                // Clear histogram
                @autoreleasepool {
                    id<MTLBlitCommandEncoder> blitEnc = StreamAccessor::createBlitEncoder(m_stream);
                    [blitEnc fillBuffer:m_heightHistogram range:NSMakeRange(0, totalNodes * sizeof(uint32_t)) value:0];
                    [blitEnc endEncoding];
                }
                
                // Build height histogram
                @autoreleasepool {
                    id<MTLComputeCommandEncoder> enc = StreamAccessor::createComputeEncoder(m_stream);
                    [enc setComputePipelineState:histogramPipe];
                    [enc setBuffer:m_nodeDataAtom offset:0 atIndex:0];
                    [enc setBuffer:m_heightHistogram offset:0 atIndex:1];
                    [enc setBytes:&totalNodes length:sizeof(uint32_t) atIndex:2];
                    
                    MTLSize tg = MTLSizeMake(256, 1, 1);
                    MTLSize grid = MTLSizeMake((totalNodes + tg.width - 1) / tg.width, 1, 1);
                    [enc dispatchThreadgroups:grid threadsPerThreadgroup:tg];
                    [enc endEncoding];
                }
                
                // Gap detection: find gaps in height histogram
                m_stream.syncCPU();
                
                // Read histogram data and find gaps
                uint32_t* histogram = (uint32_t*)[m_heightHistogram contents];
                bool foundGap = false;
                uint32_t gapHeight = 0;
                
                // Look for gaps (heights with zero nodes that have higher heights with nodes)
                for (uint32_t h = 1; h < totalNodes - 1; ++h) {
                    if (histogram[h] == 0) {
                        // Check if there are nodes at higher levels
                        bool hasHigherNodes = false;
                        for (uint32_t h2 = h + 1; h2 < totalNodes; ++h2) {
                            if (histogram[h2] > 0) {
                                hasHigherNodes = true;
                                break;
                            }
                        }
                        if (hasHigherNodes) {
                            foundGap = true;
                            gapHeight = h;
                            break;
                        }
                    }
                }
                
                // Apply gap relabel if gap found
                if (foundGap) {
                    printf("Gap found at height %u, applying gap relabel\n", gapHeight);
                    @autoreleasepool {
                        id<MTLComputeCommandEncoder> enc = StreamAccessor::createComputeEncoder(m_stream);
                        [enc setComputePipelineState:gapRelabelPipe];
                        [enc setBuffer:m_nodeDataAtom offset:0 atIndex:0];
                        [enc setBytes:&gapHeight length:sizeof(uint32_t) atIndex:1];
                        [enc setBytes:&totalNodes length:sizeof(uint32_t) atIndex:2];
                        
                        MTLSize tg = MTLSizeMake(256, 1, 1);
                        MTLSize grid = MTLSizeMake((totalNodes + tg.width - 1) / tg.width, 1, 1);
                        [enc dispatchThreadgroups:grid threadsPerThreadgroup:tg];
                        [enc endEncoding];
                    }
                    
                    // Recreate active list after gap relabel
                    createActiveList();
                    m_stream.syncCPU();
                    activeCount = *(uint32_t*)[m_levelCount contents];
                    activeListIn = m_activeList1;
                    activeListOut = m_activeList2;
                }
            }
        }
    }
    
    printf("Push-relabel completed after %d iterations with %u active nodes remaining\n", iter, activeCount);
}

void MetalGraphCut::getSegmentation(MetalMat& mask, const MetalMat& initial_mask)
{
    if (mask.empty())
        mask.create(m_graphSize, CV_8UC1);

    uint32_t width = (uint32_t)m_graphSize.width;
    uint32_t height = (uint32_t)m_graphSize.height;
    uint32_t totalNodes = width * height;

    // 1. Initialize BFS: Find nodes on source-side of cut and add to initial queue
    id<MTLComputePipelineState> bfsInitPipe = getFinalCutBfsInitPipeline();
    CV_Assert(bfsInitPipe);

    uint32_t zero = 0;
    memcpy([m_levelCount contents], &zero, sizeof(uint32_t));

    @autoreleasepool {
        id<MTLComputeCommandEncoder> enc = StreamAccessor::createComputeEncoder(m_stream);
        [enc setComputePipelineState:bfsInitPipe];
        [enc setBuffer:m_nodeDataAtom offset:0 atIndex:0];
        [enc setBuffer:m_finalCutLabels offset:0 atIndex:1];
        [enc setBuffer:m_activeList1 offset:0 atIndex:2]; // BFS queue
        [enc setBuffer:m_levelCount offset:0 atIndex:3];
        [enc setBytes:&totalNodes length:sizeof(uint32_t) atIndex:4];

        MTLSize tg = MTLSizeMake(256, 1, 1);
        MTLSize grid = MTLSizeMake((totalNodes + tg.width - 1) / tg.width, 1, 1);
        [enc dispatchThreadgroups:grid threadsPerThreadgroup:tg];
        [enc endEncoding];
    }

    // 2. Traverse graph with BFS to find all source-reachable nodes
    id<MTLComputePipelineState> bfsTraversePipe = getFinalCutBfsTraversePipeline();
    CV_Assert(bfsTraversePipe);

    m_stream.syncCPU();
    uint32_t queueCount = *(uint32_t*)[m_levelCount contents];
    id<MTLBuffer> currentQueue = m_activeList1;
    id<MTLBuffer> nextQueue = m_activeList2;

    int bfsIter = 0;
    while (queueCount > 0 && bfsIter < 1000) { // Safety break
        memcpy([m_levelCount contents], &zero, sizeof(uint32_t)); // Reset next queue count

        @autoreleasepool {
            id<MTLComputeCommandEncoder> enc = StreamAccessor::createComputeEncoder(m_stream);
            [enc setComputePipelineState:bfsTraversePipe];
            [enc setBuffer:currentQueue offset:0 atIndex:0];
            [enc setBytes:&queueCount length:sizeof(uint32_t) atIndex:1];
            [enc setBuffer:m_residualAtom offset:0 atIndex:2];
            [enc setBuffer:m_finalCutLabels offset:0 atIndex:3];
            [enc setBuffer:m_levelCount offset:0 atIndex:4]; // nextCount
            [enc setBuffer:nextQueue offset:0 atIndex:5];
            [enc setBytes:&width length:sizeof(uint32_t) atIndex:6];
            [enc setBytes:&height length:sizeof(uint32_t) atIndex:7];

            MTLSize tg = MTLSizeMake(256, 1, 1);
            MTLSize grid = MTLSizeMake((queueCount + tg.width - 1) / tg.width, 1, 1);
            [enc dispatchThreadgroups:grid threadsPerThreadgroup:tg];
            [enc endEncoding];
        }

        m_stream.syncCPU();
        queueCount = *(uint32_t*)[m_levelCount contents];
        std::swap(currentQueue, nextQueue);
        bfsIter++;
    }

    // 3. Write final mask, preserving sure regions from initial_mask
    id<MTLComputePipelineState> writeMaskPipe = getFinalCutWriteMaskPipeline();
    CV_Assert(writeMaskPipe);
    
    @autoreleasepool {
        id<MTLComputeCommandEncoder> enc = StreamAccessor::createComputeEncoder(m_stream);
        [enc setComputePipelineState:writeMaskPipe];
        [enc setBuffer:m_finalCutLabels offset:0 atIndex:0];
        [enc setTexture:mask.texture() atIndex:1];
        [enc setTexture:initial_mask.texture() atIndex:2];
        [enc setBytes:&width length:sizeof(uint32_t) atIndex:3];
        
        MTLSize tg = MTLSizeMake(16, 16, 1);
        MTLSize grid = MTLSizeMake((width + tg.width - 1) / tg.width,
                                   (height + tg.height - 1) / tg.height, 1);
        [enc dispatchThreadgroups:grid threadsPerThreadgroup:tg];
        [enc endEncoding];
    }
}

}} // namespace cv::metal

#endif // HAVE_METAL 