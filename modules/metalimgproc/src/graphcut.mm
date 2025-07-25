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

struct Residual4Atom {
    atomic_uint leftBits;
    atomic_uint tlBits;
    atomic_uint topBits;
    atomic_uint trBits;
};

struct TerminalFlow {
    float to_source;
    float to_sink;
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

    termBuf[idx].to_source = bg;
    termBuf[idx].to_sink   = fg;

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

kernel void buildGraphAtomKernel(texture2d<float, access::read> bgTerm [[texture(0)]],
                                 texture2d<float, access::read> fgTerm [[texture(1)]],
                                 texture2d<float, access::read> wLeft [[texture(2)]],
                                 texture2d<float, access::read> wTL   [[texture(3)]],
                                 texture2d<float, access::read> wTop  [[texture(4)]],
                                 texture2d<float, access::read> wTR   [[texture(5)]],
                                 device NodeDataAtom* nodeAtom [[buffer(0)]],
                                 device TerminalFlow* termBuf  [[buffer(1)]],
                                 device Residual4Atom* resAtom  [[buffer(2)]],
                                 uint2 gid [[thread_position_in_grid]])
{
    if (gid.x >= bgTerm.get_width() || gid.y >= bgTerm.get_height()) return;

    uint idx = gid.y * bgTerm.get_width() + gid.x;

    float bg = bgTerm.read(gid).x;
    float fg = fgTerm.read(gid).x;

    fstore(&nodeAtom[idx].excessBits, bg - fg);
    atomic_store_explicit(&nodeAtom[idx].label, 0, memory_order_relaxed);

    fstore(&resAtom[idx].leftBits, wLeft.read(gid).x);
    fstore(&resAtom[idx].tlBits,   wTL.read(gid).x);
    fstore(&resAtom[idx].topBits,  wTop.read(gid).x);
    fstore(&resAtom[idx].trBits,   wTR.read(gid).x);
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
                                  const device Residual4Atom* resBuf [[buffer(1)]],
                                  device uint* levelIn [[buffer(2)]],
                                  constant uint& levelInCount [[buffer(3)]],
                                  device atomic_uint* nextCount [[buffer(4)]],
                                  device uint* levelOut [[buffer(5)]],
                                  constant uint& width [[buffer(6)]],
                                  uint3 gid [[thread_position_in_grid]])
{
    uint idxIn = gid.x;
    if (idxIn >= levelInCount) return;
    uint idx = levelIn[idxIn];

    int myLabel = atomic_load_explicit(&nodeBuf[idx].label, memory_order_relaxed);

    // Offsets for neighbours (left, tl, top, tr)
    int2 offs[4] = { int2(-1,0), int2(-1,-1), int2(0,-1), int2(1,-1) };

    uint x = idx % width;
    uint y = idx / width;

    for(uint k=0;k<4;++k){
        int nx = int(x)+offs[k].x;
        int ny = int(y)+offs[k].y;
        if(nx<0 || ny<0) continue;
        uint nIdx = ny*width + nx;

        // Load residual capacity from idx to neighbour
        float cap = 0.0f;
        if(k==0) cap = fload(&resBuf[idx].leftBits);
        else if(k==1) cap = fload(&resBuf[idx].tlBits);
        else if(k==2) cap = fload(&resBuf[idx].topBits);
        else cap = fload(&resBuf[idx].trBits);

        if(cap>1e-6f){
            int neighLabel = atomic_load_explicit(&nodeBuf[nIdx].label, memory_order_relaxed);
            if(neighLabel==0){
                // set label atomically
                if(atomic_compare_exchange_weak_explicit(&nodeBuf[nIdx].label, &neighLabel, myLabel+1, memory_order_relaxed, memory_order_relaxed)){
                    uint pos = atomic_fetch_add_explicit(nextCount,1u, memory_order_relaxed);
                    levelOut[pos]=nIdx;
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
                          device Residual4Atom* resBuf    [[buffer(1)]],
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
        float cap = termBuf[idx].to_sink;
        float delta = (exc < cap) ? exc : cap;
        if(delta>EPS){
            exc = exc - delta;
            termBuf[idx].to_sink = cap - delta;
            fstore(&nodeBuf[idx].excessBits, exc);
        }
    } else if(exc < -EPS){ // pull from source (negative excess)
        float need = -exc;
        float cap = termBuf[idx].to_source;
        float delta = (need < cap) ? need : cap;
        if(delta>EPS){
            exc = exc + delta;
            termBuf[idx].to_source = cap - delta;
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

struct NodeDataAtom { 
    atomic_uint excessBits; 
    atomic_int label; 
};

inline float fload(const device atomic_uint* p) {
    return as_type<float>(atomic_load_explicit(p, memory_order_relaxed));
}

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

    float toS = termBuf[idx].to_source;
    float toT = termBuf[idx].to_sink;
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

struct NodeData { float excess; int label; };

kernel void labelSegmentationKernel(const device NodeData* nodeBuf [[buffer(0)]],
                                    texture2d<uint, access::write> maskTex [[texture(0)]],
                                    constant uint& width [[buffer(1)]],
                                    uint2 gid [[thread_position_in_grid]])
{
    if (gid.x >= maskTex.get_width() || gid.y >= maskTex.get_height()) return;
    uint idx = gid.y * width + gid.x;
    uint label = (nodeBuf[idx].label > 0) ? 3u : 2u; // 3 probable FG, 2 probable BG
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

// We'll compile the BFSInit later; placeholder to satisfy function.

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
        id<MTLFunction> fn = ctx.getMetalFunction(kGraphCutLabelSegSrc, "labelSegmentationKernel");
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
        id<MTLFunction> fn = ctx.getMetalFunction(kGraphCutRelabelAtomSrc, "relabelAtomKernel");
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

namespace cv { namespace metal {

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
                                          options:MTLResourceStorageModePrivate];
    if (!m_residualCap)
        m_residualCap = [dev newBufferWithLength:count * sizeof(simd_float4)
                                         options:MTLResourceStorageModePrivate];

    // BFS level buffers and counter (Shared for CPU visibility during debugging)
    if (!m_currLevel)
        m_currLevel = [dev newBufferWithLength:count * sizeof(uint32_t)
                                    options:MTLResourceStorageModePrivate];
    if (!m_nextLevel)
        m_nextLevel = [dev newBufferWithLength:count * sizeof(uint32_t)
                                    options:MTLResourceStorageModePrivate];
    if (!m_levelCount)
        m_levelCount = [dev newBufferWithLength:sizeof(uint32_t)
                                    options:MTLResourceStorageModeShared];

    if (!m_excessFlag)
        m_excessFlag = [dev newBufferWithLength:sizeof(uint32_t)
                                   options:MTLResourceStorageModeShared];

    // Slice 1: allocate atomic buffers (NodeDataAtom 8 bytes, Residual4Atom 16 bytes)
    if (!m_nodeDataAtom)
        m_nodeDataAtom = [dev newBufferWithLength:count * 8
                                          options:MTLResourceStorageModePrivate];
    if (!m_residualAtom)
        m_residualAtom = [dev newBufferWithLength:count * 16
                                          options:MTLResourceStorageModePrivate];
}

void MetalGraphCut::buildGraph(const MetalMat& unary_bg,
                               const MetalMat& unary_fg,
                               const MetalMat& pairwise_left,
                               const MetalMat& pairwise_top,
                               const MetalMat& pairwise_topleft,
                               const MetalMat& pairwise_topright,
                               double /*lambda*/)
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

        [enc setBuffer:m_nodeData offset:0 atIndex:0];
        [enc setBuffer:m_terminalFlow offset:0 atIndex:1];
        [enc setBuffer:m_residualCap offset:0 atIndex:2];

        MTLSize tg = MTLSizeMake(16,16,1);
        MTLSize grid = MTLSizeMake((m_graphSize.width  + tg.width  - 1)/tg.width,
                                   (m_graphSize.height + tg.height - 1)/tg.height,1);
        [enc dispatchThreadgroups:grid threadsPerThreadgroup:tg];
        [enc endEncoding];
    }

    // --- New atomic graph build ---
    id<MTLComputePipelineState> atomPipe = getBuildGraphAtomPipeline();
    if(atomPipe){
        @autoreleasepool{
            id<MTLComputeCommandEncoder> enc = StreamAccessor::createComputeEncoder(m_stream);
            [enc setComputePipelineState:atomPipe];
            [enc setTexture:unary_bg.texture() atIndex:0];
            [enc setTexture:unary_fg.texture() atIndex:1];
            [enc setTexture:pairwise_left.texture()      atIndex:2];
            [enc setTexture:pairwise_topleft.texture()   atIndex:3];
            [enc setTexture:pairwise_top.texture()       atIndex:4];
            [enc setTexture:pairwise_topright.texture()  atIndex:5];

            [enc setBuffer:m_nodeDataAtom offset:0 atIndex:0];
            // Correct buffer order: buildGraphAtomKernel expects terminal flow at index 1 and residual caps at index 2
            [enc setBuffer:m_terminalFlow offset:0 atIndex:1];
            [enc setBuffer:m_residualAtom offset:0 atIndex:2];

            MTLSize tg=MTLSizeMake(16,16,1);
            MTLSize grid=MTLSizeMake((m_graphSize.width+15)/16,(m_graphSize.height+15)/16,1);
            [enc dispatchThreadgroups:grid threadsPerThreadgroup:tg];
            [enc endEncoding];
        }
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
            [enc setBuffer:m_currLevel offset:0 atIndex:2];
            uint total=(uint)(m_graphSize.width*m_graphSize.height);
            [enc setBytes:&total length:sizeof(uint) atIndex:3];

            MTLSize tg=MTLSizeMake(256,1,1);
            MTLSize grid=MTLSizeMake((total+tg.width-1)/tg.width,1,1);
            [enc dispatchThreadgroups:grid threadsPerThreadgroup:tg];
            [enc endEncoding];
        }
    }
}

void MetalGraphCut::solve(int /*iterations*/)
{
    // Placeholder – currently just runs BFS init to set labels for future stages.

    id<MTLComputePipelineState> initPipe = getBFSInitPipeline();
    if (!initPipe) return;

    NSUInteger nodeCount = m_graphSize.width * m_graphSize.height;

    // Reset level counter to 0
    uint32_t zero = 0;
    memcpy([m_levelCount contents], &zero, sizeof(uint32_t));

    @autoreleasepool {
        id<MTLComputeCommandEncoder> enc = StreamAccessor::createComputeEncoder(m_stream);
        [enc setComputePipelineState:initPipe];
        [enc setBuffer:m_nodeData offset:0 atIndex:0];
        [enc setBuffer:m_terminalFlow offset:0 atIndex:1];
        [enc setBuffer:m_currLevel offset:0 atIndex:2];
        [enc setBuffer:m_levelCount offset:0 atIndex:3];
        uint totalNodes = (uint)nodeCount;
        [enc setBytes:&totalNodes length:sizeof(uint) atIndex:4];

        MTLSize tg = MTLSizeMake(256,1,1);
        MTLSize grid = MTLSizeMake((nodeCount + tg.width - 1)/tg.width, 1, 1);
        [enc dispatchThreadgroups:grid threadsPerThreadgroup:tg];
        [enc endEncoding];
    }

    uint32_t levelCnt = *(uint32_t*)[m_levelCount contents];
    id<MTLComputePipelineState> travPipe = getBFSTraverseAtomPipeline();
    uint iter = 0;
    while (levelCnt > 0 && iter < 1000)
    {
        uint32_t zero32 = 0; memcpy([m_levelCount contents], &zero32, sizeof(uint32_t));

        @autoreleasepool {
            id<MTLComputeCommandEncoder> enc = StreamAccessor::createComputeEncoder(m_stream);
            [enc setComputePipelineState:travPipe];
            [enc setBuffer:m_nodeDataAtom offset:0 atIndex:0];
            [enc setBuffer:m_residualAtom offset:0 atIndex:1];
            [enc setBuffer:m_currLevel offset:0 atIndex:2];
            [enc setBytes:&levelCnt length:sizeof(uint32_t) atIndex:3];
            [enc setBuffer:m_levelCount offset:0 atIndex:4]; // nextCount atomic
            [enc setBuffer:m_nextLevel offset:0 atIndex:5];
            uint widthVal = (uint)m_graphSize.width;
            [enc setBytes:&widthVal length:sizeof(uint) atIndex:6];

            MTLSize tg = MTLSizeMake(256,1,1);
            MTLSize grid = MTLSizeMake((levelCnt + tg.width - 1)/tg.width, 1,1);
            [enc dispatchThreadgroups:grid threadsPerThreadgroup:tg];
            [enc endEncoding];
        }

        m_stream.syncCPU();
        levelCnt = *(uint32_t*)[m_levelCount contents];
        std::swap(m_currLevel, m_nextLevel);
        ++iter;
    }

    id<MTLComputePipelineState> pushPipe = getPushAtomPipeline();
    uint32_t one = 1; // Track if any excess remains
    int prIter = 0;
    const int kMaxPRIters = 1000; // Maximum push-relabel iterations
    while (one && prIter < kMaxPRIters)
    {
        // zero nextCount
        uint32_t zero32 = 0; memcpy([m_levelCount contents], &zero32, sizeof(uint32_t));
        zero32 = 0; memcpy([m_excessFlag contents], &zero32, sizeof(uint32_t));

        @autoreleasepool {
            id<MTLComputeCommandEncoder> enc = StreamAccessor::createComputeEncoder(m_stream);
            [enc setComputePipelineState:pushPipe];
            [enc setBuffer:m_nodeDataAtom offset:0 atIndex:0];
            [enc setBuffer:m_residualAtom offset:0 atIndex:1];
            [enc setBuffer:m_currLevel offset:0 atIndex:2];
            uint activeCountVal = levelCnt;
            [enc setBytes:&activeCountVal length:sizeof(uint) atIndex:3];
            [enc setBuffer:m_levelCount offset:0 atIndex:4]; // nextCount (atomic)
            [enc setBuffer:m_nextLevel offset:0 atIndex:5];
            [enc setBuffer:m_terminalFlow offset:0 atIndex:6];
            [enc setBuffer:m_excessFlag offset:0 atIndex:7];
            MTLSize tg = MTLSizeMake(256,1,1);
            MTLSize grid = MTLSizeMake((activeCountVal + 255)/256,1,1);
            [enc dispatchThreadgroups:grid threadsPerThreadgroup:tg];
            [enc endEncoding];
        }

        m_stream.syncCPU();
        one = *(uint32_t*)[m_excessFlag contents];
        levelCnt = *(uint32_t*)[m_levelCount contents];
        std::swap(m_currLevel, m_nextLevel);
        ++prIter;
    }

    id<MTLComputePipelineState> relabelPipe = getRelabelAtomPipeline();
    if(relabelPipe){
        @autoreleasepool{
            id<MTLComputeCommandEncoder> enc = StreamAccessor::createComputeEncoder(m_stream);
            [enc setComputePipelineState:relabelPipe];
            [enc setBuffer:m_nodeDataAtom offset:0 atIndex:0];
            [enc setBuffer:m_currLevel offset:0 atIndex:1]; // activeList
            uint activeCount = levelCnt;
            [enc setBytes:&activeCount length:sizeof(uint) atIndex:2];

            MTLSize tg=MTLSizeMake(256,1,1);
            MTLSize grid=MTLSizeMake((activeCount+tg.width-1)/tg.width,1,1);
            [enc dispatchThreadgroups:grid threadsPerThreadgroup:tg];
            [enc endEncoding];
        }
    }
}

void MetalGraphCut::getSegmentation(MetalMat& mask)
{
    if (mask.empty())
        mask.create(m_graphSize, CV_8UC1);

    // Temporary fallback: derive segmentation directly from unary terms while
    // the full push-relabel solver is still under development. Once the atomic
    // BFS / push-relabel pipeline is feature-complete we will switch back to
    // the labelSegmentationKernel.

    id<MTLComputePipelineState> pipe = getSimpleSegmentationPipeline();
    CV_Assert(pipe);

    @autoreleasepool {
        id<MTLComputeCommandEncoder> enc = StreamAccessor::createComputeEncoder(m_stream);
        [enc setComputePipelineState:pipe];
        [enc setTexture:m_bgTermTex atIndex:0];
        [enc setTexture:m_fgTermTex atIndex:1];
        [enc setTexture:mask.texture() atIndex:2];

        MTLSize tg = MTLSizeMake(16,16,1);
        MTLSize grid = MTLSizeMake((m_graphSize.width + tg.width -1)/tg.width,
                                   (m_graphSize.height+ tg.height-1)/tg.height,1);
        [enc dispatchThreadgroups:grid threadsPerThreadgroup:tg];
        [enc endEncoding];
    }
}

}} // namespace cv::metal

#endif // HAVE_METAL 