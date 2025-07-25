#include "precomp.hpp"

#ifdef HAVE_METAL

#include "graphcut_internal.hpp"
#include <stdexcept>
#include <simd/simd.h>
#include <cstring>
#include <algorithm>
#include "opencv2/core/hal/interface.h" // for CV_8UC1 definition

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

struct TerminalFlow {
    float to_source;
    float to_sink;
};

kernel void buildGraphKernel(texture2d<float, access::read> bgTerm [[texture(0)]],
                             texture2d<float, access::read> fgTerm [[texture(1)]],
                             texture2d<float4, access::read> wLeft [[texture(2)]],
                             texture2d<float4, access::read> wTL [[texture(3)]],
                             texture2d<float4, access::read> wTop [[texture(4)]],
                             texture2d<float4, access::read> wTR [[texture(5)]],
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

// Kernel to initialize node labels and set excess based on terminal capacities
static const char* kGraphCutBFSInitSrc = R"(
#include <metal_stdlib>
using namespace metal;

struct NodeData { float excess; int label; };
struct TerminalFlow { float to_source; float to_sink; };

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
        id<MTLFunction> fn = ctx.getMetalFunction(kGraphCutBuildSrc, "buildGraphKernel");
        if (!fn) return nil;
        NSError* err = nil;
        s_pipeline = [ctx.device newComputePipelineStateWithFunction:fn error:&err];
        if (!s_pipeline || err)
            CV_Error(cv::Error::StsError, "Failed to create buildGraphKernel pipeline");
    }
    return s_pipeline;
}

id<MTLComputePipelineState> MetalGraphCut::getBFSInitPipeline()
{
    static id<MTLComputePipelineState> s_pipeline = nil;
    if (!s_pipeline)
    {
        MetalContext& ctx = MetalContext::getInstance();
        id<MTLFunction> fn = ctx.getMetalFunction(kGraphCutBFSInitSrc, "bfsInitKernel");
        if (!fn) return nil;
        NSError* err = nil;
        s_pipeline = [ctx.device newComputePipelineStateWithFunction:fn error:&err];
        if (!s_pipeline || err)
            CV_Error(cv::Error::StsError, "Failed to create bfsInitKernel pipeline");
    }
    return s_pipeline;
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
    id<MTLComputePipelineState> travPipe = getBFSTraversePipeline();
    uint iter = 0;
    while (levelCnt > 0 && iter < 1000) // safety cap
    {
        // Reset nextLevel counter
        uint32_t zero32 = 0;
        memcpy([m_levelCount contents], &zero32, sizeof(uint32_t));

        @autoreleasepool {
            id<MTLComputeCommandEncoder> enc = StreamAccessor::createComputeEncoder(m_stream);
            [enc setComputePipelineState:travPipe];
            [enc setBuffer:m_nodeData offset:0 atIndex:0];
            [enc setBuffer:m_levelCount offset:0 atIndex:1]; // atomic nextCount
            [enc setBuffer:m_currLevel offset:0 atIndex:2];
            [enc setBytes:&levelCnt length:sizeof(uint32_t) atIndex:3];
            [enc setBuffer:m_nextLevel offset:0 atIndex:4];
            [enc setBuffer:m_residualCap offset:0 atIndex:5];
            uint widthVal = (uint)m_graphSize.width;
            [enc setBytes:&widthVal length:sizeof(uint) atIndex:6];

            MTLSize tg = MTLSizeMake(256,1,1);
            MTLSize grid = MTLSizeMake((levelCnt + tg.width - 1)/tg.width, 1,1);
            [enc dispatchThreadgroups:grid threadsPerThreadgroup:tg];
            [enc endEncoding];
        }

        m_stream.syncCPU();
        levelCnt = *(uint32_t*)[m_levelCount contents];

        // swap buffers
        std::swap(m_currLevel, m_nextLevel);
        ++iter;
    }

    // ----- Placeholder push-relabel single pass -----

    // Reset excessFlag to 1 to enter loop
    uint32_t one = 1; memcpy([m_excessFlag contents], &one, sizeof(uint32_t));
    uint prIter = 0;
    id<MTLComputePipelineState> pushPipe = getPushPipeline();
    id<MTLComputePipelineState> chkPipe  = getCheckExcessPipeline();

    while (one && prIter < 1) // currently single iteration until pushKernel implemented
    {
        // push
        @autoreleasepool {
            id<MTLComputeCommandEncoder> enc = StreamAccessor::createComputeEncoder(m_stream);
            [enc setComputePipelineState:pushPipe];
            [enc setBuffer:m_nodeData offset:0 atIndex:0];
            [enc setBuffer:m_excessFlag offset:0 atIndex:1];
            [enc setBytes:&(m_graphSize.width * m_graphSize.height) length:sizeof(uint) atIndex:2];
            MTLSize grid=MTLSizeMake(1,1,1); MTLSize tg=MTLSizeMake(1,1,1);
            [enc dispatchThreadgroups:grid threadsPerThreadgroup:tg];
            [enc endEncoding];
        }

        // check
        @autoreleasepool {
            id<MTLComputeCommandEncoder> enc = StreamAccessor::createComputeEncoder(m_stream);
            [enc setComputePipelineState:chkPipe];
            [enc setBuffer:m_excessFlag offset:0 atIndex:0];
            [enc setBuffer:m_excessFlag offset:0 atIndex:1];
            MTLSize grid=MTLSizeMake(1,1,1); MTLSize tg=MTLSizeMake(1,1,1);
            [enc dispatchThreadgroups:grid threadsPerThreadgroup:tg];
            [enc endEncoding];
        }

        m_stream.syncCPU();
        one = *(uint32_t*)[m_excessFlag contents];
        ++prIter;
    }

}

void MetalGraphCut::getSegmentation(MetalMat& mask)
{
    if (mask.empty())
        mask.create(m_graphSize, CV_8UC1);

    id<MTLComputePipelineState> pipe = getLabelSegmentationPipeline();
    CV_Assert(pipe);

    uint widthVal = (uint)m_graphSize.width;

    @autoreleasepool {
        id<MTLComputeCommandEncoder> enc = StreamAccessor::createComputeEncoder(m_stream);
        [enc setComputePipelineState:pipe];
        [enc setBuffer:m_nodeData offset:0 atIndex:0];
        [enc setTexture:mask.texture() atIndex:0];
        [enc setBytes:&widthVal length:sizeof(uint) atIndex:1];

        MTLSize tg = MTLSizeMake(16,16,1);
        MTLSize grid = MTLSizeMake((m_graphSize.width + tg.width -1)/tg.width,
                                   (m_graphSize.height+ tg.height-1)/tg.height,1);
        [enc dispatchThreadgroups:grid threadsPerThreadgroup:tg];
        [enc endEncoding];
    }
}

}} // namespace cv::metal

#endif // HAVE_METAL 