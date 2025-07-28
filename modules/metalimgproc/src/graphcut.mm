#include "precomp.hpp"

#ifdef HAVE_METAL

#include "graphcut_internal.hpp"
#include <stdexcept>
#include <cstring>
#include <algorithm>
#include "opencv2/core/hal/interface.h" // for CV_8UC1 definition
#include "graphcut_kernels.hpp"

using namespace cv::metal;

using namespace cv::metal;


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
            CV_Error(cv::Error::StsError,"Failed to create pushRelabelKernel pipeline");
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

id<MTLComputePipelineState> MetalGraphCut::getFindGapPipeline()
{
    static id<MTLComputePipelineState> s_pipeline = nil;
    if (!s_pipeline)
    {
        MetalContext& ctx = MetalContext::getInstance();
        std::string src = std::string(kGraphCutCommonSrc) + kGraphCutFindGapSrc;
        id<MTLFunction> fn = ctx.getMetalFunction(src, "findGapKernel");
        if (!fn) return nil;
        NSError* err = nil;
        s_pipeline = [ctx.device newComputePipelineStateWithFunction:fn error:&err];
        if (!s_pipeline || err)
            CV_Error(cv::Error::StsError, "Failed to create findGapKernel pipeline");
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

id<MTLComputePipelineState> MetalGraphCut::getFinalCutBfsInit_SourceSet_Pipeline(id<MTLDevice> device)
{
    static id<MTLComputePipelineState> s_pipeline = nil;
    if (!s_pipeline)
    {
        MetalContext& ctx = MetalContext::getInstance();
        std::string src = std::string(kGraphCutCommonSrc) + kGraphCutFinalCutBfsInit_SourceSet_Src;
        id<MTLFunction> fn = ctx.getMetalFunction(src, "finalCutBfsInit_SourceSet_Kernel");
        if (!fn) return nil;
        NSError* err = nil;
        s_pipeline = [ctx.device newComputePipelineStateWithFunction:fn error:&err];
        if (!s_pipeline || err)
            CV_Error(cv::Error::StsError, "Failed to create finalCutBfsInit_SourceSet_Kernel pipeline");
    }
    return s_pipeline;
}

id<MTLComputePipelineState> MetalGraphCut::getFinalCutBfsTraverse_SourceSet_Pipeline(id<MTLDevice> device)
{
    static id<MTLComputePipelineState> s_pipeline = nil;
    if (!s_pipeline)
    {
        MetalContext& ctx = MetalContext::getInstance();
        std::string src = std::string(kGraphCutCommonSrc) + kGraphCutFinalCutBfsTraverse_SourceSet_Src;
        id<MTLFunction> fn = ctx.getMetalFunction(src, "finalCutBfsTraverse_SourceSet_Kernel");
        if (!fn) return nil;
        NSError* err = nil;
        s_pipeline = [ctx.device newComputePipelineStateWithFunction:fn error:&err];
        if (!s_pipeline || err)
            CV_Error(cv::Error::StsError, "Failed to create finalCutBfsTraverse_SourceSet_Kernel pipeline");
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


namespace cv { namespace metal {

void MetalGraphCut::runGlobalRelabel()
{
    NSUInteger nodeCount = m_graphSize.width * m_graphSize.height;
    uint32_t totalNodes = (uint32_t)nodeCount;
    uint32_t width = (uint32_t)m_graphSize.width;
    uint32_t height = (uint32_t)m_graphSize.height;

    id<MTLComputePipelineState> globalRelabelInitPipe = getGlobalRelabelInitPipeline();
    CV_Assert(globalRelabelInitPipe);

    // This kernel now just sets labels, no queue output
    @autoreleasepool {
        id<MTLComputeCommandEncoder> enc = StreamAccessor::createComputeEncoder(m_stream);
        [enc setComputePipelineState:globalRelabelInitPipe];
        [enc setBuffer:m_nodeDataAtom offset:0 atIndex:0];
        [enc setBuffer:m_terminalFlow offset:0 atIndex:1];
        [enc setBytes:&totalNodes length:sizeof(uint32_t) atIndex:2];
        MTLSize tg = MTLSizeMake(256, 1, 1);
        MTLSize grid = MTLSizeMake((totalNodes + tg.width - 1) / tg.width, 1, 1);
        [enc dispatchThreadgroups:grid threadsPerThreadgroup:tg];
        [enc endEncoding];
    }
    
    id<MTLComputePipelineState> globalRelabelBfsPipe = getGlobalRelabelBfsTraversePipeline();
    CV_Assert(globalRelabelBfsPipe);

    // Asynchronous BFS loop. No syncs, no queue management.
    uint32_t maxBfsIters = width + height;
    for (uint32_t bfsIter = 1; bfsIter < maxBfsIters; ++bfsIter) {
        @autoreleasepool {
            id<MTLComputeCommandEncoder> enc = StreamAccessor::createComputeEncoder(m_stream);
            [enc setComputePipelineState:globalRelabelBfsPipe];
            [enc setBuffer:m_nodeDataAtom offset:0 atIndex:0];
            [enc setBuffer:m_residualAtom offset:0 atIndex:1];
            [enc setBytes:&bfsIter length:sizeof(uint32_t) atIndex:2]; // current_bfs_level
            [enc setBytes:&width length:sizeof(uint32_t) atIndex:3];
            [enc setBytes:&height length:sizeof(uint32_t) atIndex:4];
            [enc setBytes:&totalNodes length:sizeof(uint32_t) atIndex:5];
            
            // Dispatch over the entire grid
            MTLSize tg = MTLSizeMake(256, 1, 1);
            MTLSize grid = MTLSizeMake((totalNodes + tg.width - 1) / tg.width, 1, 1);
            [enc dispatchThreadgroups:grid threadsPerThreadgroup:tg];
            [enc endEncoding];
        }
    }
}

struct CpuTerminalFlow {
    uint32_t to_source_bits;
    uint32_t to_sink_bits;
};

struct CpuResidualGraphAtom {
    uint32_t c[8];  // Capacities for 8 directions as float bits
};

struct CpuNodeDataAtom {
    uint32_t excessBits;  // float bits
    int32_t label;
};

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

    if (!m_terminalFlow)
        m_terminalFlow = [dev newBufferWithLength:count * sizeof(CpuTerminalFlow)
                                          options:MTLResourceStorageModeShared];

    if (!m_excessFlag)
        m_excessFlag = [dev newBufferWithLength:sizeof(uint32_t)
                                   options:MTLResourceStorageModeShared];
    
    // Height histogram for gap optimization (shared memory for CPU access)
    NSUInteger max_height = count; // worst case: all nodes at different heights
    if (!m_heightHistogram)
        m_heightHistogram = [dev newBufferWithLength:max_height * sizeof(uint32_t)
                                            options:MTLResourceStorageModeShared];
    
    if (!m_gapInfo)
        m_gapInfo = [dev newBufferWithLength:sizeof(uint32_t)
                                    options:MTLResourceStorageModeShared];

    // Final cut reachability labels
    if (!m_finalCutLabels)
        m_finalCutLabels = [dev newBufferWithLength:count * sizeof(int32_t)
                                            options:MTLResourceStorageModeShared];

    // Slice 1: allocate atomic buffers (NodeDataAtom 8 bytes, ResidualGraphAtom 32 bytes)
    if (!m_nodeDataAtom)
        m_nodeDataAtom = [dev newBufferWithLength:count * 8
                                          options:MTLResourceStorageModeShared];
    if (!m_residualAtom)
        m_residualAtom = [dev newBufferWithLength:count * 32
                                          options:MTLResourceStorageModeShared];
    if (!m_capToSourceBuf)
        m_capToSourceBuf = [dev newBufferWithLength:count * sizeof(float)
                                           options:MTLResourceStorageModeShared];
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

    // --- New atomic graph build ---
    id<MTLComputePipelineState> atomPipe = getBuildGraphAtomPipeline();
    CV_Assert(atomPipe);

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
        [enc setBuffer:m_capToSourceBuf offset:0 atIndex:4];

        MTLSize tg=MTLSizeMake(16,16,1);
        MTLSize grid=MTLSizeMake((m_graphSize.width+15)/16,(m_graphSize.height+15)/16,1);
        [enc dispatchThreadgroups:grid threadsPerThreadgroup:tg];
        [enc endEncoding];
    }
}

void MetalGraphCut::solve(int maxIterations)
{
    NSUInteger nodeCount = m_graphSize.width * m_graphSize.height;

    uint32_t totalNodes = (uint32_t)nodeCount;
    uint32_t width = (uint32_t)m_graphSize.width;
    uint32_t height = (uint32_t)m_graphSize.height;

    // Get necessary pipelines
    id<MTLComputePipelineState> pushRelabelPipe = getPushRelabelPipeline();
    CV_Assert(pushRelabelPipe);
    id<MTLComputePipelineState> histogramPipe = getBuildHeightHistogramPipeline();
    id<MTLComputePipelineState> findGapPipe = getFindGapPipeline();
    id<MTLComputePipelineState> gapRelabelPipe = getGapRelabelPipeline();

    for (int iter = 0; iter < maxIterations; ++iter) {
        // Heuristics are the only potential sync points.
        int globalRelabelFreq = std::max(1, (int)sqrt(totalNodes));
        bool isGlobalRelabelTime = (iter > 0 && iter % globalRelabelFreq == 0);
        bool isGapRelabelTime = (iter > 0 && iter % 10 == 0);

        if (isGlobalRelabelTime) {
            runGlobalRelabel(); // This is fully asynchronous
        }
        else if (isGapRelabelTime) {
            if (histogramPipe && findGapPipe && gapRelabelPipe) {
                // 1. Clear histogram and gapInfo buffer
                @autoreleasepool {
                    id<MTLBlitCommandEncoder> blitEnc = StreamAccessor::createBlitEncoder(m_stream);
                    [blitEnc fillBuffer:m_heightHistogram range:NSMakeRange(0, totalNodes * sizeof(uint32_t)) value:0];
                    [blitEnc fillBuffer:m_gapInfo range:NSMakeRange(0, sizeof(uint32_t)) value:0];
                    [blitEnc endEncoding];
                }

                // 2. Build height histogram
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

                // 3. Find gap in histogram (asynchronously)
                @autoreleasepool {
                    id<MTLComputeCommandEncoder> enc = StreamAccessor::createComputeEncoder(m_stream);
                    [enc setComputePipelineState:findGapPipe];
                    [enc setBuffer:m_heightHistogram offset:0 atIndex:0];
                    [enc setBuffer:m_gapInfo offset:0 atIndex:1];
                    [enc setBytes:&totalNodes length:sizeof(uint32_t) atIndex:2];
                    [enc dispatchThreadgroups:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
                    [enc endEncoding];
                }

                // 4. Apply gap relabel using the result from the findGapKernel
                @autoreleasepool {
                    id<MTLComputeCommandEncoder> enc = StreamAccessor::createComputeEncoder(m_stream);
                    [enc setComputePipelineState:gapRelabelPipe];
                    [enc setBuffer:m_nodeDataAtom offset:0 atIndex:0];
                    [enc setBuffer:m_gapInfo offset:0 atIndex:1];
                    [enc setBytes:&totalNodes length:sizeof(uint32_t) atIndex:2];

                    MTLSize tg = MTLSizeMake(256, 1, 1);
                    MTLSize grid = MTLSizeMake((totalNodes + tg.width - 1) / tg.width, 1, 1);
                    [enc dispatchThreadgroups:grid threadsPerThreadgroup:tg];
                    [enc endEncoding];
                }
            }
        }

        // Dispatch main push-relabel kernel over the entire grid
        @autoreleasepool {
            id<MTLComputeCommandEncoder> enc = StreamAccessor::createComputeEncoder(m_stream);
            [enc setComputePipelineState:pushRelabelPipe];
            [enc setBuffer:m_nodeDataAtom offset:0 atIndex:0];
            [enc setBuffer:m_residualAtom offset:0 atIndex:1];
            [enc setBuffer:m_terminalFlow offset:0 atIndex:2];
            [enc setBytes:&width length:sizeof(uint32_t) atIndex:3];
            [enc setBytes:&height length:sizeof(uint32_t) atIndex:4];
            [enc setBuffer:m_capToSourceBuf offset:0 atIndex:5];
            [enc setBytes:&totalNodes length:sizeof(uint32_t) atIndex:6];

            MTLSize tg = MTLSizeMake(256, 1, 1);
            MTLSize grid = MTLSizeMake((totalNodes + tg.width - 1) / tg.width, 1, 1);
            [enc dispatchThreadgroups:grid threadsPerThreadgroup:tg];
            [enc endEncoding];
        }
    }

    // Final sync before exiting
    m_stream.syncCPU();
    printf("[MetalGraphCut] Completed %d iterations\n", maxIterations);
}


void MetalGraphCut::getSegmentation(MetalMat& mask, const MetalMat& initial_mask)
{
    if (mask.empty())
        mask.create(m_graphSize, CV_8UC1);

    uint32_t width = (uint32_t)m_graphSize.width;
    uint32_t height = (uint32_t)m_graphSize.height;
    uint32_t totalNodes = width * height;

    // 1. Initialize BFS: Find nodes at source height and mark their labels.
    id<MTLComputePipelineState> bfsInitPipe = getFinalCutBfsInit_SourceSet_Pipeline(MetalContext::getInstance().device);
    CV_Assert(bfsInitPipe);

    @autoreleasepool {
        id<MTLComputeCommandEncoder> enc = StreamAccessor::createComputeEncoder(m_stream);
        [enc setComputePipelineState:bfsInitPipe];
        
        [enc setBuffer:m_nodeDataAtom offset:0 atIndex:0];
        [enc setBuffer:m_finalCutLabels offset:0 atIndex:1];
        [enc setBytes:&totalNodes length:sizeof(uint32_t) atIndex:2];

        // Dispatch 1D over all nodes
        MTLSize tg = MTLSizeMake(256, 1, 1);
        MTLSize grid = MTLSizeMake((totalNodes + tg.width - 1) / tg.width, 1, 1);
        [enc dispatchThreadgroups:grid threadsPerThreadgroup:tg];
        [enc endEncoding];
    }

    // 2. Traverse graph with asynchronous BFS to find all source-reachable nodes
    id<MTLComputePipelineState> bfsTraversePipe = getFinalCutBfsTraverse_SourceSet_Pipeline(MetalContext::getInstance().device);
    CV_Assert(bfsTraversePipe);

    // Asynchronous BFS loop
    uint32_t maxBfsIters = width + height;
    for (uint32_t bfsIter = 1; bfsIter < maxBfsIters; ++bfsIter) {
        @autoreleasepool {
            id<MTLComputeCommandEncoder> enc = StreamAccessor::createComputeEncoder(m_stream);
            [enc setComputePipelineState:bfsTraversePipe];
            [enc setBuffer:m_residualAtom offset:0 atIndex:0];
            [enc setBuffer:m_finalCutLabels offset:0 atIndex:1];
            [enc setBytes:&bfsIter length:sizeof(uint32_t) atIndex:2]; // current_bfs_level
            [enc setBytes:&width length:sizeof(uint32_t) atIndex:3];
            [enc setBytes:&height length:sizeof(uint32_t) atIndex:4];
            [enc setBytes:&totalNodes length:sizeof(uint32_t) atIndex:5];

            MTLSize tg = MTLSizeMake(256, 1, 1);
            MTLSize grid = MTLSizeMake((totalNodes + tg.width - 1) / tg.width, 1, 1);
            [enc dispatchThreadgroups:grid threadsPerThreadgroup:tg];
            [enc endEncoding];
        }
    }

    // --- DEBUG: Check reachable labels after BFS ---
    m_stream.syncCPU();
    const int32_t* reachableLabels = (const int32_t*)[m_finalCutLabels contents];
    uint32_t reachable_count = 0;
    for (NSUInteger i = 0; i < totalNodes; ++i) {
        if (reachableLabels[i] != 0) {
            reachable_count++;
        }
    }
    printf("[MetalGraphCut DEBUG] After BFS: %u / %u nodes are reachable from source.\n",
           reachable_count, totalNodes);
    // --- END DEBUG ---

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