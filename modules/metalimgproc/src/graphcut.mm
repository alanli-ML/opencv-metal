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

    @autoreleasepool {
        id<MTLBlitCommandEncoder> blitEnc = StreamAccessor::createBlitEncoder(m_stream);
        [blitEnc fillBuffer:m_levelCount range:NSMakeRange(0, sizeof(uint32_t)) value:0];
        [blitEnc endEncoding];
    }

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
        @autoreleasepool {
            id<MTLBlitCommandEncoder> blitEnc = StreamAccessor::createBlitEncoder(m_stream);
            [blitEnc fillBuffer:m_levelCount range:NSMakeRange(0, sizeof(uint32_t)) value:0];
            [blitEnc endEncoding];
        }
        
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

    @autoreleasepool {
        id<MTLBlitCommandEncoder> blitEnc = StreamAccessor::createBlitEncoder(m_stream);
        [blitEnc fillBuffer:m_levelCount range:NSMakeRange(0, sizeof(uint32_t)) value:0];
        [blitEnc endEncoding];
    }

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

    // Phase 1: Create initial active list from nodes with excess flow
    createActiveList();

    // Phase 2: Main push-relabel loop
    id<MTLComputePipelineState> pushRelabelPipe = getPushRelabelPipeline();
    if (!pushRelabelPipe) {
        CV_Error(cv::Error::StsError, "Failed to get push-relabel pipeline");
        return;
    }
    id<MTLComputePipelineState> histogramPipe = getBuildHeightHistogramPipeline();
    id<MTLComputePipelineState> gapRelabelPipe = getGapRelabelPipeline();

    m_stream.syncCPU();
    uint32_t activeCount = *(uint32_t*)[m_levelCount contents];
    
    id<MTLBuffer> activeListIn = m_activeList1;
    id<MTLBuffer> activeListOut = m_activeList2;

    int iter = 0;
    
    // Phase 1.1: Convert to while loop for true convergence
    while (activeCount > 0 && iter < maxIterations) {

        // Reset output active counter using a non-blocking blit command
        @autoreleasepool {
            id<MTLBlitCommandEncoder> blitEnc = StreamAccessor::createBlitEncoder(m_stream);
            [blitEnc fillBuffer:m_levelCount range:NSMakeRange(0, sizeof(uint32_t)) value:0];
            [blitEnc endEncoding];
        }
        

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
            [enc setBuffer:m_capToSourceBuf offset:0 atIndex:9]; // Residual capacity to source
            [enc setBytes:&totalNodes length:sizeof(uint32_t) atIndex:10]; // Total nodes for height calculations

            MTLSize tg = MTLSizeMake(256, 1, 1);
            MTLSize grid = MTLSizeMake((activeCount + tg.width - 1) / tg.width, 1, 1);
            [enc dispatchThreadgroups:grid threadsPerThreadgroup:tg];
            [enc endEncoding];
        }

        std::swap(activeListIn, activeListOut);
        
        // Synchronize to read the new active count
        m_stream.syncCPU();
        activeCount = *(uint32_t*)[m_levelCount contents];
        
        iter++;
        
        // Heuristics serve as periodic sync points.
        int globalRelabelFreq = std::max(1, (int)sqrt(totalNodes));
        bool isGlobalRelabelTime = (iter > 0 && iter % globalRelabelFreq == 0);
        bool isGapRelabelTime = (iter > 0 && iter % 10 == 0);

        if (isGlobalRelabelTime && activeCount > 0) {
            runGlobalRelabel();
            createActiveList();
            m_stream.syncCPU();
            activeCount = *(uint32_t*)[m_levelCount contents];
            activeListIn = m_activeList1;
            activeListOut = m_activeList2;
        }
        else if (isGapRelabelTime && activeCount > 0) {

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

    // Log convergence status
    if (activeCount == 0) {
        printf("[MetalGraphCut] Converged after %d iterations\n", iter);
    } else {
        printf("[MetalGraphCut] Reached maximum iterations (%d) with %u active nodes remaining\n", iter, activeCount);
    }
}


void MetalGraphCut::getSegmentation(MetalMat& mask, const MetalMat& initial_mask)
{
    if (mask.empty())
        mask.create(m_graphSize, CV_8UC1);

    uint32_t width = (uint32_t)m_graphSize.width;
    uint32_t height = (uint32_t)m_graphSize.height;
    uint32_t totalNodes = width * height;

    // 1. Initialize BFS: Find nodes at source height and add to initial queue
    id<MTLComputePipelineState> bfsInitPipe = getFinalCutBfsInit_SourceSet_Pipeline(MetalContext::getInstance().device);
    CV_Assert(bfsInitPipe);

    uint32_t zero = 0;
    memcpy([m_levelCount contents], &zero, sizeof(uint32_t));

    @autoreleasepool {
        id<MTLComputeCommandEncoder> enc = StreamAccessor::createComputeEncoder(m_stream);
        [enc setComputePipelineState:bfsInitPipe];
        
        // Source-set BFS init arguments (simplified)
        // Buffers
        [enc setBuffer:m_nodeDataAtom offset:0 atIndex:0];
        [enc setBuffer:m_finalCutLabels offset:0 atIndex:1];
        [enc setBuffer:m_activeList1 offset:0 atIndex:2];    // BFS queue
        [enc setBuffer:m_levelCount offset:0 atIndex:3];
        
        // Constants
        [enc setBytes:&totalNodes length:sizeof(uint32_t) atIndex:4];

        // Dispatch 1D over all nodes
        MTLSize tg = MTLSizeMake(256, 1, 1);
        MTLSize grid = MTLSizeMake((totalNodes + tg.width - 1) / tg.width, 1, 1);
        [enc dispatchThreadgroups:grid threadsPerThreadgroup:tg];
        [enc endEncoding];
    }

    // 2. Traverse graph with BFS to find all source-reachable nodes
    id<MTLComputePipelineState> bfsTraversePipe = getFinalCutBfsTraverse_SourceSet_Pipeline(MetalContext::getInstance().device);
    CV_Assert(bfsTraversePipe);

    m_stream.syncCPU();
    uint32_t queueCount = *(uint32_t*)[m_levelCount contents];
    printf("[GraphCut Final BFS] Initial source-connected nodes: %u\n", queueCount);
    
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