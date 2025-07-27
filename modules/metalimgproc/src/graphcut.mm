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

    m_stream.syncCPU();
    uint32_t activeCount = *(uint32_t*)[m_levelCount contents];
    printf("[MetalGraphCut DEBUG] Initial active nodes: %u\n", activeCount);
    id<MTLBuffer> activeListIn = m_activeList1;
    id<MTLBuffer> activeListOut = m_activeList2;

    int iter = 0;
    int globalRelabelFreq = std::max(1, (int)sqrt(totalNodes)); // Global relabel frequency

    if (totalNodes > 10000) {
        printf("[MetalGraphCut] Large graph: %u nodes, globalRelabelFreq=%d, maxIterations=%d\n", 
               totalNodes, globalRelabelFreq, maxIterations);
    }

    for (iter = 0; iter < maxIterations; ++iter) {
        if (activeCount == 0) {
            printf("[MetalGraphCut DEBUG] Converged after %d iterations.\n", iter);
            break;
        }

        // Reset output active counter using a non-blocking blit command
        @autoreleasepool {
            id<MTLBlitCommandEncoder> blitEnc = StreamAccessor::createBlitEncoder(m_stream);
            [blitEnc fillBuffer:m_levelCount range:NSMakeRange(0, sizeof(uint32_t)) value:0];
            [blitEnc endEncoding];
        }
        
        if (iter < 5 || (totalNodes > 10000 && iter % 100 == 0)) {
            printf("[MetalGraphCut DEBUG] Iteration %d: dispatching with activeCount=%u (stale)\n", iter, activeCount);
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

        // Heuristics serve as periodic sync points.
        bool isGlobalRelabelTime = (iter > 0 && (iter + 1) % globalRelabelFreq == 0);
        bool isGapRelabelTime = (iter > 0 && (iter + 1) % 10 == 0);

        if (isGlobalRelabelTime) {
            m_stream.syncCPU();
            activeCount = *(uint32_t*)[m_levelCount contents];
            if (activeCount > 0) {
                printf("Running global relabel at iteration %d (activeCount=%u)\n", iter + 1, activeCount);
                runGlobalRelabel();
                createActiveList();
                m_stream.syncCPU();
                activeCount = *(uint32_t*)[m_levelCount contents];
                activeListIn = m_activeList1;
                activeListOut = m_activeList2;
            }
        }
        else if (isGapRelabelTime) {
            m_stream.syncCPU();
            activeCount = *(uint32_t*)[m_levelCount contents];
            if (activeCount > 0) {
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
    }

    m_stream.syncCPU();
    activeCount = *(uint32_t*)[m_levelCount contents];
    printf("Push-relabel completed after %d iterations with %u active nodes remaining\n", iter, activeCount);

    // --- DEBUG: Check terminal flow after solve ---
    m_stream.syncCPU();

    const CpuTerminalFlow* termFlow = (const CpuTerminalFlow*)[m_terminalFlow contents];
    uint32_t source_connected_count = 0;
    float min_to_source = 1e9f, max_to_source = 0.0f, sum_to_source = 0.0f;

    for (NSUInteger i = 0; i < nodeCount; ++i) {
        uint32_t source_bits = termFlow[i].to_source_bits;
        float to_source_val;
        memcpy(&to_source_val, &source_bits, sizeof(float));

        if (to_source_val > 1e-6f) {
            source_connected_count++;
            if (to_source_val < min_to_source) min_to_source = to_source_val;
            if (to_source_val > max_to_source) max_to_source = to_source_val;
            sum_to_source += to_source_val;
        }
    }

    printf("[MetalGraphCut DEBUG] After solve: %u / %lu nodes have residual capacity from sink (to_source > 0).\n",
           source_connected_count, nodeCount);
    if (source_connected_count > 0) {
        printf("[MetalGraphCut DEBUG] to_source stats: min=%.4f, max=%.4f, avg=%.4f\n",
               min_to_source, max_to_source, sum_to_source / source_connected_count);
    }
    
    // --- DEBUG: Check residual graph ---
    const CpuResidualGraphAtom* resGraph = (const CpuResidualGraphAtom*)[m_residualAtom contents];
    uint32_t saturated_edges = 0;
    uint32_t total_edges = 0;
    float min_cap = 1e9f, max_cap = 0.0f;
    
    // Sample a few nodes and their edges
    uint32_t sample_nodes = std::min(10u, totalNodes);
    for (uint32_t i = 0; i < sample_nodes; ++i) {
        for (int dir = 0; dir < 8; ++dir) {
            uint32_t cap_bits = resGraph[i].c[dir];
            float cap;
            memcpy(&cap, &cap_bits, sizeof(float));
            
            total_edges++;
            if (cap < 1e-6f) {
                saturated_edges++;
            } else {
                if (cap < min_cap) min_cap = cap;
                if (cap > max_cap) max_cap = cap;
            }
        }
    }
    
    printf("[MetalGraphCut DEBUG] Residual graph sample: %u/%u edges saturated (near 0)\n",
           saturated_edges, total_edges);
    if (total_edges > saturated_edges) {
        printf("[MetalGraphCut DEBUG] Non-saturated edge capacities: min=%.6f, max=%.6f\n",
               min_cap, max_cap);
    }
    
    // Print capacities for node 0 as an example
    if (nodeCount > 0) {
        printf("[MetalGraphCut DEBUG] Node 0 residual capacities (W,NW,N,NE,E,SE,S,SW): ");
        for (int dir = 0; dir < 8; ++dir) {
            uint32_t cap_bits = resGraph[0].c[dir];
            float cap;
            memcpy(&cap, &cap_bits, sizeof(float));
            printf("%.4f ", cap);
        }
        printf("\n");
    }
    
    // --- DEBUG: Check for nodes with excess flow ---
    const CpuNodeDataAtom* nodeData = (const CpuNodeDataAtom*)[m_nodeDataAtom contents];
    uint32_t nodes_with_excess = 0;
    uint32_t stuck_nodes = 0;
    float total_excess = 0.0f;
    
    for (NSUInteger i = 0; i < nodeCount; ++i) {
        uint32_t excess_bits = nodeData[i].excessBits;
        float excess;
        memcpy(&excess, &excess_bits, sizeof(float));
        
        if (excess > 1e-6f) {
            nodes_with_excess++;
            total_excess += excess;
            int height = nodeData[i].label;
            
            // Check if this node could have been relabeled
            int minHeight = INT_MAX;
            
            // Check neighbors
            uint32_t x = i % width;
            uint32_t y = i / width;
            
            // First check if this is the stuck node and print its details
            if (nodes_with_excess <= 5) {
                printf("[MetalGraphCut DEBUG] Node %lu with excess %.6f at height %d, pos (%u,%u)\n",
                       i, excess, height, x, y);
            }
            for (int dir = 0; dir < 8; ++dir) {
                int nx = x;
                int ny = y;
                // Directions: 0:W, 1:NW, 2:N, 3:NE, 4:E, 5:SE, 6:S, 7:SW
                switch(dir) {
                    case 0: nx -= 1; break;           // W
                    case 1: nx -= 1; ny -= 1; break;  // NW
                    case 2: ny -= 1; break;           // N
                    case 3: nx += 1; ny -= 1; break;  // NE
                    case 4: nx += 1; break;           // E
                    case 5: nx += 1; ny += 1; break;  // SE
                    case 6: ny += 1; break;           // S
                    case 7: nx -= 1; ny += 1; break;  // SW
                }
                
                if (nx >= 0 && nx < (int)width && ny >= 0 && ny < (int)height) {
                    uint32_t nidx = ny * width + nx;
                    uint32_t cap_bits = resGraph[i].c[dir];
                    float cap;
                    memcpy(&cap, &cap_bits, sizeof(float));
                    
                    if (cap > 1e-6f) {
                        int neighHeight = nodeData[nidx].label;
                        if (neighHeight < minHeight) {
                            minHeight = neighHeight;
                        }
                    }
                }
            }
            
            // Check sink
            uint32_t sink_cap_bits = termFlow[i].to_sink_bits;
            float sink_cap;
            memcpy(&sink_cap, &sink_cap_bits, sizeof(float));
            if (sink_cap > 1e-6f) {
                minHeight = std::min(minHeight, 0); // Sink has height 0
            }
            
            // Check source (capToSourceBuf is defined later, need to check it separately)
            
            if (minHeight == INT_MAX) {
                stuck_nodes++;
                if (stuck_nodes <= 5) {
                    printf("[MetalGraphCut DEBUG] Stuck node %lu: excess=%.6f, height=%d, no valid neighbors\n",
                           i, excess, height);
                    // Check source capacity for this stuck node
                    const float* capToSourcePtr = (const float*)[m_capToSourceBuf contents];
                    float source_cap = capToSourcePtr[i];
                    printf("  - Residual capacity to source: %.6f\n", source_cap);
                }
            } else if (nodes_with_excess <= 5) {
                printf("  - Min neighbor height found: %d (would relabel to %d)\n",
                       minHeight, minHeight + 1);
                // Check source capacity
                const float* capToSourcePtr = (const float*)[m_capToSourceBuf contents];
                float source_cap = capToSourcePtr[i];
                printf("  - Residual capacity to source: %.6f (source height=%u)\n",
                       source_cap, totalNodes);
                if (source_cap > 1e-6f && minHeight >= (int)totalNodes) {
                    printf("  - Node CAN push to source after relabel to height %d\n", minHeight + 1);
                }
            }
        }
    }
    
    printf("[MetalGraphCut DEBUG] Nodes with excess flow: %u / %u (total excess=%.6f)\n",
           nodes_with_excess, totalNodes, total_excess);
    if (nodes_with_excess > 0) {
        printf("[MetalGraphCut DEBUG] Stuck nodes (no valid relabel): %u / %u\n",
               stuck_nodes, nodes_with_excess);
    }
    
    // --- DEBUG: Check capToSourceBuf ---
    const float* capToSource = (const float*)[m_capToSourceBuf contents];
    uint32_t nodes_with_source_cap = 0;
    float total_source_cap = 0.0f;
    float min_source_cap = 1e9f, max_source_cap = 0.0f;
    
    for (NSUInteger i = 0; i < nodeCount; ++i) {
        float cap = capToSource[i];
        if (cap > 1e-6f) {
            nodes_with_source_cap++;
            total_source_cap += cap;
            if (cap < min_source_cap) min_source_cap = cap;
            if (cap > max_source_cap) max_source_cap = cap;
        }
    }
    
    printf("[MetalGraphCut DEBUG] Nodes with residual capacity to source: %u / %u\n",
           nodes_with_source_cap, totalNodes);
    if (nodes_with_source_cap > 0) {
        printf("[MetalGraphCut DEBUG] Source capacity stats: min=%.6f, max=%.6f, total=%.6f\n",
               min_source_cap, max_source_cap, total_source_cap);
    }
    // --- END DEBUG ---
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
        [enc setBuffer:m_terminalFlow offset:0 atIndex:1];   // Pass the terminal flow buffer
        [enc setBuffer:m_finalCutLabels offset:0 atIndex:2]; // Shifted from 1
        [enc setBuffer:m_activeList1 offset:0 atIndex:3];    // Shifted from 2 - BFS queue
        [enc setBuffer:m_levelCount offset:0 atIndex:4];     // Shifted from 3
        [enc setBytes:&totalNodes length:sizeof(uint32_t) atIndex:5]; // Shifted from 4

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
    printf("[GraphCut Final BFS] Initial sink-connected nodes: %u\n", queueCount);
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
    printf("[MetalGraphCut DEBUG] After BFS: %u / %u nodes are reachable from sink.\n",
           reachable_count, totalNodes);
    if (reachable_count > 0 && reachable_count < 20) {
        printf("[MetalGraphCut DEBUG] Reachable node indices: ");
        int printed_count = 0;
        for (NSUInteger i = 0; i < totalNodes && printed_count < 20; ++i) {
            if (reachableLabels[i] != 0) {
                printf("%lu ", i);
                printed_count++;
            }
        }
        printf("\n");
    }
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