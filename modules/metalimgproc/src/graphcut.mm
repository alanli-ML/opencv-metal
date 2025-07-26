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

    if (!m_terminalFlow)
        m_terminalFlow = [dev newBufferWithLength:count * sizeof(float) * 2
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