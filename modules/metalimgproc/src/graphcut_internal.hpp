#ifndef OPENCV_METALIMGPROC_GRAPHCUT_INTERNAL_HPP
#define OPENCV_METALIMGPROC_GRAPHCUT_INTERNAL_HPP

#ifdef HAVE_METAL

#include <opencv2/core/metal.hpp>

#ifdef __OBJC__
#import <Metal/Metal.h>
#else
typedef void* id;
#endif

namespace cv { namespace metal {

// Forward declarations
class MetalMat;
class Stream;

/**
 * @brief GPU-accelerated push-relabel max-flow / min-cut implementation for GrabCut.
 *
 * This is a *work-in-progress* skeleton that will be incrementally filled out.
 * For now the class provides stubbed public methods so that the project can
 * compile while we bring the full implementation online.
 */
class MetalGraphCut
{
public:
    /**
     * Construct a MetalGraphCut solver for an image of size \p sz.
     * All GPU resources are allocated upfront and reused across iterations.
     */
    MetalGraphCut(cv::Size sz, Stream& s);

    /**
     * Build/refresh graph capacities from unary and pairwise terms.
     * The MetalMat arguments are expected to be GPU-resident textures or
     * buffers matching the image size.
     */
    void buildGraph(const MetalMat& unary_bg,
                    const MetalMat& unary_fg,
                    const MetalMat& pairwise_left,
                    const MetalMat& pairwise_top,
                    const MetalMat& pairwise_topleft,
                    const MetalMat& pairwise_topright,
                    const MetalMat& mask,
                    double lambda);

    /**
     * Run the push-relabel optimisation for a fixed number of iterations.
     * A value of 1 corresponds roughly to one global relabel + push stage.
     */
    void solve(int iterations);

    /**
     * Retrieve the final segmentation mask into \\p mask. The mask must be a
     * MetalMat of type CV_8UC1 and same spatial dimensions as the image.
     */
    void getSegmentation(MetalMat& mask, const MetalMat& initial_mask);

    // DEBUG: Expose buffers for external inspection
    id<MTLBuffer> getNodeDataBufferForDebug() const { return m_nodeDataAtom; }
    id<MTLBuffer> getTerminalFlowBufferForDebug() const { return m_terminalFlow; }

private:
    // We deliberately keep the member list minimal for now – additional GPU
    // buffers and pipeline states will be introduced with subsequent commits.
    Stream&         m_stream;
    cv::Size        m_graphSize;

    // --- Buffers for upcoming push-relabel implementation ---
    id<MTLBuffer> m_terminalFlow  = nil; // source/sink capacities

    // BFS traversal buffers / Active Lists (ping-pong)
    id<MTLBuffer> m_activeList1    = nil;
    id<MTLBuffer> m_activeList2    = nil;
    id<MTLBuffer> m_levelCount   = nil; // uint[1] counter for active list size

    id<MTLBuffer> m_excessFlag   = nil; // uint[1] flag for push-relabel convergence
    
    // Phase 3 Optimization buffers
    id<MTLBuffer> m_heightHistogram = nil;
    
    // Phase 4 Final cut buffers
    id<MTLBuffer> m_finalCutLabels  = nil; // int array for final BFS reachability

    // --- Atomics refactor buffers (slice 1) ---
    id<MTLBuffer> m_nodeDataAtom   = nil; // NodeDataAtom array (atomic excess+label)
    id<MTLBuffer> m_residualAtom   = nil; // Residual4Atom array (atomic neighbour caps)
    id<MTLBuffer> m_capToSourceBuf = nil; // Residual capacity from pixel to source

    // Texture references for provisional segmentation
    id<MTLTexture> m_bgTermTex = nil;
    id<MTLTexture> m_fgTermTex = nil;

    // Allocation helper
    void allocateGraphBuffers();

    // Phase 3 Optimization helpers
    void runGlobalRelabel();
    void createActiveList();

    // Lazy-initialised compute pipeline for the provisional segmentation pass.
    static id<MTLComputePipelineState> getSimpleSegmentationPipeline();

    // Pipeline for building graph capacities (unary & pairwise)
    static id<MTLComputePipelineState> getBuildGraphPipeline();

    // Pipeline to initialize labels for BFS (sets source/sink labels)
    static id<MTLComputePipelineState> getBFSInitPipeline();
    static id<MTLComputePipelineState> getBFSTraversePipeline();
    static id<MTLComputePipelineState> getLabelSegmentationPipeline();
    
    // New pipelines for push-relabel and optimizations
    static id<MTLComputePipelineState> getGlobalRelabelInitPipeline();
    static id<MTLComputePipelineState> getGlobalRelabelBfsTraversePipeline();
    static id<MTLComputePipelineState> getCreateInitialActiveListPipeline();
    static id<MTLComputePipelineState> getPushRelabelPipeline();
    static id<MTLComputePipelineState> getBuildHeightHistogramPipeline();
    static id<MTLComputePipelineState> getGapRelabelPipeline();
    static id<MTLComputePipelineState> getFinalCutBfsInit_SourceSet_Pipeline(id<MTLDevice> device);
    static id<MTLComputePipelineState> getFinalCutBfsTraverse_SourceSet_Pipeline(id<MTLDevice> device);
    static id<MTLComputePipelineState> getFinalCutWriteMaskPipeline();
    static id<MTLComputePipelineState> getSimpleFinalCutPipeline();
};

}} // namespace cv::metal

#endif // HAVE_METAL
#endif // OPENCV_METALIMGPROC_GRAPHCUT_INTERNAL_HPP 