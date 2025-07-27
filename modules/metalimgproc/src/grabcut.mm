#include "precomp.hpp"

#ifdef HAVE_METAL

#include "opencv2/imgproc/detail/gcgraph.hpp"
#include "gmm_internal.hpp"
#include "graphcut_internal.hpp"
#include <memory>
namespace cv { namespace metal {

// Forward declaration for k-means from clustering.mm
double kmeans(const MetalMat& data, int K, MetalMat& bestLabels, 
             TermCriteria criteria, int attempts, int flags, MetalMat& centers, Stream& stream);

}} // end cv::metal namespace temporarily


namespace cv { namespace metal {

// Enhanced k-means initialization function that can use either CPU or Metal backend
void initGMMsWithKMeans(const cv::Mat& img, const cv::Mat& mask, cv::Mat& compIdxs, 
                       bool useMetalKMeans, uint64_t seed, Stream* stream) {
    const int kMeansItCount = 10;
    const int kMeansType = cv::KMEANS_PP_CENTERS;
    const int componentsCount = 5; // Match GMM::componentsCount
    
    // Set deterministic seed if provided
    if (seed != 0) {
        cv::theRNG().state = seed;
    }
    
    if (useMetalKMeans && stream) {
        printf("[MetalGrabCut DEBUG] Attempting Metal k-means path...\n");
        // METAL PATH: Use direct mask-based k-means without intermediate samples
        try {
            // Convert input image to BGRA format for Metal
            cv::Mat imgBGRA;
            if (img.channels() == 3) {
                cv::cvtColor(img, imgBGRA, cv::COLOR_BGR2BGRA);
            } else {
                imgBGRA = img.clone();
            }
            
            // Upload to Metal
            cv::metal::MetalMat metalImg, metalMask;
            metalImg.upload(imgBGRA);
            metalMask.upload(mask);
            
            // Run mask-based k-means for background and foreground separately
            cv::metal::MetalMat bgLabels, fgLabels;
            cv::Mat bgCentroids, fgCentroids;
            
            // Cluster background pixels (GC_BGD and GC_PR_BGD)
            cv::metal::kmeansClusterByMask(metalImg, metalMask, true, bgLabels, bgCentroids, *stream);
            
            // Cluster foreground pixels (GC_FGD and GC_PR_FGD)
            cv::metal::kmeansClusterByMask(metalImg, metalMask, false, fgLabels, fgCentroids, *stream);
            
            // 🚀 PHASE 1.3 GPU OPTIMIZATION: Replace CPU loops with GPU parallel processing
            // This eliminates ~80MB data transfers and CPU pixel-by-pixel processing
            
            // Create MetalMat for component assignments (reuse existing metalMask)
            cv::metal::MetalMat metalComponents;
            
            // GPU Component Assignment - replaces downloads + CPU loops
            cv::metal::assignComponents(metalMask, bgLabels, fgLabels, metalComponents, *stream);
            
            // Download only the final component assignments (much smaller than k-means labels)
            metalComponents.download(compIdxs, *stream, true);
            
            return; // Success - exit early
            
        } catch (const cv::Exception& e) {
            printf("[MetalGrabCut DEBUG] Metal k-means path failed with exception: %s\n", e.what());
            printf("[MetalGrabCut DEBUG] Falling back to CPU k-means path.\n");
            // Fall through to CPU implementation
        }
    }
    
    // CPU PATH: Traditional sample-based approach (fallback or when Metal disabled)
    printf("[MetalGrabCut DEBUG] Using CPU k-means path.\n");
    
    cv::Mat bgdLabels, fgdLabels;
    std::vector<cv::Vec3f> bgdSamples, fgdSamples;
    
    // Collect background and foreground pixel samples
    cv::Point p;
    for (p.y = 0; p.y < img.rows; p.y++) {
        for (p.x = 0; p.x < img.cols; p.x++) {
            if (mask.at<uchar>(p) == cv::GC_BGD || mask.at<uchar>(p) == cv::GC_PR_BGD) {
                bgdSamples.push_back((cv::Vec3f)img.at<cv::Vec3b>(p));
            } else { // cv::GC_FGD | cv::GC_PR_FGD
                fgdSamples.push_back((cv::Vec3f)img.at<cv::Vec3b>(p));
            }
        }
    }
    
    CV_Assert(!bgdSamples.empty() && !fgdSamples.empty());
    
    // --- BEGIN DEBUG: Print sample counts ---
    printf("[MetalGrabCut DEBUG] K-Means Sampling (CPU Path): bgd_samples=%zu, fgd_samples=%zu\n",
           bgdSamples.size(), fgdSamples.size());
    // --- END DEBUG ---
    
    // Perform k-means clustering on background samples
    cv::Mat bgdCenters, fgdCenters;
    {
        cv::Mat _bgdSamples((int)bgdSamples.size(), 3, CV_32FC1, &bgdSamples[0][0]);
        int num_clusters = componentsCount;
        num_clusters = std::min(num_clusters, (int)bgdSamples.size());
        

        cv::kmeans(_bgdSamples, num_clusters, bgdLabels,
                  cv::TermCriteria(cv::TermCriteria::MAX_ITER, kMeansItCount, 0.0), 0, kMeansType, bgdCenters);
        
        // Validate K-means results to prevent bounds errors
        CV_Assert(!bgdLabels.empty() && bgdLabels.cols >= 1 && bgdLabels.type() == CV_32SC1);
        

    }

    // Perform k-means clustering on foreground samples
    {
        cv::Mat _fgdSamples((int)fgdSamples.size(), 3, CV_32FC1, &fgdSamples[0][0]);
        int num_clusters = componentsCount;
        num_clusters = std::min(num_clusters, (int)fgdSamples.size());
        

        cv::kmeans(_fgdSamples, num_clusters, fgdLabels,
                  cv::TermCriteria(cv::TermCriteria::MAX_ITER, kMeansItCount, 0.0), 0, kMeansType, fgdCenters);
        
        // Validate K-means results to prevent bounds errors
        CV_Assert(!fgdLabels.empty() && fgdLabels.cols >= 1 && fgdLabels.type() == CV_32SC1);
        

    }
    
    // Create component index map for all pixels
    compIdxs.create(img.size(), CV_8UC1); // Use 8-bit to match Metal texture format
    compIdxs.setTo(0); // Initialize all pixels to component 0
    
    // Count CPU component assignment
    std::vector<int> cpuBgComponentCounts(componentsCount, 0);
    std::vector<int> cpuFgComponentCounts(componentsCount, 0);
    
    // Assign background pixels to their k-means clusters
    int bgdIdx = 0;
    for (p.y = 0; p.y < img.rows; p.y++) {
        for (p.x = 0; p.x < img.cols; p.x++) {
            if (mask.at<uchar>(p) == cv::GC_BGD || mask.at<uchar>(p) == cv::GC_PR_BGD) {
                // Check bounds before matrix access
                if (bgdIdx < bgdLabels.rows && bgdLabels.cols > 0) {
                    int label = bgdLabels.at<int>(bgdIdx, 0);
                    // Ensure label is within valid component range [0, componentsCount-1]
                    label = std::max(0, std::min(label, componentsCount - 1));
                    compIdxs.at<uchar>(p) = (uchar)label;
                    cpuBgComponentCounts[label]++;
                    bgdIdx++;
                } else {
                    compIdxs.at<uchar>(p) = 0; // Fallback to component 0
                    cpuBgComponentCounts[0]++;
                }
            }
        }
    }
    
    // Assign foreground pixels to their k-means clusters
    int fgdIdx = 0;
    for (p.y = 0; p.y < img.rows; p.y++) {
        for (p.x = 0; p.x < img.cols; p.x++) {
            if (mask.at<uchar>(p) == cv::GC_FGD || mask.at<uchar>(p) == cv::GC_PR_FGD) {
                // Check bounds before matrix access
                if (fgdIdx < fgdLabels.rows && fgdLabels.cols > 0) {
                    int label = fgdLabels.at<int>(fgdIdx, 0);
                    // Ensure label is within valid component range [0, componentsCount-1]
                    label = std::max(0, std::min(label, componentsCount - 1));
                    // CRITICAL FIX: Components are always 0-4, mask determines which GMM to use
                    compIdxs.at<uchar>(p) = (uchar)label;
                    cpuFgComponentCounts[label]++;
                    fgdIdx++;
                } else {
                    // CRITICAL FIX: Components are always 0-4, mask determines which GMM to use
                    compIdxs.at<uchar>(p) = 0;
                    cpuFgComponentCounts[0]++;
                }
            }
        }
    }
    

}

// Legacy CPU-only version for backwards compatibility
void initGMMsWithKMeans(const cv::Mat& img, const cv::Mat& mask, cv::Mat& compIdxs) {
    initGMMsWithKMeans(img, mask, compIdxs, false, 0, nullptr);
}

}} // cv::metal

// CPU-only entry point outside cv::metal namespace for global access
void initGMMsWithKMeans(const cv::Mat& img, const cv::Mat& mask, cv::Mat& compIdxs) {
    cv::metal::initGMMsWithKMeans(img, mask, compIdxs, false, 0, nullptr);
}

namespace cv { namespace metal {

// Forward declarations for utility functions
void trimapFromRect(MetalMat& mask, const Rect& rect, Stream& stream);
double calcBeta(const MetalMat& image, Stream& stream);
id<MTLBuffer> calcBetaAsync(const MetalMat& image, Stream& stream);
void calcNWeights(const MetalMat& image, MetalMat& leftW, MetalMat& topleftW, MetalMat& topW, MetalMat& toprightW, 
                  id<MTLBuffer> betaBuffer, double gamma, Stream& stream);
bool checkConvergence(const MetalMat& oldMask, const MetalMat& newMask, Stream& stream);
id<MTLBuffer> checkConvergenceAsync(const MetalMat& oldMask, const MetalMat& newMask, Stream& stream);

// Internal GrabCut implementation class
class GrabCutImpl {
public:
    GrabCutImpl();
    ~GrabCutImpl();
    
    void run(const MetalMat& image, MetalMat& mask, const Rect& rect,
             MetalMat& bgdModel, MetalMat& fgdModel,
             int iterCount, int mode, bool useGpuGraphCut, Stream& stream);
    
    // PHASE 4: Direct access to GPU GMM buffers to avoid transfers
    void getGMMBuffers(id<MTLBuffer>& bgBuffer, id<MTLBuffer>& fgBuffer) const;

    void runWithSharedKMeans(const MetalMat& image, MetalMat& mask, const Rect& rect,
                            MetalMat& bgdModel, MetalMat& fgdModel,
                            int iterCount, int mode, uint64_t randomSeed, Stream& stream);
    
    // Debug access to GraphCut solver
    const MetalGraphCut& getGraphCutSolver() const { return *m_metalGraphCut; }

private:
    void initMaskWithRect(MetalMat& mask, Size imageSize, const Rect& rect, Stream& stream);
    void checkMask(const MetalMat& image, const MetalMat& mask);
    
    // Graph construction and max-flow (CPU implementation)
    void constructGCGraph(const MetalMat& image, const MetalMat& mask, 
                         const MetalMat& bgTerm, const MetalMat& fgTerm,
                         const MetalMat& leftW, const MetalMat& topleftW, 
                         const MetalMat& topW, const MetalMat& toprightW,
                         double lambda, ::cv::detail::GCGraph<double>& graph, Stream& stream);
    
    void estimateSegmentation(::cv::detail::GCGraph<double>& graph, MetalMat& mask, Stream& stream);
    
    ::cv::Ptr<GMM> m_gmm;
    MetalMat m_pairwiseWeights[4]; // left, topleft, top, topright
    MetalMat m_components;
    MetalMat m_bgTerm, m_fgTerm;
    MetalMat m_prevMask;
    Size m_imageSize;
    double m_beta;
    bool m_initialized;

    // Placeholder for upcoming full-GPU graph-cut implementation
    std::unique_ptr<MetalGraphCut> m_metalGraphCut;

    // Unified K-means component initialization using shared CPU/Metal utility
    void initGMMsWithMetalKMeans(const MetalMat& imageBGRA, MetalMat& mask, uint64_t seed, Stream& stream) {
        if (m_components.empty()) {
            // Download to CPU for the shared utility function
            cv::Mat h_img, h_mask;
            imageBGRA.download(h_img, stream, true);
            mask.download(h_mask, stream, true);
            
            // Convert BGRA to BGR for OpenCV compatibility
            cv::Mat h_bgr;
            cv::cvtColor(h_img, h_bgr, cv::COLOR_BGRA2BGR);
            
                         // Use Metal K-means for component initialization
             cv::Mat componentMap;
             initGMMsWithKMeans(h_bgr, h_mask, componentMap, true, seed, &stream);
            
            // Upload the component map back to Metal
            m_components.upload(componentMap);
        }
        m_gmm->learnGMMs(imageBGRA, mask, m_components, stream);
        stream.syncCPU();
    }
};

GrabCutImpl::GrabCutImpl() : m_beta(0.0), m_initialized(false) {
}

GrabCutImpl::~GrabCutImpl() {
    // Smart pointers and MetalMat handle cleanup automatically
}

// PHASE 4: Direct access to GPU GMM buffers to avoid transfers
void GrabCutImpl::getGMMBuffers(id<MTLBuffer>& bgBuffer, id<MTLBuffer>& fgBuffer) const {
    if (m_gmm) {
        bgBuffer = m_gmm->getBgBuffer();
        fgBuffer = m_gmm->getFgBuffer();
    } else {
        bgBuffer = nil;
        fgBuffer = nil;
    }
}

void GrabCutImpl::run(const MetalMat& image, MetalMat& mask, const Rect& rect,
                      MetalMat& bgdModel, MetalMat& fgdModel,
                      int iterCount, int mode, bool useGpuGraphCut, Stream& stream) {
    
    CV_Assert(!image.empty());
    
    CV_Assert(image.type() == CV_8UC4 || image.type() == CV_8UC3);
    
    m_imageSize = image.size();
    
    // Initialize GMM if not already done
    if (!m_gmm || m_imageSize != m_gmm->size()) {
        m_gmm = createGMM(m_imageSize);
        m_initialized = false;
    }
    
    // Initialize mask and models if needed
    if (mode == GC_INIT_WITH_RECT || mode == GC_INIT_WITH_MASK) {
        if (mode == GC_INIT_WITH_RECT) {
            initMaskWithRect(mask, m_imageSize, rect, stream);
            
            // --- BEGIN DEBUG: Inspect mask after initMaskWithRect ---
            printf("\n[MetalGrabCut DEBUG] Inspecting mask state immediately after initMaskWithRect...\n");
            stream.syncCPU(); // Ensure kernel is finished

            cv::Mat h_init_mask;
            mask.download(h_init_mask, stream, true);

            int sure_bg = 0, sure_fg = 0, pr_bg = 0, pr_fg = 0, other = 0;
            for (int y = 0; y < h_init_mask.rows; ++y) {
                for (int x = 0; x < h_init_mask.cols; ++x) {
                    uchar val = h_init_mask.at<uchar>(y, x);
                    if (val == cv::GC_BGD) sure_bg++;
                    else if (val == cv::GC_FGD) sure_fg++;
                    else if (val == cv::GC_PR_BGD) pr_bg++;
                    else if (val == cv::GC_PR_FGD) pr_fg++;
                    else other++;
                }
            }
            printf("[MetalGrabCut DEBUG] Mask Counts: SureBG=%d, SureFG=%d, ProbBG=%d, ProbFG=%d, Other=%d\n\n",
                   sure_bg, sure_fg, pr_bg, pr_fg, other);
            // --- END DEBUG ---
            
        } else {
            checkMask(image, mask);
        }
        
        // Initialize GMMs only if not already initialized
        if (!m_initialized) {
            // CRITICAL: Do proper K-means + initial learning like CPU
            // This ensures GMM parameters are ready for assignment from iteration 0
            
            // Download CPU versions for k-means clustering and initial learning
            cv::Mat h_image, h_mask;
            image.download(h_image, stream, true);
            mask.download(h_mask, stream, true);
            
            // Convert BGRA to BGR if needed
            cv::Mat h_img_bgr;
            if (h_image.channels() == 4) {
                cv::cvtColor(h_image, h_img_bgr, cv::COLOR_BGRA2BGR);
            } else {
                h_img_bgr = h_image;
            }
            
            // Perform k-means clustering using shared utility (try Metal k-means first)
            cv::Mat h_components;
            cv::metal::initGMMsWithKMeans(h_img_bgr, h_mask, h_components, true, 0, &stream);
            
            // Upload the k-means component assignments to Metal
            m_components.upload(h_components);
            
            // Do initial GMM learning using K-means assignments on the input stream
            m_gmm->learnGMMs(image, mask, m_components, stream);
            
            // Phase 4: Keep GMM parameters on GPU during iterations
            // Only extract when we need output models (initialization or final result)
            if (iterCount <= 0) {
                // Synchronize only when we need to extract parameters for output
                stream.syncCPU();
                
                // Extract learned parameters for 0 iterations case
                Mat tempBgdModel, tempFgdModel;
                m_gmm->extractGMMParameters(tempBgdModel, tempFgdModel);
                bgdModel.upload(tempBgdModel);
                fgdModel.upload(tempFgdModel);
            }
            
            m_initialized = true;
        }
    }
    
    // Calculate beta parameter and pairwise weights for all modes that need graph construction
    // PHASE 1 OPTIMIZATION: Use async beta calculation to eliminate CPU synchronization
    id<MTLBuffer> betaBuffer = calcBetaAsync(image, stream);
    
    // Calculate pairwise weights using async beta buffer
    const double gamma = 50.0;
    calcNWeights(image, m_pairwiseWeights[0], m_pairwiseWeights[1], 
                m_pairwiseWeights[2], m_pairwiseWeights[3], 
                betaBuffer, gamma, stream);
    
    // Keep the synchronous beta for legacy compatibility (only sync when needed)
    m_beta = calcBeta(image, stream);
    
    if (iterCount <= 0) return;
    
    if (mode == GC_EVAL_FREEZE_MODEL) {
        iterCount = 1;
    }
    
    if (mode == GC_EVAL || mode == GC_EVAL_FREEZE_MODEL) {
        checkMask(image, mask);
    }
    
    // Create previous mask for convergence checking
    m_prevMask.create(mask.size(), mask.type());
    
    // Main GrabCut iterative loop
    const double lambda = 9.0 * 50.0; // 9 * gamma (450)
    
    MetalMat initial_mask;
    if (useGpuGraphCut) {
        initial_mask = mask.clone();
    }
    
    for (int iter = 0; iter < iterCount; iter++) {
        
        // Save current mask for convergence check
        m_prevMask = mask.clone();

        // PHASE 1: Batch all GPU operations on the input stream (NO intermediate commits)
        // This allows GPU operations to pipeline and overlap for maximum performance
        
        // Ensure components texture exists for assignment
        if (m_components.empty()) {
            m_components.create(mask.size(), CV_8UC1);
        }
        
        m_gmm->assignGMMs(image, mask, m_components, stream);
        
        // Learn GMM parameters (except in freeze model mode)  
        if (mode != GC_EVAL_FREEZE_MODEL) {
            m_gmm->learnGMMs(image, mask, m_components, stream);
            
            // Phase 4: Keep GMM parameters on GPU during iterations
            // No extractGMMParameters call here - parameters stay GPU-resident
            
            // DETAILED TRACKING: Show GMM parameters AFTER learning
            // Debug: Extract and inspect GMM parameters
            if (iter == 0 && useGpuGraphCut) {
                stream.syncCPU();
                
                cv::Mat bgModel, fgModel;
                m_gmm->extractGMMParameters(bgModel, fgModel);
                
                printf("[MetalGrabCut DEBUG] GMM Parameters after learning (iter %d):\n", iter);
                
                // CRITICAL FIX: Read as float, not double - GPU uses float!
                const float* fgData = fgModel.ptr<float>(0);
                const float* bgData = bgModel.ptr<float>(0);
                
                // GMM buffer layout from extractGMMParameters:
                // [weights(5), means(15), covariances(45)] = 65 total floats
                printf("  FG Model:\n");
                for (int c = 0; c < 5; c++) {
                    float weight = fgData[c];  // weights are first 5 values
                    const float* mean = &fgData[5 + c * 3];  // means start at 5
                    const float* cov = &fgData[5 + 15 + c * 9];  // covs start at 20
                    
                    printf("    Component %d: weight=%.6f, mean=(%.2f,%.2f,%.2f)\n", 
                           c, weight, mean[0], mean[1], mean[2]);
                    
                    // Calculate covariance determinant
                    float det = cov[0]*(cov[4]*cov[8]-cov[5]*cov[7]) -
                                cov[1]*(cov[3]*cov[8]-cov[5]*cov[6]) +
                                cov[2]*(cov[3]*cov[7]-cov[4]*cov[6]);
                    printf("      Cov determinant: %.9f\n", det);
                    
                    // Check diagonal elements
                    printf("      Cov diagonal: [%.6f, %.6f, %.6f]\n", cov[0], cov[4], cov[8]);
                    
                    if (det <= 1e-9) {
                        printf("      WARNING: Singular or near-singular covariance!\n");
                    }
                }
                
                printf("  BG Model:\n"); 
                for (int c = 0; c < 5; c++) {
                    float weight = bgData[c];
                    const float* mean = &bgData[5 + c * 3];
                    printf("    Component %d: weight=%.6f, mean=(%.2f,%.2f,%.2f)\n", 
                           c, weight, mean[0], mean[1], mean[2]);
                }
            }
            
        }
        
        // Compute unary potentials (data term) - still on same stream
        m_gmm->computeDataTerm(image, mask, m_bgTerm, m_fgTerm, stream);

        // --- DEBUG: Dump unary term textures ---
        if (iter == 0 && useGpuGraphCut)
        {
            printf("[MetalGrabCut DEBUG] Dumping unary term textures...\n");
            stream.syncCPU(); // Ensure computeDataTerm is finished

            cv::Mat h_bgTerm, h_fgTerm;
            m_bgTerm.download(h_bgTerm, stream, true);
            m_fgTerm.download(h_fgTerm, stream, true);

            double bg_min, bg_max, fg_min, fg_max;
            cv::minMaxLoc(h_bgTerm, &bg_min, &bg_max);
            cv::minMaxLoc(h_fgTerm, &fg_min, &fg_max);

            printf("[MetalGrabCut DEBUG] Unary BG cost (capacity from Source): min=%.6f, max=%.6f\n", bg_min, bg_max);
            printf("[MetalGrabCut DEBUG] Unary FG cost (capacity to Sink):   min=%.6f, max=%.6f\n", fg_min, fg_max);

            // --- BEGIN DETAILED UNARY COST ANALYSIS ---
            cv::Mat h_mask;
            mask.download(h_mask, stream, true);

            std::vector<float> pr_fgd_bg_costs;
            std::vector<float> pr_fgd_fg_costs;
            int pr_fgd_count = 0;

            for (int y = 0; y < h_mask.rows; ++y) {
                for (int x = 0; x < h_mask.cols; ++x) {
                    if (h_mask.at<uchar>(y, x) == cv::GC_PR_FGD) {
                        pr_fgd_count++;
                        pr_fgd_bg_costs.push_back(h_bgTerm.at<float>(y, x));
                        pr_fgd_fg_costs.push_back(h_fgTerm.at<float>(y, x));
                    }
                }
            }

            if (pr_fgd_count > 0) {
                double sum_bg_costs = 0, sum_fg_costs = 0;
                float min_bg_cost = pr_fgd_bg_costs[0], max_bg_cost = pr_fgd_bg_costs[0];
                float min_fg_cost = pr_fgd_fg_costs[0], max_fg_cost = pr_fgd_fg_costs[0];

                for (float cost : pr_fgd_bg_costs) {
                    sum_bg_costs += cost;
                    if (cost < min_bg_cost) min_bg_cost = cost;
                    if (cost > max_bg_cost) max_bg_cost = cost;
                }
                for (float cost : pr_fgd_fg_costs) {
                    sum_fg_costs += cost;
                    if (cost < min_fg_cost) min_fg_cost = cost;
                    if (cost > max_fg_cost) max_fg_cost = cost;
                }

                double mean_bg_cost = sum_bg_costs / pr_fgd_count;
                double mean_fg_cost = sum_fg_costs / pr_fgd_count;

                printf("[MetalGrabCut DEBUG] Analysis for %d GC_PR_FGD pixels:\n", pr_fgd_count);
                printf("[MetalGrabCut DEBUG]   - BG Cost (from Source):  min=%.4f, max=%.4f, mean=%.4f\n", min_bg_cost, max_bg_cost, mean_bg_cost);
                printf("[MetalGrabCut DEBUG]   - FG Cost (to Sink):      min=%.4f, max=%.4f, mean=%.4f\n", min_fg_cost, max_fg_cost, mean_fg_cost);
            } else {
                printf("[MetalGrabCut DEBUG] No GC_PR_FGD pixels found for unary cost analysis.\n");
            }
            // --- END DETAILED UNARY COST ANALYSIS ---
        }
        // --- END DEBUG ---
        
        if (useGpuGraphCut) {
            // ---------------------------------------------------------------------------------
            // FULL-GPU PATH
            // ---------------------------------------------------------------------------------
            // Keep everything on the GPU – do NOT syncCPU().
            if (!m_metalGraphCut)
            {
                m_metalGraphCut.reset(new MetalGraphCut(image.size(), stream));
            }

            // Build the graph directly on the GPU from unary & pairwise terms
            printf("[MetalGrabCut] Calling buildGraph for %dx%d image\n", image.cols(), image.rows());
            m_metalGraphCut->buildGraph(m_bgTerm, m_fgTerm,
                                        m_pairwiseWeights[0], m_pairwiseWeights[2],
                                        m_pairwiseWeights[1], m_pairwiseWeights[3],
                                        mask, lambda);
            printf("[MetalGrabCut] buildGraph returned\n");

            // --- BEGIN DEBUG: Inspect graph state after buildGraph ---
            if (iter == 0) {
                printf("\n[MetalGrabCut DEBUG] Inspecting graph state after buildGraph...\n");
                stream.syncCPU(); // Ensure buildGraph is finished

                // Get buffers from solver
                const MetalGraphCut& solver = *m_metalGraphCut;
                id<MTLBuffer> nodeDataBuffer = solver.getNodeDataBufferForDebug();
                id<MTLBuffer> terminalFlowBuffer = solver.getTerminalFlowBufferForDebug();
                
                // Define CPU-side structs to interpret buffer data
                struct CpuNodeDataAtom {
                    uint32_t excessBits;
                    int32_t label;
                };
                struct CpuTerminalFlow {
                    uint32_t to_source_bits;
                    uint32_t to_sink_bits;
                };

                // Download buffers and initial mask
                cv::Mat h_mask;
                mask.download(h_mask, stream, true);
                
                NSUInteger nodeCount = image.cols() * image.rows();
                CpuNodeDataAtom* nodeData = (CpuNodeDataAtom*)[nodeDataBuffer contents];
                CpuTerminalFlow* termFlow = (CpuTerminalFlow*)[terminalFlowBuffer contents];

                // Counters for analysis
                int sure_bg_count = 0, sure_fg_count = 0, pr_bg_count = 0, pr_fg_count = 0;
                double total_excess = 0.0;
                int nodes_with_excess = 0;
                int probable_nodes_with_excess = 0;
                double total_sink_cap = 0.0, total_source_cap = 0.0;

                for (NSUInteger i = 0; i < nodeCount; ++i) {
                    uchar maskVal = h_mask.at<uchar>(i);
                    float excess = *(float*)&nodeData[i].excessBits;
                    float cap_to_sink = *(float*)&termFlow[i].to_sink_bits;
                    
                    // Count mask values
                    if (maskVal == cv::GC_BGD) sure_bg_count++;
                    else if (maskVal == cv::GC_FGD) sure_fg_count++;
                    else if (maskVal == cv::GC_PR_BGD) pr_bg_count++;
                    else if (maskVal == cv::GC_PR_FGD) pr_fg_count++;
                    
                    // Analyze excess flow
                    if (excess > 1e-6f) {
                        nodes_with_excess++;
                        total_excess += excess;
                        if (maskVal == cv::GC_PR_FGD || maskVal == cv::GC_PR_BGD) {
                            probable_nodes_with_excess++;
                        }
                    }
                    
                    // Analyze terminal capacities
                    // Note: fromSource capacity becomes initial excess, so we check that.
                    // to_sink capacity is what we check from the terminal flow buffer.
                    total_source_cap += excess;
                    total_sink_cap += cap_to_sink;
                }

                printf("[MetalGrabCut DEBUG] Initial Mask Counts: SureBG=%d, SureFG=%d, ProbBG=%d, ProbFG=%d\n",
                       sure_bg_count, sure_fg_count, pr_bg_count, pr_fg_count);
                printf("[MetalGrabCut DEBUG] Initial Excess Flow: total=%.4f, nodes_with_excess=%d, probable_nodes_with_excess=%d\n",
                       total_excess, nodes_with_excess, probable_nodes_with_excess);
                printf("[MetalGrabCut DEBUG] Terminal Capacities: total_from_source=%.4f, total_to_sink=%.4f\n\n",
                       total_source_cap, total_sink_cap);
            }
            // --- END DEBUG ---

            // The push-relabel algorithm for max-flow is highly iterative and must run
            // until it converges (i.e., no more "active" nodes with excess flow).
            // We provide a generous iteration limit as a safeguard; the solver in
            // graphcut.mm is designed to terminate early once convergence is reached.
            // Note: Some nodes may need to relabel to height totalNodes+1 to push to source
            const int maxFlowIterations = std::max(4000, (int)(10 * std::sqrt(mask.cols() * mask.rows())));
            m_metalGraphCut->solve(maxFlowIterations);

            // Retrieve updated mask (GPU → GPU). Note: getSegmentation writes into
            // an existing MetalMat to avoid reallocations.
            m_metalGraphCut->getSegmentation(mask, initial_mask);

        } else {
            // ---------------------------------------------------------------------------------
            // CPU FALLBACK PATH
            // ---------------------------------------------------------------------------------

            // PHASE 2: Let downloads handle synchronization automatically
            // First download will sync if commands are queued, subsequent downloads will be fast

            // PHASE 3: CPU-bound operations (these don't use streams since they're CPU-only)
            // Construct graph and solve min-cut max-flow on CPU
            ::cv::detail::GCGraph<double> graph;
            constructGCGraph(image, mask, m_bgTerm, m_fgTerm,
                            m_pairwiseWeights[0], m_pairwiseWeights[1],
                            m_pairwiseWeights[2], m_pairwiseWeights[3],
                            lambda, graph, stream);
            
            // Estimate segmentation using max-flow
            estimateSegmentation(graph, mask, stream);
        }
        
        // Check for convergence
        if (checkConvergence(m_prevMask, mask, stream)) {
            break;
        }
    }
    
    // Phase 4: Extract final GMM parameters after all iterations complete
    // This ensures output models contain the final learned parameters
    if (iterCount > 0) {
        stream.syncCPU();
        Mat tempBgdModel, tempFgdModel;
        m_gmm->extractGMMParameters(tempBgdModel, tempFgdModel);
        bgdModel.upload(tempBgdModel);
        fgdModel.upload(tempFgdModel);
    }
}


void GrabCutImpl::initMaskWithRect(MetalMat& mask, Size imageSize, const Rect& rect, Stream& stream) {
    mask.create(imageSize, CV_8UC1);
    trimapFromRect(mask, rect, stream);
}

void GrabCutImpl::checkMask(const MetalMat& image, const MetalMat& mask) {
    CV_Assert(!mask.empty());
    CV_Assert(mask.type() == CV_8UC1);
    CV_Assert(mask.size() == image.size());
    
    // Additional validation could be done here
    // For now, we trust that the mask contains valid GrabCut class values
}

void GrabCutImpl::constructGCGraph(const MetalMat& image, const MetalMat& mask,
                                  const MetalMat& bgTerm, const MetalMat& fgTerm,
                                  const MetalMat& leftW, const MetalMat& topleftW,
                                  const MetalMat& topW, const MetalMat& toprightW,
                                  double lambda, cv::detail::GCGraph<double>& graph, Stream& stream) {
    
    // Download GPU data to CPU for graph construction
    // Note: Use stream-aware downloads for proper synchronization
    Mat h_mask, h_bgTerm, h_fgTerm;
    Mat h_leftW, h_topleftW, h_topW, h_toprightW;
    
    mask.download(h_mask, stream, true);
    bgTerm.download(h_bgTerm, stream, true);
    fgTerm.download(h_fgTerm, stream, true);
    leftW.download(h_leftW, stream, true);
    topleftW.download(h_topleftW, stream, true);
    topW.download(h_topW, stream, true);
    toprightW.download(h_toprightW, stream, true);
    
    int vtxCount = image.cols() * image.rows();
    int edgeCount = 2 * (4 * image.cols() * image.rows() - 3 * (image.cols() + image.rows()) + 2);
    graph.create(vtxCount, edgeCount);
    
    // Construct graph on CPU (following OpenCV's implementation)
    for (int y = 0; y < image.rows(); y++) {
        for (int x = 0; x < image.cols(); x++) {
            int vtxIdx = graph.addVtx();
            uchar maskValue = h_mask.at<uchar>(y, x);
            
            // Set terminal weights (unary potentials)
            double fromSource, toSink;
            if (maskValue == GC_PR_BGD || maskValue == GC_PR_FGD) {
                fromSource = h_bgTerm.at<float>(y, x);      // Cost to connect to SOURCE (background model cost, like CPU)
                toSink = h_fgTerm.at<float>(y, x);          // Cost to connect to SINK (foreground model cost, like CPU)
            } else { // GC_BGD or GC_FGD
                if (maskValue == GC_BGD) {
                    fromSource = 0;
                    toSink = lambda;
                } else { // GC_FGD
                    fromSource = lambda;
                    toSink = 0;
                }
            }
            graph.addTermWeights(vtxIdx, fromSource, toSink);
            
            // Set neighborhood weights (pairwise potentials)
            if (x > 0) {
                cv::Vec4f v_left = h_leftW.at<cv::Vec4f>(y, x);
                double w = v_left[0];
                graph.addEdges(vtxIdx, vtxIdx - 1, w, w);
            }
            if (x > 0 && y > 0) {
                cv::Vec4f v_topleft = h_topleftW.at<cv::Vec4f>(y, x);
                double w = v_topleft[0];
                graph.addEdges(vtxIdx, vtxIdx - image.cols() - 1, w, w);
            }
            if (y > 0) {
                cv::Vec4f v_top = h_topW.at<cv::Vec4f>(y, x);
                double w = v_top[0];
                graph.addEdges(vtxIdx, vtxIdx - image.cols(), w, w);
            }
            if (x < image.cols() - 1 && y > 0) {
                cv::Vec4f v_topright = h_toprightW.at<cv::Vec4f>(y, x);
                double w = v_topright[0];
                graph.addEdges(vtxIdx, vtxIdx - image.cols() + 1, w, w);
            }
        }
    }
}

void GrabCutImpl::estimateSegmentation(cv::detail::GCGraph<double>& graph, MetalMat& mask, Stream& stream) {
    // Solve max-flow min-cut on CPU
    graph.maxFlow();
    
    // Create CPU mask for result
    Mat h_mask(mask.size(), CV_8UC1);
    Mat h_currentMask;
    mask.download(h_currentMask, stream, true);
    
    // Convert from Metal texture format back to GrabCut mask values
    // Metal stores GrabCut values directly as discrete integers [0,1,2,3]
    for (int y = 0; y < h_currentMask.rows; y++) {
        for (int x = 0; x < h_currentMask.cols; x++) {
            uchar metalValue = h_currentMask.at<uchar>(y, x);
            // Metal stores GrabCut values directly as discrete integers [0,1,2,3]
            uchar grabcutValue = metalValue;
            if (grabcutValue <= 3) { // Valid GrabCut values
                h_currentMask.at<uchar>(y, x) = grabcutValue;
            } else {
                // Invalid value, default to background
                h_currentMask.at<uchar>(y, x) = 0;
            }
        }
    }
    
    // Update mask based on graph cut results
    for (int y = 0; y < mask.rows(); y++) {
        for (int x = 0; x < mask.cols(); x++) {
            uchar currentValue = h_currentMask.at<uchar>(y, x);
            if (currentValue == GC_PR_BGD || currentValue == GC_PR_FGD) {
                int vtxIdx = y * mask.cols() + x;
                if (graph.inSourceSegment(vtxIdx)) {
                    h_mask.at<uchar>(y, x) = GC_PR_FGD;
                } else {
                    h_mask.at<uchar>(y, x) = GC_PR_BGD;
                }
            } else {
                h_mask.at<uchar>(y, x) = currentValue; // Keep sure pixels unchanged
            }
        }
    }
    
    // Convert GrabCut mask values back to Metal texture format before upload
    Mat h_metalMask(mask.size(), CV_8UC1);
    for (int y = 0; y < h_mask.rows; y++) {
        for (int x = 0; x < h_mask.cols; x++) {
            uchar grabcutValue = h_mask.at<uchar>(y, x);
            // Store GrabCut values directly as discrete integers [0,1,2,3]
            // Metal kernels now expect these values and convert via * 255.0f + 0.5f
            h_metalMask.at<uchar>(y, x) = grabcutValue;
        }
    }
    
    // Upload result back to GPU
    mask.upload(h_metalMask);
    
    // Note: mask.upload() is synchronous, so the mask is immediately available for GPU operations
}

namespace {

// Forward declaration of functions used in grabCut

} // anonymous namespace

// Public API implementation
void grabCut(InputArray _img, InputOutputArray _mask, Rect rect,
             InputOutputArray _bgdModel, InputOutputArray _fgdModel,
             int iterCount, int mode, bool useGpuGraphCut, Stream& stream) {
    
    printf("[cv::metal::grabCut] Entry - image %dx%d, iterCount=%d, mode=%d, useGpuGraphCut=%d\n",
           _img.cols(), _img.rows(), iterCount, mode, useGpuGraphCut);
    
    CV_Assert(!_img.empty());
    CV_Assert(_img.type() == CV_8UC3);
    
    Mat img = _img.getMat();
    Mat& mask = _mask.getMatRef();
    Mat& bgdModel = _bgdModel.getMatRef();
    Mat& fgdModel = _fgdModel.getMatRef();
    
    // Convert BGR to BGRA for Metal processing
    Mat imgBGRA;
    if (img.channels() == 3) {
        cvtColor(img, imgBGRA, COLOR_BGR2BGRA);
    } else {
        imgBGRA = img;
    }
    
    // Upload to Metal
    MetalMat d_img(imgBGRA);
    MetalMat d_mask;
    
    if (!mask.empty()) {
        // Convert user's GrabCut mask to Metal format before upload
        Mat metalMask(mask.size(), CV_8UC1);
        for (int y = 0; y < mask.rows; y++) {
            for (int x = 0; x < mask.cols; x++) {
                uchar grabcutValue = mask.at<uchar>(y, x);
                // Store GrabCut values directly as discrete integers [0,1,2,3]
                // Metal kernels now expect these values and convert via * 255.0f + 0.5f
                metalMask.at<uchar>(y, x) = grabcutValue;
            }
        }
        d_mask.upload(metalMask);
    } else {
        d_mask.create(img.size(), CV_8UC1);
    }
    
    // Create Metal buffers for model data (currently unused in this implementation)
    MetalMat d_bgdModel, d_fgdModel;
    
    // Create and run GrabCut implementation
    GrabCutImpl grabCutImpl; // Create fresh instance to avoid Metal object retention issues
    grabCutImpl.run(d_img, d_mask, rect, d_bgdModel, d_fgdModel, iterCount, mode, useGpuGraphCut, stream);
    
    // Download result
    Mat metalResult;
    d_mask.download(metalResult, stream, true);
    
    // Convert from Metal format back to standard GrabCut mask values
    mask.create(metalResult.size(), CV_8UC1);
    for (int y = 0; y < metalResult.rows; y++) {
        for (int x = 0; x < metalResult.cols; x++) {
            uchar metalValue = metalResult.at<uchar>(y, x);
            // Metal stores GrabCut values directly as discrete integers [0,1,2,3]
            mask.at<uchar>(y, x) = metalValue;
        }
    }
    
    // Ensure proper mask initialization
    if (mask.empty()) {
        mask.create(img.size(), CV_8UC1);
        mask.setTo(GC_BGD);
    }
    
    // CRITICAL FIX: Download the learned GMM parameters from Metal back to CPU
    // Create model arrays with correct size and type
    if (bgdModel.empty()) {
        bgdModel.create(1, 65, CV_32FC1); // 65 = 5 weights + 15 means + 45 covariances  
        bgdModel.setTo(Scalar::all(0));
    }
    if (fgdModel.empty()) {
        fgdModel.create(1, 65, CV_32FC1);
        fgdModel.setTo(Scalar::all(0));
    }
    
    // Download the final learned GMM parameters from Metal to CPU
    if (!d_bgdModel.empty() && !d_fgdModel.empty()) {
        d_bgdModel.download(bgdModel, stream, true);
        d_fgdModel.download(fgdModel, stream, true);
    }
}

void grabCut(InputArray img, InputOutputArray mask, Rect rect,
             InputOutputArray bgdModel, InputOutputArray fgdModel,
             int iterCount, int mode, bool useGpuGraphCut)
{
    Stream defaultStream;
    grabCut(img, mask, rect, bgdModel, fgdModel, iterCount, mode, useGpuGraphCut, defaultStream);
    defaultStream.commitAndWait();
}

void grabCutWithSharedKMeans(InputArray img, InputOutputArray mask, Rect rect,
                             InputOutputArray bgdModel, InputOutputArray fgdModel,
                             int iterCount, int mode, uint64_t randomSeed, 
                             bool useGpuGraphCut, Stream& stream)
{
    MetalMat metal_img, metal_mask, metal_bgdModel, metal_fgdModel;
    
    metal_img.upload(img.getMat());
    
    // Handle mask - initialize if needed for GC_INIT_WITH_RECT mode
    if (mode == GC_INIT_WITH_RECT) {
        mask.create(img.size(), CV_8UC1);
        mask.setTo(Scalar(GC_BGD));
    }
    metal_mask.upload(mask.getMat());
    
    // Handle models - initialize if empty (they are outputs, not inputs)
    // Note: Use CV_32FC1 instead of CV_64FC1 for Metal compatibility
    if (bgdModel.empty()) {
        bgdModel.create(1, 65, CV_32FC1); // Standard GMM model size: 13 components * 5 values each
        bgdModel.setTo(Scalar(0));
    }
    if (fgdModel.empty()) {
        fgdModel.create(1, 65, CV_32FC1);
        fgdModel.setTo(Scalar(0));
    }
    
    metal_bgdModel.upload(bgdModel.getMat());
    metal_fgdModel.upload(fgdModel.getMat());
    
    GrabCutImpl impl;
    // TODO: Pass randomSeed to run() method when Metal k-means is properly integrated
    impl.run(metal_img, metal_mask, rect, metal_bgdModel, metal_fgdModel, 
             iterCount, mode, true, stream);
    
    metal_mask.download(mask.getMatRef(), stream, true);
    metal_bgdModel.download(bgdModel.getMatRef(), stream, true);
    metal_fgdModel.download(fgdModel.getMatRef(), stream, true);
}

void grabCutWithSharedKMeans(InputArray img, InputOutputArray mask, Rect rect,
                             InputOutputArray bgdModel, InputOutputArray fgdModel,
                             int iterCount, int mode, uint64_t randomSeed, bool useGpuGraphCut)
{
    Stream defaultStream;
    grabCutWithSharedKMeans(img, mask, rect, bgdModel, fgdModel, iterCount, mode, randomSeed, useGpuGraphCut, defaultStream);
    defaultStream.commitAndWait();
}

}} // cv::metal

#endif // HAVE_METAL 