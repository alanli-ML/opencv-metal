#include "precomp.hpp"

#ifdef HAVE_METAL

#include "opencv2/imgproc/detail/gcgraph.hpp"
#include "opencv2/imgproc/grabcut_shared.hpp"
#include "gmm_internal.hpp"
#include "graphcut_internal.hpp"
#include <memory>

// Set this to 1 to begin testing the full-GPU graph-cut solver. Until the
// MetalGraphCut implementation is ready, leave at 0 so the CPU fallback stays
// active and unit tests continue to pass.
#ifndef USE_METAL_GRAPHCUT
#define USE_METAL_GRAPHCUT 0
#endif

// (Removed unused extern enableCPUKMeansMode)

// CPU k-means initialization function that matches the CPU GrabCut implementation
void initGMMsWithKMeans(const cv::Mat& img, const cv::Mat& mask, cv::Mat& compIdxs) {
    const int kMeansItCount = 10;
    const int kMeansType = cv::KMEANS_PP_CENTERS;
    const int componentsCount = 5; // Match GMM::componentsCount
    
    cv::Mat bgdLabels, fgdLabels;
    std::vector<cv::Vec3f> bgdSamples, fgdSamples;
    
    // Collect background and foreground pixel samples (exact copy of CPU logic)
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
    
        // Perform k-means clustering on background samples (exact copy of CPU logic)
    {
        cv::Mat _bgdSamples((int)bgdSamples.size(), 3, CV_32FC1, &bgdSamples[0][0]);
        int num_clusters = componentsCount;
        num_clusters = std::min(num_clusters, (int)bgdSamples.size());
        cv::kmeans(_bgdSamples, num_clusters, bgdLabels,
                   cv::TermCriteria(cv::TermCriteria::MAX_ITER, kMeansItCount, 0.0), 0, kMeansType);
        
        // Validate K-means results to prevent bounds errors
        CV_Assert(!bgdLabels.empty() && bgdLabels.cols >= 1 && bgdLabels.type() == CV_32SC1);
    }

    // Perform k-means clustering on foreground samples (exact copy of CPU logic)  
    {
        cv::Mat _fgdSamples((int)fgdSamples.size(), 3, CV_32FC1, &fgdSamples[0][0]);
        int num_clusters = componentsCount;
        num_clusters = std::min(num_clusters, (int)fgdSamples.size());
        cv::kmeans(_fgdSamples, num_clusters, fgdLabels,
                   cv::TermCriteria(cv::TermCriteria::MAX_ITER, kMeansItCount, 0.0), 0, kMeansType);
        
        // Validate K-means results to prevent bounds errors
        CV_Assert(!fgdLabels.empty() && fgdLabels.cols >= 1 && fgdLabels.type() == CV_32SC1);
    }
    
    // Create component index map for all pixels
    compIdxs.create(img.size(), CV_8UC1); // Use 8-bit to match Metal texture format
    
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
                    bgdIdx++;
                } else {
                    compIdxs.at<uchar>(p) = 0; // Fallback to component 0
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
                    // Assign to foreground components 5-9 (add componentsCount offset)
                    compIdxs.at<uchar>(p) = (uchar)label + componentsCount;
                    fgdIdx++;
                } else {
                    compIdxs.at<uchar>(p) = componentsCount; // Fallback to component 5
                }
            }
        }
    }
}

namespace cv { namespace metal {

// Forward declarations for utility functions
void trimapFromRect(MetalMat& mask, const Rect& rect, Stream& stream);
double calcBeta(const MetalMat& image, Stream& stream);
void calcNWeights(const MetalMat& image, MetalMat& leftW, MetalMat& topleftW, MetalMat& topW, MetalMat& toprightW, 
                  double beta, double gamma, Stream& stream);
bool checkConvergence(const MetalMat& oldMask, const MetalMat& newMask, Stream& stream);

// Internal GrabCut implementation class
class GrabCutImpl {
public:
    GrabCutImpl();
    ~GrabCutImpl();
    
    void run(const MetalMat& image, MetalMat& mask, const Rect& rect,
             MetalMat& bgdModel, MetalMat& fgdModel,
             int iterCount, int mode, Stream& stream);

    void runWithSharedKMeans(const MetalMat& image, MetalMat& mask, const Rect& rect,
                            MetalMat& bgdModel, MetalMat& fgdModel,
                            int iterCount, int mode, uint64_t randomSeed, Stream& stream);

private:
    void initMaskWithRect(MetalMat& mask, Size imageSize, const Rect& rect, Stream& stream);
    void checkMask(const MetalMat& image, const MetalMat& mask);
    
    // Graph construction and max-flow (CPU implementation)
    void constructGCGraph(const MetalMat& image, const MetalMat& mask, 
                         const MetalMat& bgTerm, const MetalMat& fgTerm,
                         const MetalMat& leftW, const MetalMat& topleftW, 
                         const MetalMat& topW, const MetalMat& toprightW,
                         double lambda, cv::detail::GCGraph<double>& graph, Stream& stream);
    
    void estimateSegmentation(cv::detail::GCGraph<double>& graph, MetalMat& mask, Stream& stream);
    
    cv::Ptr<GMM> m_gmm;
    MetalMat m_pairwiseWeights[4]; // left, topleft, top, topright
    MetalMat m_components;
    MetalMat m_bgTerm, m_fgTerm;
    MetalMat m_prevMask;
    Size m_imageSize;
    double m_beta;
    bool m_initialized;

    // Placeholder for upcoming full-GPU graph-cut implementation
    std::unique_ptr<MetalGraphCut> m_metalGraphCut;
};

GrabCutImpl::GrabCutImpl() : m_beta(0.0), m_initialized(false) {
}

GrabCutImpl::~GrabCutImpl() {
    // Smart pointers and MetalMat handle cleanup automatically
}

void GrabCutImpl::run(const MetalMat& image, MetalMat& mask, const Rect& rect,
                      MetalMat& bgdModel, MetalMat& fgdModel,
                      int iterCount, int mode, Stream& stream) {
    
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
            
            // Perform CPU k-means clustering to get initial component assignments
            cv::Mat h_components;
            initGMMsWithKMeans(h_image, h_mask, h_components);
            
            // Upload the k-means component assignments to Metal
            m_components.upload(h_components);
            
            // Do initial GMM learning using K-means assignments on the input stream
            m_gmm->learnGMMs(image, mask, m_components, stream);
            
            // Synchronize only when we need to extract parameters for output
            stream.syncCPU();
            
            // Extract learned parameters even for 0 iterations
            Mat tempBgdModel, tempFgdModel;
            m_gmm->extractGMMParameters(tempBgdModel, tempFgdModel);
            bgdModel.upload(tempBgdModel);
            fgdModel.upload(tempFgdModel);
            
            m_initialized = true;
        }
    }
    
    // Calculate beta parameter and pairwise weights for all modes that need graph construction
    m_beta = calcBeta(image, stream);
    
    // Calculate pairwise weights
    const double gamma = 50.0;
    calcNWeights(image, m_pairwiseWeights[0], m_pairwiseWeights[1], 
                m_pairwiseWeights[2], m_pairwiseWeights[3], 
                m_beta, gamma, stream);
    
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
        
        // DETAILED TRACKING: Show component assignments after assignment
        
        // Sample a few component assignments
        
        // Learn GMM parameters (except in freeze model mode)  
        if (mode != GC_EVAL_FREEZE_MODEL) {
            m_gmm->learnGMMs(image, mask, m_components, stream);
            
            // Extract learned GMM parameters to output models for comparison with CPU
            Mat tempBgdModel, tempFgdModel;
            m_gmm->extractGMMParameters(tempBgdModel, tempFgdModel);
            bgdModel.upload(tempBgdModel);
            fgdModel.upload(tempFgdModel);
            
            // DETAILED TRACKING: Show GMM parameters AFTER learning
            
        }
        
        // Compute unary potentials (data term) - still on same stream
        m_gmm->computeDataTerm(image, mask, m_bgTerm, m_fgTerm, stream);
        

#if USE_METAL_GRAPHCUT
        // ---------------------------------------------------------------------------------
        // EXPERIMENTAL FULL-GPU PATH (in progress)
        // ---------------------------------------------------------------------------------
        // Keep everything on the GPU – do NOT syncCPU().
        if (!m_metalGraphCut)
        {
            m_metalGraphCut = std::make_unique<MetalGraphCut>(image.size(), stream);
        }

        // Build the graph directly on the GPU from unary & pairwise terms
        m_metalGraphCut->buildGraph(m_bgTerm, m_fgTerm,
                                    m_pairwiseWeights[0], m_pairwiseWeights[2],
                                    m_pairwiseWeights[1], m_pairwiseWeights[3],
                                    lambda);

        // For now we run a single iteration of push-relabel per GrabCut iter.
        m_metalGraphCut->solve(1);

        // Retrieve updated mask (GPU → GPU). Note: getSegmentation writes into
        // an existing MetalMat to avoid reallocations.
        m_metalGraphCut->getSegmentation(mask);

#else  // CPU FALLBACK PATH
        // ---------------------------------------------------------------------------------
        // Existing implementation: download GPU data → CPU max-flow → upload mask
        // ---------------------------------------------------------------------------------

        // PHASE 2: Single synchronization point - only when CPU needs GPU data
        // This is the ONLY place we should synchronize in the entire iteration
        stream.syncCPU();

        // PHASE 3: CPU-bound operations (these don't use streams since they're CPU-only)
        // Construct graph and solve min-cut max-flow on CPU
        cv::detail::GCGraph<double> graph;
        constructGCGraph(image, mask, m_bgTerm, m_fgTerm,
                        m_pairwiseWeights[0], m_pairwiseWeights[1],
                        m_pairwiseWeights[2], m_pairwiseWeights[3],
                        lambda, graph, stream);
        
        // Estimate segmentation using max-flow
        estimateSegmentation(graph, mask, stream);
#endif
        
        // Check for convergence
        if (checkConvergence(m_prevMask, mask, stream)) {
            break;
        }
    }
}

void GrabCutImpl::runWithSharedKMeans(const MetalMat& image, MetalMat& mask, const Rect& rect,
                                     MetalMat& bgdModel, MetalMat& fgdModel,
                                     int iterCount, int mode, uint64_t randomSeed, Stream& stream) {
    
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
        } else {
            checkMask(image, mask);
        }
        
        // Initialize GMMs only if not already initialized
        if (!m_initialized) {
            // Download CPU versions for shared K-means clustering and initial learning
            cv::Mat h_image, h_mask;
            image.download(h_image, stream, true);
            mask.download(h_mask, stream, true);
            
            // CRITICAL: Convert to BGR for shared K-means compatibility with CPU
            cv::Mat h_image_bgr;
            if (h_image.type() == CV_8UC4) {
                cv::cvtColor(h_image, h_image_bgr, cv::COLOR_BGRA2BGR);
            } else {
                h_image_bgr = h_image;
            }
            
            // CRITICAL: Fix MetalMat original channels tracking for BGRA images
            // The image parameter has original_channels_=3, but we need it as 4 for GMM learning
            const_cast<MetalMat&>(image).setOriginalChannels(4);
            
            // Use shared deterministic K-means with fixed seed
            setRNGSeed(randomSeed); // Reset RNG to exact same state as CPU
            cv::SharedKMeansResult sharedKMeans = cv::performDeterministicKMeans(h_image_bgr, h_mask, randomSeed);
            
            // Create component assignments for Metal format (8UC1)
            cv::Mat h_components;
            cv::createComponentIndexMap(h_image_bgr, h_mask, sharedKMeans, h_components, 0, 5); // Use explicit 0 instead of CV_8UC1
            
            // Upload the shared K-means component assignments to Metal
            m_components.upload(h_components);
            
            // CRITICAL DEBUG: Verify component assignments were uploaded correctly  
            
            // Sample a few uploaded component values
            
            // Do initial GMM learning using shared K-means assignments on the input stream
            m_gmm->learnGMMs(image, mask, m_components, stream);
            
            // Synchronize only when we need to extract parameters for output
            stream.syncCPU();
            
            // CRITICAL FIX: Extract learned parameters even for 0 iterations (like regular path)
            Mat tempBgdModel, tempFgdModel;
            m_gmm->extractGMMParameters(tempBgdModel, tempFgdModel);
            bgdModel.upload(tempBgdModel);
            fgdModel.upload(tempFgdModel);
            
            m_initialized = true;
        }
    }
    
    // Calculate beta parameter and pairwise weights for all modes that need graph construction
    m_beta = calcBeta(image, stream);
    
    // Calculate pairwise weights
    const double gamma = 50.0;
    calcNWeights(image, m_pairwiseWeights[0], m_pairwiseWeights[1], 
                m_pairwiseWeights[2], m_pairwiseWeights[3], 
                m_beta, gamma, stream);
    
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
    
    for (int iter = 0; iter < iterCount; iter++) {
        
        // DETAILED TRACKING: Extract and compare GMM parameters BEFORE this iteration
        
        // Sample a few GMM parameters to track divergence
        
        // Save current mask for convergence check
        m_prevMask = mask.clone();

        // PHASE 1: Batch all GPU operations on the input stream (NO intermediate commits)
        // This allows GPU operations to pipeline and overlap for maximum performance
        
        // Ensure components texture exists for assignment
        if (m_components.empty()) {
            m_components.create(mask.size(), CV_8UC1);
        }
        
        m_gmm->assignGMMs(image, mask, m_components, stream);
        
        // DETAILED TRACKING: Show component assignments after assignment
        
        // Learn GMM parameters (except in freeze model mode)  
        if (mode != GC_EVAL_FREEZE_MODEL) {
            m_gmm->learnGMMs(image, mask, m_components, stream);
            stream.syncCPU(); // Ensure GPU learning completes before reading buffers

            // Extract learned GMM parameters to output models for comparison with CPU
            // Note: This is asynchronous - parameters will be available after stream commits
            Mat tempBgdModel, tempFgdModel;
            m_gmm->extractGMMParameters(tempBgdModel, tempFgdModel);
            bgdModel.upload(tempBgdModel);
            fgdModel.upload(tempFgdModel);
        }
        
        // Compute unary potentials (data term) - still on same stream
        m_gmm->computeDataTerm(image, mask, m_bgTerm, m_fgTerm, stream);
        
        // PHASE 2: Single synchronization point - only when CPU needs GPU data
        // This is the ONLY place we should synchronize in the entire iteration
        stream.syncCPU();
        
        // PHASE 3: CPU-bound operations (these don't use streams since they're CPU-only)
        // Construct graph and solve min-cut max-flow on CPU
        cv::detail::GCGraph<double> graph;
        constructGCGraph(image, mask, m_bgTerm, m_fgTerm,
                        m_pairwiseWeights[0], m_pairwiseWeights[1],
                        m_pairwiseWeights[2], m_pairwiseWeights[3],
                        lambda, graph, stream);
        
        // Estimate segmentation using max-flow
        estimateSegmentation(graph, mask, stream);
        
        // Check for convergence
        if (checkConvergence(m_prevMask, mask, stream)) {
            break;
        }
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
    int probableDebugCount = 0;
    int totalProbablePixels = 0;
    int totalSurePixels = 0;
    for (int y = 0; y < image.rows(); y++) {
        for (int x = 0; x < image.cols(); x++) {
            int vtxIdx = graph.addVtx();
            uchar maskValue = h_mask.at<uchar>(y, x);
            
            if (maskValue == GC_PR_BGD || maskValue == GC_PR_FGD) {
                totalProbablePixels++;
            } else {
                totalSurePixels++;
            }
            
            // Set terminal weights (unary potentials)
            double fromSource, toSink;
            if (maskValue == GC_PR_BGD || maskValue == GC_PR_FGD) {
                fromSource = h_bgTerm.at<float>(y, x);      // Cost to connect to SOURCE (background model cost, like CPU)
                toSink = h_fgTerm.at<float>(y, x);          // Cost to connect to SINK (foreground model cost, like CPU)
                // Debug: sample some terminal weights
                if (probableDebugCount < 5) {
                    probableDebugCount++;
                }
            } else if (maskValue == GC_BGD) {
                fromSource = 0;
                toSink = lambda;
            } else { // GC_FGD
                fromSource = lambda;
                toSink = 0;
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
    
    if (probableDebugCount == 0) {
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
    // Metal stores: 0.0, 0.33, 0.67, 1.0 → CV_8UC1: 0, 84, 171, 255
    // Need to convert to: 0, 1, 2, 3 (GrabCut mask values)
    int valueCount[256] = {0};
    int grabcutCounts[4] = {0}; // Count GrabCut values after conversion
    for (int y = 0; y < h_currentMask.rows; y++) {
        for (int x = 0; x < h_currentMask.cols; x++) {
            uchar metalValue = h_currentMask.at<uchar>(y, x);
            valueCount[metalValue]++;
            // Metal stores GrabCut values directly as discrete integers [0,1,2,3]
            uchar grabcutValue = metalValue;
            if (grabcutValue <= 3) { // Valid GrabCut values
                grabcutCounts[grabcutValue]++;
                h_currentMask.at<uchar>(y, x) = grabcutValue;
            } else {
                // Invalid value, default to background
                grabcutCounts[0]++;
                h_currentMask.at<uchar>(y, x) = 0;
            }
        }
    }
    
    // Debug: show what values we actually downloaded from Metal
    
    // Update mask based on graph cut results
    int fgDecisions = 0, bgDecisions = 0, totalProbable = 0;
    int checkedPixels = 0;
    for (int y = 0; y < mask.rows(); y++) {
        for (int x = 0; x < mask.cols(); x++) {
            uchar currentValue = h_currentMask.at<uchar>(y, x);
            if (currentValue == GC_PR_BGD || currentValue == GC_PR_FGD) {
                totalProbable++;
                int vtxIdx = y * mask.cols() + x;
                if (graph.inSourceSegment(vtxIdx)) {
                    h_mask.at<uchar>(y, x) = GC_PR_FGD;
                    fgDecisions++;
                } else {
                    h_mask.at<uchar>(y, x) = GC_PR_BGD;
                    bgDecisions++;
                }
                checkedPixels++;
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
             int iterCount, int mode, Stream& stream) {
    
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
    grabCutImpl.run(d_img, d_mask, rect, d_bgdModel, d_fgdModel, iterCount, mode, stream);
    
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
             int iterCount, int mode)
{
    Stream defaultStream;
    grabCut(img, mask, rect, bgdModel, fgdModel, iterCount, mode, defaultStream);
    defaultStream.commitAndWait();
}

void grabCutWithSharedKMeans(InputArray img, InputOutputArray mask, Rect rect,
                             InputOutputArray bgdModel, InputOutputArray fgdModel,
                             int iterCount, int mode, uint64_t randomSeed, Stream& stream)
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
    impl.runWithSharedKMeans(metal_img, metal_mask, rect, metal_bgdModel, metal_fgdModel, 
                            iterCount, mode, randomSeed, stream);
    
    metal_mask.download(mask.getMatRef(), stream, true);
    metal_bgdModel.download(bgdModel.getMatRef(), stream, true);
    metal_fgdModel.download(fgdModel.getMatRef(), stream, true);
}

void grabCutWithSharedKMeans(InputArray img, InputOutputArray mask, Rect rect,
                             InputOutputArray bgdModel, InputOutputArray fgdModel,
                             int iterCount, int mode, uint64_t randomSeed)
{
    Stream defaultStream;
    grabCutWithSharedKMeans(img, mask, rect, bgdModel, fgdModel, iterCount, mode, randomSeed, defaultStream);
    defaultStream.commitAndWait();
}

}} // cv::metal

#endif // HAVE_METAL 