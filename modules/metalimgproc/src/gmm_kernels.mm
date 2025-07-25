#include "precomp.hpp"

#ifdef HAVE_METAL

#include "gmm_internal.hpp"

namespace cv { namespace metal {

// MSL kernel source for GMM operations
static const char* gmmKernelsSource = R"(
#include <metal_stdlib>
using namespace metal;

// GMM constants
constant int GMM_COMPONENTS_PER_CLASS = 5;
constant int GMM_TOTAL_COMPONENTS = 10; // 5 for background, 5 for foreground
constant float EPSILON = 1e-3f;

// GMM data structure (matches CUDA implementation)
struct GMMComponent {
    float weight;
    float mean_r;
    float mean_g;
    float mean_b;
    // 3x3 covariance inverse matrix (symmetric, upper triangular stored)  
    float cov_inv_00, cov_inv_01, cov_inv_02;
    float              cov_inv_11, cov_inv_12;
    float                          cov_inv_22;
    float inv_sqrt_det; // This is 1/sqrt(det), weight multiplied in kernel like CPU
};

// Debug structure to capture probability calculations
struct DebugInfo {
    float pixel_r, pixel_g, pixel_b;
    float prob[5];  // Probabilities for each component
    float mahal_dist[5];  // Mahalanobis distances for each component
    int best_component;
    float best_prob;
    // Add GMM parameter debugging for component 1
    float comp1_weight;
    float comp1_mean_r, comp1_mean_g, comp1_mean_b;
    float comp1_cov_inv_00, comp1_cov_inv_01, comp1_cov_inv_02;
    float comp1_cov_inv_11, comp1_cov_inv_12, comp1_cov_inv_22;
    float comp1_inv_sqrt_det;
    // Add detailed calculation steps for component 1
    float comp1_pixel_minus_mean_r, comp1_pixel_minus_mean_g, comp1_pixel_minus_mean_b;
    float comp1_xxa, comp1_yyd, comp1_zzf;
    float comp1_yxb, comp1_zxc, comp1_zye;
    float comp1_mahal_before_clamp;
    float comp1_exp_term;
};

// Initialize GMM from mask kernel
kernel void gmmInitializeKernel(texture2d<float, access::sample> image [[texture(0)]],
                               texture2d<uint, access::read> mask [[texture(1)]],
                               device GMMComponent* gmm_bg [[buffer(0)]],
                               device GMMComponent* gmm_fg [[buffer(1)]],
                               device atomic<float>* bg_counts [[buffer(2)]],
                               device atomic<float>* fg_counts [[buffer(3)]],
                               device float3* bg_sums [[buffer(4)]],
                               device float3* fg_sums [[buffer(5)]],
                               uint2 gid [[thread_position_in_grid]])
{
    if (gid.x >= image.get_width() || gid.y >= image.get_height()) return;
    
    uint mask_value = mask.read(gid).x;
    
    if (mask_value == 0 || mask_value == 2) { // Background (sure or probable)
        for (int i = 0; i < GMM_COMPONENTS_PER_CLASS; i++) {
            atomic_fetch_add_explicit(&bg_counts[i], 1.0f / GMM_COMPONENTS_PER_CLASS, memory_order_relaxed);
            // Note: This is a simplified initialization. Full implementation would use better clustering.
        }
    } else if (mask_value == 1 || mask_value == 3) { // Foreground (sure or probable)
        for (int i = 0; i < GMM_COMPONENTS_PER_CLASS; i++) {
            atomic_fetch_add_explicit(&fg_counts[i], 1.0f / GMM_COMPONENTS_PER_CLASS, memory_order_relaxed);
        }
    }
}

// Assign pixels to GMM components kernel with stable fallback
kernel void gmmAssignKernel(texture2d<float, access::sample> image [[texture(0)]],
                           texture2d<uint, access::read> mask [[texture(1)]],
                           texture2d<uint, access::write> components [[texture(2)]],
                           constant GMMComponent* gmm_bg [[buffer(0)]],
                           constant GMMComponent* gmm_fg [[buffer(1)]],
                           device DebugInfo* debug_buffer [[buffer(2)]],
                           uint2 gid [[thread_position_in_grid]])
{
    constexpr sampler s(coord::pixel, address::clamp_to_edge, filter::nearest);
    
    if (gid.x >= image.get_width() || gid.y >= image.get_height()) return;
    
    float4 pixel = image.sample(s, float2(gid) + 0.5f);
    uint mask_value = mask.read(gid).x; // Read uint directly
    
    int best_component = 0;
    float best_prob = 0.0f; // Use raw probability like CPU, not log probability
    bool use_spatial_fallback = false;
    
    // CRITICAL FIX: Match CPU implementation exactly
    // CPU assigns components 0-4 for BOTH background and foreground
    // The mask value determines which GMM to use, not the component index
    if (mask_value == 0 || mask_value == 2) { // Background
        // Calculate raw probabilities for all components (match CPU exactly)
        for (int i = 0; i < GMM_COMPONENTS_PER_CLASS; i++) {
            float prob = 0.0f;
            // CPU: Always calculates probability regardless of weight, only weight=0 gives prob=0
            if (gmm_bg[i].weight > 0.0f) {
                // CRITICAL FIX: Convert pixel from [0,1] to [0,255] range to match GMM means
                // CRITICAL FIX 2024-07-24: Use BGR order to match GMM training data
                float3 pixel_255 = float3(pixel.b, pixel.g, pixel.r) * 255.0f;
                float3 v = pixel_255 - float3(gmm_bg[i].mean_r, gmm_bg[i].mean_g, gmm_bg[i].mean_b);
                // Full Mahalanobis distance with complete covariance matrix
                float xxa = v.x * v.x * gmm_bg[i].cov_inv_00;
                float yyd = v.y * v.y * gmm_bg[i].cov_inv_11;
                float zzf = v.z * v.z * gmm_bg[i].cov_inv_22;
                float yxb = v.x * v.y * gmm_bg[i].cov_inv_01;
                float zxc = v.z * v.x * gmm_bg[i].cov_inv_02;  // FIXED: Now using cov_inv_02
                float zye = v.z * v.y * gmm_bg[i].cov_inv_12;  // FIXED: Now using cov_inv_12
                
                float mahal_dist = xxa + yyd + zzf + 2.0f * (yxb + zxc + zye);
                mahal_dist = min(mahal_dist, 50.0f); // Clamp to prevent numerical issues
                
                // CRITICAL FIX: Metal should match CPU's whichComponent() which uses UNWEIGHTED probability
                // CPU's whichComponent() calls operator()(ci, color) which does NOT multiply by weight
                prob = gmm_bg[i].inv_sqrt_det * exp(-0.5f * mahal_dist);
            }
            
            // Simple natural selection: find maximum probability (like CPU)
            if (prob > best_prob) {
                best_prob = prob;
                best_component = i; // CRITICAL FIX: Background component index 0-4 (like CPU)
            }
        }
        
        // Only use spatial fallback if ALL components have zero probability (completely degenerate)
        if (best_prob <= 0.0f) {
            use_spatial_fallback = true;
        }
    } else { // Foreground
        // Calculate raw probabilities for all components (match CPU exactly)
        for (int i = 0; i < GMM_COMPONENTS_PER_CLASS; i++) {
            float prob = 0.0f;
            // CPU: Always calculates probability regardless of weight, only weight=0 gives prob=0
            if (gmm_fg[i].weight > 0.0f) {
                // CRITICAL FIX: Convert pixel from [0,1] to [0,255] range to match GMM means
                // CRITICAL FIX 2024-07-24: Use BGR order to match GMM training data
                float3 pixel_255 = float3(pixel.b, pixel.g, pixel.r) * 255.0f;
                float3 v = pixel_255 - float3(gmm_fg[i].mean_r, gmm_fg[i].mean_g, gmm_fg[i].mean_b);
                // Full Mahalanobis distance with complete covariance matrix
                float xxa = v.x * v.x * gmm_fg[i].cov_inv_00;
                float yyd = v.y * v.y * gmm_fg[i].cov_inv_11; 
                float zzf = v.z * v.z * gmm_fg[i].cov_inv_22;
                float yxb = v.x * v.y * gmm_fg[i].cov_inv_01;
                float zxc = v.z * v.x * gmm_fg[i].cov_inv_02;
                float zye = v.z * v.y * gmm_fg[i].cov_inv_12;
                
                float mahal_dist = xxa + yyd + zzf + 2.0f * (yxb + zxc + zye);
                mahal_dist = min(mahal_dist, 50.0f); // Clamp to prevent numerical issues
                
                // CPU formula: res = 1.0f/sqrt(covDeterms[ci]) * exp(-0.5f*mult);
                // CRITICAL FIX: Metal should match CPU's whichComponent() which uses UNWEIGHTED probability  
                // CPU's whichComponent() calls operator()(ci, color) which does NOT multiply by weight
                prob = gmm_fg[i].inv_sqrt_det * exp(-0.5f * mahal_dist);
            }
            
            // Simple natural selection: find maximum probability (like CPU)
            if (prob > best_prob) {
                best_prob = prob;
                best_component = i; // CRITICAL FIX: Foreground component index 0-4 (like CPU)
            }
        }
        
        // Only use spatial fallback if ALL components have zero probability (completely degenerate)
        if (best_prob <= 0.0f) {
            use_spatial_fallback = true;
        }
    }
    
    // Apply spatial fallback if needed for stable assignment
    if (use_spatial_fallback) {
        // Use improved spatial pattern for better distribution across all components
        // Combine position-based and pixel-intensity-based assignment for diversity
        int spatial_x = (gid.x / 32) % 5; // 32x32 block pattern in X
        int spatial_y = (gid.y / 32) % 5; // 32x32 block pattern in Y
        int intensity_based = int((pixel.r + pixel.g + pixel.b) * 255.0f * 5.0f) % 5; // Intensity-based component
        
        // Combine spatial and intensity for better distribution
        int spatial_component = (spatial_x + spatial_y + intensity_based) % 5;
        
        // CRITICAL FIX: Both background and foreground use 0-4 component range (like CPU)
        best_component = spatial_component;
    }
    
    // CRITICAL FIX: Write component assignment as 0-4 for both BG and FG (like CPU)
    // The mask value determines which GMM to accumulate to, not the component index
    // CRITICAL FIX: Write only the component value, not uint4 for R8Uint texture
    components.write(uint(best_component), gid);
    
    // DEBUG: Sample component assignments to verify they're in range 0-4
    if ((gid.x == 500 && gid.y >= 373 && gid.y <= 382) || (gid.x >= 13 && gid.x <= 20 && gid.y == 0)) {
        // For debugging specific pixels - do nothing here to avoid Metal print limitations
        // Debug output will be checked via CPU download
    }
}

// Compute data term (unary potentials) kernel
kernel void gmmDataTermKernel(texture2d<float, access::sample> image [[texture(0)]],
                             texture2d<float, access::read> mask [[texture(1)]],
                             texture2d<float, access::write> bg_term [[texture(2)]],
                             texture2d<float, access::write> fg_term [[texture(3)]],
                             constant GMMComponent* gmm_bg [[buffer(0)]],
                             constant GMMComponent* gmm_fg [[buffer(1)]],
                             uint2 gid [[thread_position_in_grid]])
{
    constexpr sampler s(coord::pixel, address::clamp_to_edge, filter::nearest);
    
    if (gid.x >= image.get_width() || gid.y >= image.get_height()) return;
    
    float4 pixel = image.sample(s, float2(gid) + 0.5f);
    
    // Convert to [0,255] range like CUDA
    // CRITICAL FIX 2024-07-24: Use BGR order to match GMM training data
    float3 color = float3(pixel.b, pixel.g, pixel.r) * 255.0f;
    
    // Calculate background probability (following CPU implementation exactly)
    float data_bg = 0.0f;
    for (int i = 0; i < GMM_COMPONENTS_PER_CLASS; i++) {
        if (gmm_bg[i].weight > 0) {
            float3 v = color - float3(gmm_bg[i].mean_r, gmm_bg[i].mean_g, gmm_bg[i].mean_b);
            
            float xxa = v.x * v.x * gmm_bg[i].cov_inv_00;
            float yyd = v.y * v.y * gmm_bg[i].cov_inv_11; 
            float zzf = v.z * v.z * gmm_bg[i].cov_inv_22;
            float yxb = v.x * v.y * gmm_bg[i].cov_inv_01;
            float zxc = v.z * v.x * gmm_bg[i].cov_inv_02;
            float zye = v.z * v.y * gmm_bg[i].cov_inv_12;
            
            float mahal_dist = xxa + yyd + zzf + 2.0f * (yxb + zxc + zye);
            // CPU: res += coefs[ci] * (1.0f/sqrt(covDeterms[ci]) * exp(-0.5f*mult))
            data_bg += gmm_bg[i].weight * gmm_bg[i].inv_sqrt_det * exp(-0.5f * mahal_dist);
        }
    }
    
    // Calculate foreground probability  
    float data_fg = 0.0f;
    for (int i = 0; i < GMM_COMPONENTS_PER_CLASS; i++) {
        if (gmm_fg[i].weight > 0) {
            float3 v = color - float3(gmm_fg[i].mean_r, gmm_fg[i].mean_g, gmm_fg[i].mean_b);
            
            float xxa = v.x * v.x * gmm_fg[i].cov_inv_00;
            float yyd = v.y * v.y * gmm_fg[i].cov_inv_11; 
            float zzf = v.z * v.z * gmm_fg[i].cov_inv_22;
            float yxb = v.x * v.y * gmm_fg[i].cov_inv_01;
            float zxc = v.z * v.x * gmm_fg[i].cov_inv_02;
            float zye = v.z * v.y * gmm_fg[i].cov_inv_12;
            
            float mahal_dist = xxa + yyd + zzf + 2.0f * (yxb + zxc + zye);
            data_fg += gmm_fg[i].weight * gmm_fg[i].inv_sqrt_det * exp(-0.5f * mahal_dist);
        }
    }
    
    // CPU: fromSource = -log(bgdGMM(color)); toSink = -log(fgdGMM(color));
    // Prevent complete collapse by providing reasonable fallback probabilities
    // when components have collapsed (weight=0)
    
    float bg_penalty, fg_penalty;
    
    if (data_bg > 0.0f) {
        bg_penalty = -log(data_bg);
    } else {
        // Fallback: Use balanced probability to prevent extreme bias
        // When GMM components collapse, provide reasonable fallback guidance
        float bg_fallback_prob = 0.5f; // Neutral probability
        bg_penalty = -log(bg_fallback_prob);
    }
    
    if (data_fg > 0.0f) {
        fg_penalty = -log(data_fg);
    } else {
        // Fallback: Use balanced probability to prevent extreme bias  
        // When GMM components collapse, provide reasonable fallback guidance
        float fg_fallback_prob = 0.5f; // Neutral probability
        fg_penalty = -log(fg_fallback_prob);
    }
     
    bg_term.write(bg_penalty, gid);
    fg_term.write(fg_penalty, gid);
}

// Reduction kernel for GMM learning (simplified version)
kernel void gmmReductionKernel(texture2d<float, access::sample> image [[texture(0)]],
                              texture2d<uint, access::read> components [[texture(1)]], // FIXED: uint texture for integer component IDs
                              texture2d<uint, access::read> mask [[texture(2)]],
                              device atomic<float>* counts [[buffer(0)]],
                              device atomic<float>* mean_sums_r [[buffer(1)]],
                              device atomic<float>* mean_sums_g [[buffer(2)]],
                              device atomic<float>* mean_sums_b [[buffer(3)]],
                              constant int& component_id [[buffer(4)]],
                              uint2 gid [[thread_position_in_grid]])
{
    constexpr sampler s(coord::pixel, address::clamp_to_edge, filter::nearest);
    
    if (gid.x >= image.get_width() || gid.y >= image.get_height()) return;
    
    uint assigned_component = components.read(gid).x; // Read uint directly - no conversion needed
    uint mask_value = mask.read(gid).x;
    
    if (int(assigned_component) == component_id) { // Fix sign comparison warning
        float4 pixel = image.sample(s, float2(gid) + 0.5f);
        // CRITICAL FIX 2024-07-24: Use BGR order to match GMM training data
        float3 color = float3(pixel.b, pixel.g, pixel.r) * 255.0f;
        
        atomic_fetch_add_explicit(&counts[0], 1.0f, memory_order_relaxed);
        atomic_fetch_add_explicit(&mean_sums_r[0], color.r, memory_order_relaxed);
        atomic_fetch_add_explicit(&mean_sums_g[0], color.g, memory_order_relaxed);
        atomic_fetch_add_explicit(&mean_sums_b[0], color.b, memory_order_relaxed);
    }
}

// Count pixels per component from accumulated stats  
kernel void gmmCountPixelsKernel(device float* bgStats [[buffer(0)]],
                                 device float* fgStats [[buffer(1)]],
                                 device uint* pixelCounts [[buffer(2)]],
                                 uint tid [[thread_position_in_grid]])
{
    if (tid == 0) { // Single thread to perform the reduction
        // Calculate total background and foreground pixel counts
        uint totalBgPixels = 0;
        uint totalFgPixels = 0;
        
        for (int i = 0; i < 5; ++i) {
            totalBgPixels += (uint)bgStats[i * 10];     // Sum BG component counts
            totalFgPixels += (uint)fgStats[i * 10];     // Sum FG component counts
        }
        
        // Store totals that gmmFinalizeParametersKernel expects
        pixelCounts[0] = totalBgPixels;  // Total background pixels
        pixelCounts[1] = totalFgPixels;  // Total foreground pixels
    }
}

// Comprehensive GMM statistics accumulation kernel
kernel void gmmAccumulateStatsKernel(texture2d<float, access::read> image [[texture(0)]],
                                    texture2d<uint, access::read> components [[texture(1)]],
                                    texture2d<uint, access::read> mask [[texture(2)]],
                                    device float* bgStats [[buffer(0)]],
                                    device float* fgStats [[buffer(1)]],
                                    uint2 gid [[thread_position_in_grid]])
{
    if (gid.x >= image.get_width() || gid.y >= image.get_height()) {
        return;
    }
    
    float4 pixelColor = image.read(gid);
    // CRITICAL FIX: Extract individual BGR channels to match CPU implementation
    
    uint componentId = components.read(gid).x;
    uint maskValue = mask.read(gid).x;
    
    // CRITICAL FIX: Match CPU implementation exactly
    // Component ID is always 0-4 for both background and foreground
    // Use mask value to determine which GMM to accumulate to
    int comp = int(componentId);
    
    // Clamp to valid range [0, 4] (like CPU)
    comp = min(max(comp, 0), 4);
    
    // CRITICAL FIX: Determine if pixel belongs to background or foreground using mask
    // This matches the CPU logic: mask.at<uchar>(p) == GC_BGD || mask.at<uchar>(p) == GC_PR_BGD
    bool isBackground = (maskValue == 0 || maskValue == 2); // 0=GC_BGD, 2=GC_PR_BGD
    
    // CRITICAL FIX: Components are always 0-4, mask determines which GMM to use
    if (comp < 0 || comp >= 5) return; // All components must be 0-4
    
    // Accumulate statistics for appropriate GMM component based on mask value
    device float* targetStats = isBackground ? bgStats : fgStats;
    int offset = comp * 10; // Each component has 10 statistics
    
    // Use atomic operations to accumulate safely across threads
    // CRITICAL FIX: Store in BGR order to match CPU exactly
    // CPU stores B,G,R in positions 1,2,3 so GPU should do the same
    float blue = pixelColor.b * 255.0f;
    float green = pixelColor.g * 255.0f;  
    float red = pixelColor.r * 255.0f;
    
    // Statistics format: [count, sum_b, sum_g, sum_r, cov_bb, cov_bg, cov_br, cov_gg, cov_gr, cov_rr]
    atomic_fetch_add_explicit((device atomic<float>*)&targetStats[offset + 0], 1.0f, memory_order_relaxed); // count
    atomic_fetch_add_explicit((device atomic<float>*)&targetStats[offset + 1], blue, memory_order_relaxed); // sum_b (BGR order)
    atomic_fetch_add_explicit((device atomic<float>*)&targetStats[offset + 2], green, memory_order_relaxed); // sum_g
    atomic_fetch_add_explicit((device atomic<float>*)&targetStats[offset + 3], red, memory_order_relaxed); // sum_r
    atomic_fetch_add_explicit((device atomic<float>*)&targetStats[offset + 4], blue * blue, memory_order_relaxed); // sum_bb
    atomic_fetch_add_explicit((device atomic<float>*)&targetStats[offset + 5], blue * green, memory_order_relaxed); // sum_bg
    atomic_fetch_add_explicit((device atomic<float>*)&targetStats[offset + 6], blue * red, memory_order_relaxed); // sum_br
    atomic_fetch_add_explicit((device atomic<float>*)&targetStats[offset + 7], green * green, memory_order_relaxed); // sum_gg
    atomic_fetch_add_explicit((device atomic<float>*)&targetStats[offset + 8], green * red, memory_order_relaxed); // sum_gr
    atomic_fetch_add_explicit((device atomic<float>*)&targetStats[offset + 9], red * red, memory_order_relaxed); // sum_rr
}

// Finalize GMM parameters from accumulated statistics
kernel void gmmFinalizeParametersKernel(device float* bgStats [[buffer(0)]],
                                       device float* fgStats [[buffer(1)]],
                                       device float* bgGmm [[buffer(2)]],
                                       device float* fgGmm [[buffer(3)]],
                                       device uint32_t* pixelCounts [[buffer(4)]],
                                       constant uint& componentsCount [[buffer(5)]],
                                       constant float& totalBgComponentSamples [[buffer(6)]],
                                       constant float& totalFgComponentSamples [[buffer(7)]],
                                       uint gid [[thread_position_in_grid]])
{
    int comp = int(gid);
    
    // Get pixel counts (kept for backwards compatibility, but not used for weight calculation)
    uint32_t totalBgPixels = pixelCounts[0];
    uint32_t totalFgPixels = pixelCounts[1];
    
    // CRITICAL FIX: Use pre-calculated component sample totals passed from host
    // This eliminates any potential race conditions or calculation errors inside the kernel
    
    // CRITICAL FIX: Add safeguards against component degeneration
    const float MIN_COMPONENT_WEIGHT = 1e-6f;  // Minimum weight to prevent complete degeneration
    const float MIN_VARIANCE = 1.0f;           // Minimum variance for diagonal elements
    const float REGULARIZATION = 0.01f;        // Regularization amount when needed
    
    // Process background component
    {
        float count = bgStats[comp * 10 + 0];
        
        // CRITICAL FIX: Safeguard against zero-count components
        if (count > 0 && totalBgComponentSamples > 0) {
            // Calculate mean (BGR order: B=1, G=2, R=3)
            float mean_b = bgStats[comp * 10 + 1] / count;
            float mean_g = bgStats[comp * 10 + 2] / count;
            float mean_r = bgStats[comp * 10 + 3] / count;
            
            // Calculate covariance matrix elements (BGR order: BB=4, BG=5, BR=6, GG=7, GR=8, RR=9)
            float cov_bb = bgStats[comp * 10 + 4] / count - mean_b * mean_b;
            float cov_bg = bgStats[comp * 10 + 5] / count - mean_b * mean_g;
            float cov_br = bgStats[comp * 10 + 6] / count - mean_b * mean_r;
            float cov_gg = bgStats[comp * 10 + 7] / count - mean_g * mean_g;
            float cov_gr = bgStats[comp * 10 + 8] / count - mean_g * mean_r;
            float cov_rr = bgStats[comp * 10 + 9] / count - mean_r * mean_r;
            
            // Match CPU logic: do not clamp variances upfront.  We rely on the
            // conditional regularisation block below (executed when det ≤ 1e-6)
            // to stabilise near-singular matrices, exactly like
            // GMM::calcInverseCovAndDeterm() on the CPU path.
            
            // Calculate determinant (BGR order)
            float det = cov_rr * (cov_gg * cov_bb - cov_bg * cov_bg) -
                       cov_gr * (cov_gr * cov_bb - cov_bg * cov_br) +
                       cov_br * (cov_gr * cov_bg - cov_gg * cov_br);
            
            // CRITICAL FIX: Match CPU conditional regularization exactly
            if (det <= 1e-6f) {
                // Add regularization only when needed, like CPU
                cov_rr += REGULARIZATION;
                cov_gg += REGULARIZATION;
                cov_bb += REGULARIZATION;
                
                // Recalculate determinant after regularization
                det = cov_rr * (cov_gg * cov_bb - cov_bg * cov_bg) -
                     cov_gr * (cov_gr * cov_bb - cov_bg * cov_br) +
                     cov_br * (cov_gr * cov_bg - cov_gg * cov_br);
            }
            
            // Calculate inverse covariance matrix (BGR order)
            float invDet = 1.0f / det;
            float invCov_bb = (cov_gg * cov_rr - cov_gr * cov_gr) * invDet;
            float invCov_bg = -(cov_bg * cov_rr - cov_br * cov_gr) * invDet; 
            float invCov_br = (cov_bg * cov_gr - cov_gg * cov_br) * invDet;
            float invCov_gg = (cov_bb * cov_rr - cov_br * cov_br) * invDet;
            float invCov_gr = -(cov_bb * cov_gr - cov_bg * cov_br) * invDet;
            float invCov_rr = (cov_bb * cov_gg - cov_bg * cov_bg) * invDet;
            
            // CRITICAL FIX: Use pre-calculated component sample total as denominator like CPU
            // CPU: coefs[ci] = (double)n/totalSampleCount (where totalSampleCount = sum of all component counts)
            float weight = max(count / totalBgComponentSamples, MIN_COMPONENT_WEIGHT);  // SAFEGUARD: Ensure minimum weight
            float invSqrtDet = 1.0f / sqrt(det);
            
            bgGmm[comp * 11 + 0] = weight;
            bgGmm[comp * 11 + 1] = mean_b;  // Store as BGR order
            bgGmm[comp * 11 + 2] = mean_g;
            bgGmm[comp * 11 + 3] = mean_r;
            bgGmm[comp * 11 + 4] = invCov_bb; // Store covariance as BGR order
            bgGmm[comp * 11 + 5] = invCov_bg;
            bgGmm[comp * 11 + 6] = invCov_br;
            bgGmm[comp * 11 + 7] = invCov_gg;
            bgGmm[comp * 11 + 8] = invCov_gr;
            bgGmm[comp * 11 + 9] = invCov_rr;
            bgGmm[comp * 11 + 10] = invSqrtDet;
        } else {
            // CRITICAL FIX: Provide fallback parameters for degenerate components
            // Use neutral mean and identity covariance to keep component alive
            float fallback_weight = MIN_COMPONENT_WEIGHT;
            float fallback_mean = 128.0f; // Neutral gray value
            float fallback_var = 100.0f;  // Reasonable variance
            float fallback_inv_cov = 1.0f / fallback_var;
            float fallback_inv_sqrt_det = 1.0f / sqrt(fallback_var * fallback_var * fallback_var);
            
            bgGmm[comp * 11 + 0] = fallback_weight;
            bgGmm[comp * 11 + 1] = fallback_mean;  // B
            bgGmm[comp * 11 + 2] = fallback_mean;  // G
            bgGmm[comp * 11 + 3] = fallback_mean;  // R
            bgGmm[comp * 11 + 4] = fallback_inv_cov; // BB
            bgGmm[comp * 11 + 5] = 0.0f;            // BG (off-diagonal)
            bgGmm[comp * 11 + 6] = 0.0f;            // BR (off-diagonal)
            bgGmm[comp * 11 + 7] = fallback_inv_cov; // GG
            bgGmm[comp * 11 + 8] = 0.0f;            // GR (off-diagonal)
            bgGmm[comp * 11 + 9] = fallback_inv_cov; // RR
            bgGmm[comp * 11 + 10] = fallback_inv_sqrt_det;
        }
    }
    
    // Process foreground component with same safeguards
    {
        float count = fgStats[comp * 10 + 0];
        
        // CRITICAL FIX: Safeguard against zero-count components
        if (count > 0 && totalFgComponentSamples > 0) {
            // Calculate mean (BGR order: B=1, G=2, R=3)
            float mean_b = fgStats[comp * 10 + 1] / count;
            float mean_g = fgStats[comp * 10 + 2] / count;
            float mean_r = fgStats[comp * 10 + 3] / count;
            
            // Calculate covariance matrix elements (BGR order: BB=4, BG=5, BR=6, GG=7, GR=8, RR=9)
            float cov_bb = fgStats[comp * 10 + 4] / count - mean_b * mean_b;
            float cov_bg = fgStats[comp * 10 + 5] / count - mean_b * mean_g;
            float cov_br = fgStats[comp * 10 + 6] / count - mean_b * mean_r;
            float cov_gg = fgStats[comp * 10 + 7] / count - mean_g * mean_g;
            float cov_gr = fgStats[comp * 10 + 8] / count - mean_g * mean_r;
            float cov_rr = fgStats[comp * 10 + 9] / count - mean_r * mean_r;
            
            // Match CPU logic: no unconditional variance clamping – we only add
            // REGULARIZATION later when the determinant test fails, mirroring the
            // behaviour of the CPU implementation.
            
            // Calculate determinant (BGR order)
            float det = cov_rr * (cov_gg * cov_bb - cov_bg * cov_bg) -
                       cov_gr * (cov_gr * cov_bb - cov_bg * cov_br) +
                       cov_br * (cov_gr * cov_bg - cov_gg * cov_br);
            
            // CRITICAL FIX: Match CPU conditional regularization exactly
            if (det <= 1e-6f) {
                // Add regularization only when needed, like CPU
                cov_rr += REGULARIZATION;
                cov_gg += REGULARIZATION;
                cov_bb += REGULARIZATION;
                
                // Recalculate determinant after regularization
                det = cov_rr * (cov_gg * cov_bb - cov_bg * cov_bg) -
                     cov_gr * (cov_gr * cov_bb - cov_bg * cov_br) +
                     cov_br * (cov_gr * cov_bg - cov_gg * cov_br);
            }
            
            // Calculate inverse covariance matrix (BGR order)
            float invDet = 1.0f / det;
            float invCov_bb = (cov_gg * cov_rr - cov_gr * cov_gr) * invDet;
            float invCov_bg = -(cov_bg * cov_rr - cov_br * cov_gr) * invDet;
            float invCov_br = (cov_bg * cov_gr - cov_gg * cov_br) * invDet;
            float invCov_gg = (cov_bb * cov_rr - cov_br * cov_br) * invDet;
            float invCov_gr = -(cov_bb * cov_gr - cov_bg * cov_br) * invDet;
            float invCov_rr = (cov_bb * cov_gg - cov_bg * cov_bg) * invDet;
            
            // CRITICAL FIX: Use pre-calculated component sample total as denominator like CPU
            float weight = max(count / totalFgComponentSamples, MIN_COMPONENT_WEIGHT);  // SAFEGUARD: Ensure minimum weight
            float invSqrtDet = 1.0f / sqrt(det);
            
            fgGmm[comp * 11 + 0] = weight;
            fgGmm[comp * 11 + 1] = mean_b;  // Store as BGR order
            fgGmm[comp * 11 + 2] = mean_g;
            fgGmm[comp * 11 + 3] = mean_r;
            fgGmm[comp * 11 + 4] = invCov_bb; // Store covariance as BGR order
            fgGmm[comp * 11 + 5] = invCov_bg;
            fgGmm[comp * 11 + 6] = invCov_br;
            fgGmm[comp * 11 + 7] = invCov_gg;
            fgGmm[comp * 11 + 8] = invCov_gr;
            fgGmm[comp * 11 + 9] = invCov_rr;
            fgGmm[comp * 11 + 10] = invSqrtDet;
        } else {
            // CRITICAL FIX: Provide fallback parameters for degenerate foreground components
            // Use neutral mean and identity covariance to keep component alive
            float fallback_weight = MIN_COMPONENT_WEIGHT;
            float fallback_mean = 64.0f;  // Darker mean for foreground
            float fallback_var = 100.0f;
            float fallback_inv_cov = 1.0f / fallback_var;
            float fallback_inv_sqrt_det = 1.0f / sqrt(fallback_var * fallback_var * fallback_var);
            
            fgGmm[comp * 11 + 0] = fallback_weight;
            fgGmm[comp * 11 + 1] = fallback_mean;  // B
            fgGmm[comp * 11 + 2] = fallback_mean;  // G
            fgGmm[comp * 11 + 3] = fallback_mean;  // R
            fgGmm[comp * 11 + 4] = fallback_inv_cov; // BB
            fgGmm[comp * 11 + 5] = 0.0f;            // BG (off-diagonal)
            fgGmm[comp * 11 + 6] = 0.0f;            // BR (off-diagonal)
            fgGmm[comp * 11 + 7] = fallback_inv_cov; // GG
            fgGmm[comp * 11 + 8] = 0.0f;            // GR (off-diagonal)
            fgGmm[comp * 11 + 9] = fallback_inv_cov; // RR
            fgGmm[comp * 11 + 10] = fallback_inv_sqrt_det;
        }
    }
}

)";

// Static pipeline state cache
static id<MTLComputePipelineState> g_gmmInitializePipeline = nil;
static id<MTLComputePipelineState> g_gmmAssignPipeline = nil;
static id<MTLComputePipelineState> g_gmmDataTermPipeline = nil;
static id<MTLComputePipelineState> g_gmmReductionPipeline = nil;
static id<MTLComputePipelineState> g_gmmAccumulateStatsPipeline = nil;
static id<MTLComputePipelineState> g_gmmCountPixelsPipeline = nil;
static id<MTLComputePipelineState> g_gmmFinalizeParametersPipeline = nil;
static dispatch_once_t g_gmmPipelinesOnce = 0;

// Initialize GMM pipelines
void initializeGMMPipelines() {
    dispatch_once(&g_gmmPipelinesOnce, ^{
        @autoreleasepool {
            NSError *error = nil;
            MetalContext& ctx = MetalContext::getInstance();
            id<MTLDevice> device = ctx.device;
            
            NSString *kernelSource = [NSString stringWithUTF8String:gmmKernelsSource];
            id<MTLLibrary> library = [device newLibraryWithSource:kernelSource options:nil error:&error];
            
            if (error) {
                NSLog(@"Error compiling GMM kernels: %@", error.localizedDescription);
                return;
            }
            
            // Create compute pipeline states
            id<MTLFunction> initFunc = [library newFunctionWithName:@"gmmInitializeKernel"];
            g_gmmInitializePipeline = [device newComputePipelineStateWithFunction:initFunc error:&error];
            
            id<MTLFunction> assignFunc = [library newFunctionWithName:@"gmmAssignKernel"];
            g_gmmAssignPipeline = [device newComputePipelineStateWithFunction:assignFunc error:&error];
            
            id<MTLFunction> dataTermFunc = [library newFunctionWithName:@"gmmDataTermKernel"];
            g_gmmDataTermPipeline = [device newComputePipelineStateWithFunction:dataTermFunc error:&error];
            
            id<MTLFunction> reductionFunc = [library newFunctionWithName:@"gmmReductionKernel"];
            g_gmmReductionPipeline = [device newComputePipelineStateWithFunction:reductionFunc error:&error];
            
            // Create new GMM learning pipelines
            id<MTLFunction> accumulateStatsFunc = [library newFunctionWithName:@"gmmAccumulateStatsKernel"];
            g_gmmAccumulateStatsPipeline = [device newComputePipelineStateWithFunction:accumulateStatsFunc error:&error];
            
            id<MTLFunction> countPixelsFunc = [library newFunctionWithName:@"gmmCountPixelsKernel"];
            g_gmmCountPixelsPipeline = [device newComputePipelineStateWithFunction:countPixelsFunc error:&error];
            
            id<MTLFunction> finalizeParamsFunc = [library newFunctionWithName:@"gmmFinalizeParametersKernel"];
            g_gmmFinalizeParametersPipeline = [device newComputePipelineStateWithFunction:finalizeParamsFunc error:&error];
        }
    });
}

// Pipeline accessors
id<MTLComputePipelineState> getGMMInitializePipeline() { return g_gmmInitializePipeline; }
id<MTLComputePipelineState> getGMMAssignPipeline() { return g_gmmAssignPipeline; }
id<MTLComputePipelineState> getGMMDataTermPipeline() { return g_gmmDataTermPipeline; }
id<MTLComputePipelineState> getGMMReductionPipeline() { return g_gmmReductionPipeline; }
id<MTLComputePipelineState> getGMMAccumulateStatsPipeline() { return g_gmmAccumulateStatsPipeline; }
id<MTLComputePipelineState> getGMMCountPixelsPipeline() { return g_gmmCountPixelsPipeline; }
id<MTLComputePipelineState> getGMMFinalizeParametersPipeline() { return g_gmmFinalizeParametersPipeline; }

}} // cv::metal

#endif // HAVE_METAL 