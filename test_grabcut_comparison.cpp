#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/metalimgproc.hpp>
#include <iostream>
#include <chrono>

using namespace cv;
using namespace std;

// Convert mask to visualization (background=black, probable_bg=dark_gray, probable_fg=light_gray, foreground=white)
Mat maskToVisualization(const Mat& mask) {
    Mat vis(mask.size(), CV_8UC3);
    
    for (int y = 0; y < mask.rows; y++) {
        for (int x = 0; x < mask.cols; x++) {
            uchar val = mask.at<uchar>(y, x);
            Vec3b color;
            
            switch(val) {
                case GC_BGD:     color = Vec3b(0, 0, 0);       break; // Black (certain background)
                case GC_FGD:     color = Vec3b(255, 255, 255); break; // White (certain foreground)  
                case GC_PR_BGD:  color = Vec3b(64, 64, 64);    break; // Dark gray (probable background)
                case GC_PR_FGD:  color = Vec3b(192, 192, 192); break; // Light gray (probable foreground)
                default:         color = Vec3b(128, 0, 128);   break; // Purple (invalid)
            }
            
            vis.at<Vec3b>(y, x) = color;
        }
    }
    
    return vis;
}

// Create foreground extraction (foreground pixels from original image, background=black)
Mat extractForeground(const Mat& image, const Mat& mask) {
    Mat result;
    image.copyTo(result);
    
    for (int y = 0; y < mask.rows; y++) {
        for (int x = 0; x < mask.cols; x++) {
            uchar val = mask.at<uchar>(y, x);
            if (val == GC_BGD || val == GC_PR_BGD) {
                // Background pixel - set to black
                result.at<Vec3b>(y, x) = Vec3b(0, 0, 0);
            }
        }
    }
    
    return result;
}

int main(int argc, char** argv) {
    cout << "GrabCut Comparison Test - CPU vs Metal" << endl;
    cout << "=======================================" << endl;
    
    // Load test image
    string imagePath = "WID-small.jpg";
    Mat image = imread(imagePath, IMREAD_COLOR);
    if (image.empty()) {
        cerr << "Error: Cannot load image " << imagePath << endl;
        return -1;
    }
    
    cout << "Loaded image: " << image.cols << "x" << image.rows << " pixels" << endl;
    
    // Define rectangle around central subject
    Rect rect(image.cols/4, image.rows/4, image.cols/2, image.rows/2);
    cout << "Using rectangle: " << rect << endl;
    
    const int iterations = 5;
    
    // ========================
    // CPU GRABCUT
    // ========================
    cout << "\nRunning CPU GrabCut..." << endl;
    Mat mask_cpu(image.size(), CV_8UC1, Scalar(GC_BGD));
    Mat bgdModel_cpu, fgdModel_cpu;
    
    auto start_cpu = chrono::high_resolution_clock::now();
    grabCut(image, mask_cpu, rect, bgdModel_cpu, fgdModel_cpu, iterations, GC_INIT_WITH_RECT);
    auto end_cpu = chrono::high_resolution_clock::now();
    auto cpu_time = chrono::duration_cast<chrono::milliseconds>(end_cpu - start_cpu);
    
    cout << "CPU GrabCut completed in " << cpu_time.count() << "ms" << endl;
    
    // ========================  
    // METAL GRABCUT
    // ========================
    cout << "\nRunning Metal GrabCut..." << endl;
    Mat mask_metal(image.size(), CV_8UC1, Scalar(GC_BGD));
    Mat bgdModel_metal, fgdModel_metal;
    
    auto start_metal = chrono::high_resolution_clock::now();
    cv::metal::grabCutWithSharedKMeans(image, mask_metal, rect, bgdModel_metal, fgdModel_metal, 
                                      iterations, GC_INIT_WITH_RECT, 42);
    auto end_metal = chrono::high_resolution_clock::now();
    auto metal_time = chrono::duration_cast<chrono::milliseconds>(end_metal - start_metal);
    
    cout << "Metal GrabCut completed in " << metal_time.count() << "ms" << endl;
    cout << "Speedup: " << (double)cpu_time.count() / metal_time.count() << "x" << endl;
    
    // ========================
    // ANALYSIS
    // ========================
    cout << "\nAnalyzing results..." << endl;
    
    // Count mask values
    vector<int> cpu_counts(4, 0), metal_counts(4, 0);
    int agreement = 0, total = 0;
    
    for (int y = 0; y < image.rows; y++) {
        for (int x = 0; x < image.cols; x++) {
            uchar cpu_val = mask_cpu.at<uchar>(y, x);
            uchar metal_val = mask_metal.at<uchar>(y, x);
            
            if (cpu_val <= 3) cpu_counts[cpu_val]++;
            if (metal_val <= 3) metal_counts[metal_val]++;
            
            // Check if both classify as foreground or both as background
            bool cpu_fg = (cpu_val == GC_FGD || cpu_val == GC_PR_FGD);
            bool metal_fg = (metal_val == GC_FGD || metal_val == GC_PR_FGD);
            if (cpu_fg == metal_fg) agreement++;
            total++;
        }
    }
    
    cout << "\nMask distribution comparison:" << endl;
    cout << "CPU:   BGD=" << cpu_counts[0] << " FGD=" << cpu_counts[1] 
         << " PR_BGD=" << cpu_counts[2] << " PR_FGD=" << cpu_counts[3] << endl;
    cout << "Metal: BGD=" << metal_counts[0] << " FGD=" << metal_counts[1] 
         << " PR_BGD=" << metal_counts[2] << " PR_FGD=" << metal_counts[3] << endl;
    cout << "Foreground/Background agreement: " << (double)agreement/total*100.0 << "%" << endl;
    
    // ========================
    // SAVE OUTPUT IMAGES
    // ========================
    cout << "\nSaving output images..." << endl;
    
    // Save original image with rectangle overlay
    Mat image_with_rect;
    image.copyTo(image_with_rect);
    rectangle(image_with_rect, rect, Scalar(0, 255, 0), 3);
    imwrite("original_with_rect.jpg", image_with_rect);
    cout << "✓ Saved: original_with_rect.jpg" << endl;
    
    // Save mask visualizations
    Mat cpu_mask_vis = maskToVisualization(mask_cpu);
    Mat metal_mask_vis = maskToVisualization(mask_metal);
    imwrite("cpu_mask.jpg", cpu_mask_vis);
    imwrite("metal_mask.jpg", metal_mask_vis);
    cout << "✓ Saved: cpu_mask.jpg" << endl;
    cout << "✓ Saved: metal_mask.jpg" << endl;
    
    // Save foreground extractions
    Mat cpu_foreground = extractForeground(image, mask_cpu);
    Mat metal_foreground = extractForeground(image, mask_metal);
    imwrite("cpu_foreground.jpg", cpu_foreground);
    imwrite("metal_foreground.jpg", metal_foreground);
    cout << "✓ Saved: cpu_foreground.jpg" << endl;
    cout << "✓ Saved: metal_foreground.jpg" << endl;
    
    // Create side-by-side comparison images
    cout << "\nCreating comparison images..." << endl;
    
    // Mask comparison
    Mat mask_comparison;
    hconcat(cpu_mask_vis, metal_mask_vis, mask_comparison);
    
    // Add labels
    putText(mask_comparison, "CPU", Point(cpu_mask_vis.cols/2 - 30, 30), 
            FONT_HERSHEY_SIMPLEX, 1, Scalar(255, 255, 0), 2);
    putText(mask_comparison, "Metal", Point(cpu_mask_vis.cols + metal_mask_vis.cols/2 - 50, 30), 
            FONT_HERSHEY_SIMPLEX, 1, Scalar(255, 255, 0), 2);
    
    imwrite("mask_comparison.jpg", mask_comparison);
    cout << "✓ Saved: mask_comparison.jpg" << endl;
    
    // Foreground comparison
    Mat foreground_comparison;
    hconcat(cpu_foreground, metal_foreground, foreground_comparison);
    
    putText(foreground_comparison, "CPU", Point(cpu_foreground.cols/2 - 30, 30), 
            FONT_HERSHEY_SIMPLEX, 1, Scalar(255, 255, 0), 2);
    putText(foreground_comparison, "Metal", Point(cpu_foreground.cols + metal_foreground.cols/2 - 50, 30), 
            FONT_HERSHEY_SIMPLEX, 1, Scalar(255, 255, 0), 2);
    
    imwrite("foreground_comparison.jpg", foreground_comparison);
    cout << "✓ Saved: foreground_comparison.jpg" << endl;
    
    // Create difference map
    Mat diff_map(mask_cpu.size(), CV_8UC3, Scalar(0, 0, 0)); // Start with black
    for (int y = 0; y < mask_cpu.rows; y++) {
        for (int x = 0; x < mask_cpu.cols; x++) {
            uchar cpu_val = mask_cpu.at<uchar>(y, x);
            uchar metal_val = mask_metal.at<uchar>(y, x);
            
            if (cpu_val != metal_val) {
                // Different classification - mark in red
                diff_map.at<Vec3b>(y, x) = Vec3b(0, 0, 255); // Red for differences
            } else {
                // Same classification - mark in green
                diff_map.at<Vec3b>(y, x) = Vec3b(0, 128, 0);  // Dark green for agreement
            }
        }
    }
    imwrite("difference_map.jpg", diff_map);
    cout << "✓ Saved: difference_map.jpg (red=different, green=same)" << endl;
    
    cout << "\n=======================================" << endl;
    cout << "Comparison complete! Output files:" << endl;
    cout << "• original_with_rect.jpg - Input image with selection rectangle" << endl;
    cout << "• cpu_mask.jpg / metal_mask.jpg - Individual mask visualizations" << endl;
    cout << "• cpu_foreground.jpg / metal_foreground.jpg - Extracted foregrounds" << endl;
    cout << "• mask_comparison.jpg - Side-by-side mask comparison" << endl;
    cout << "• foreground_comparison.jpg - Side-by-side foreground comparison" << endl;
    cout << "• difference_map.jpg - Pixel-level difference visualization" << endl;
    cout << "=======================================" << endl;
    
    return 0;
} 