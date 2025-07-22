#include <iostream>

#include "opencv2/core.hpp"
#include "opencv2/core/metal.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/imgproc/metal.hpp"
#include "opencv2/highgui.hpp"

using namespace std;
using namespace cv;

void help()
{
    cout << "Demonstrates the use of the Metal backend for a chained image processing pipeline." << endl;
    cout << "Usage:" << endl;
    cout << "  ./sample_metal_morphology <image_path>" << endl;
    cout << "The sample performs resize, Gaussian blur, and Sobel filtering on the GPU." << endl;
}

int main(int argc, char** argv)
{
    if (argc != 2)
    {
        help();
        return -1;
    }

    Mat src_host = imread(argv[1], IMREAD_COLOR);
    if (src_host.empty())
    {
        cerr << "Can't open image " << argv[1] << endl;
        return -1;
    }

    // Metal backend requires BGRA format for 4-channel images, and doesn't support 3-channel.
    // So we convert BGR to BGRA.
    Mat bgra_host;
    cvtColor(src_host, bgra_host, COLOR_BGR2BGRA);

    // Also, for Sobel, we need a 32-bit float grayscale image.
    Mat gray_host;
    cvtColor(src_host, gray_host, COLOR_BGR2GRAY);
    Mat gray_float_host;
    gray_host.convertTo(gray_float_host, CV_32F, 1.0/255.0);


    cout << "Demonstrating chained operations without a stream (synchronous)..." << endl;
    int64 t_sync_start = getTickCount();

    // Upload image to GPU
    metal::MetalMat src_device(gray_float_host);
    metal::MetalMat resized_device, blurred_device, sobel_device_sync;

    // Execute operations one by one. Each call blocks until the GPU operation is complete.
    metal::resize(src_device, resized_device, Size(), 0.5, 0.5, INTER_LINEAR);
    metal::GaussianBlur(resized_device, blurred_device, Size(5, 5), 1.5);
    metal::Sobel(blurred_device, sobel_device_sync, -1, 1, 0, 3);

    int64 t_sync_end = getTickCount();
    double sync_time = (t_sync_end - t_sync_start) * 1000.0 / getTickFrequency();
    cout << "Synchronous execution time: " << sync_time << " ms" << endl;

    Mat sobel_host_sync;
    sobel_device_sync.download(sobel_host_sync);
    sobel_host_sync.convertTo(sobel_host_sync, CV_8U, 255.0);
    imshow("Sobel Result (Synchronous)", sobel_host_sync);


    cout << "\nDemonstrating chained operations with a stream (asynchronous)..." << endl;
    int64 t_async_start = getTickCount();

    // Create a stream
    metal::Stream stream;

    // The source is already on the device from the previous step.
    metal::MetalMat sobel_device_async;

    // Chain operations on the stream. These calls are non-blocking.
    metal::resize(src_device, resized_device, Size(), 0.5, 0.5, INTER_LINEAR, stream);
    metal::GaussianBlur(resized_device, blurred_device, Size(5, 5), 1.5, stream);
    metal::Sobel(blurred_device, sobel_device_async, -1, 1, 0, 3, stream);

    // Commit the command buffer and wait for the GPU to finish all enqueued operations.
    stream.commitAndWait();

    int64 t_async_end = getTickCount();
    double async_time = (t_async_end - t_async_start) * 1000.0 / getTickFrequency();
    cout << "Asynchronous (stream) execution time: " << async_time << " ms" << endl;

    Mat sobel_host_async;
    sobel_device_async.download(sobel_host_async);
    sobel_host_async.convertTo(sobel_host_async, CV_8U, 255.0);
    imshow("Sobel Result (Asynchronous)", sobel_host_async);


    cout << "\nPerformance difference (Sync vs. Stream): " << sync_time / async_time << "x" << endl;
    cout << "Press any key to exit." << endl;
    waitKey(0);

    return 0;
}