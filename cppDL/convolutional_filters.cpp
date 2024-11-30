#include <opencv2/opencv.hpp>
#include <iostream>
using namespace std;
using namespace cv;

Mat applyConvolution(const Mat &image, const Mat &filter) {
    int filterHeight = filter.rows;
    int filterWidth = filter.cols;
    int imageHeight = image.rows;
    int imageWidth = image.cols;
    // output image after applying convolution
    Mat output(image.size(), image.type());
    // apply convolution
    for (int i = 0; i < imageHeight - filterHeight + 1; i++) {
        for (int j = 0; j < imageWidth - filterWidth + 1; j++) {
            double sum = 0.0;
            // iterate over the filter matrix
            for (int k = 0; k < filterHeight; k++) {
                for (int l = 0; l < filterWidth; l++) {
                    sum += image.at<uchar>(i + k, j + l) * filter.at<double>(k, l);
                }
            }
            // assign the computed sum to the output image
            output.at<uchar>(i + filterHeight / 2, j + filterWidth / 2) = static_cast<uchar>(sum);
        }
    }
    return output;
}
int main() {
    // Load the image as a color image (3 channels)
    cv::Mat image = cv::imread("images\\bird.png", IMREAD_COLOR);
    if (image.empty()) {
        cout << "Error loading image!" << endl;
        return -1;
    }
    // Resize the image to 400x400
    resize(image, image, Size(400, 400));
    // Define Sobel edge detection filters
    Mat sobelX = (Mat_<double>(3, 3) <<
                   1, 0, -1,
                   2, 0, -2,
                   1, 0, -1);
    Mat sobelY = (Mat_<double>(3, 3) <<
                   1, 2, 1,
                   0, 0, 0,
                   -1, -2, -1);
    // Convert the image to grayscale before applying Sobel for edge detection
    Mat grayImage;
    cvtColor(image, grayImage, COLOR_BGR2GRAY);

    // Apply convolution (Sobel filters) on the grayscale image
    Mat edgeX = applyConvolution(grayImage, sobelX);  // Sobel X (Horizontal)
    Mat edgeY = applyConvolution(grayImage, sobelY);  // Sobel Y (Vertical)

    // Combine Sobel X and Y results to get edge-detected image
    Mat edgeDetected = edgeX + edgeY;
    // Normalize the result for better visibility
    normalize(edgeDetected, edgeDetected, 0, 255, NORM_MINMAX);
    // **Scaling Factor** to control intensity of the edge detection result
    double scaleFactor = 0.2; // Change this value to increase/decrease the intensity
    // Apply scaling to adjust the intensity of the edges
    edgeDetected = edgeDetected * scaleFactor;
    // Normalize again to make sure values are between 0 and 255
    normalize(edgeDetected, edgeDetected, 0, 255, NORM_MINMAX);
    // Display the original and edge-detected images
    imshow("Original Image", image);
    imshow("Edge Detected Image (Adjusted Intensity)", edgeDetected);
    // Wait for a key press and check if it is a valid key
    int key = cv::waitKey(0); // Wait indefinitely for a key press
    // If any key is pressed, destroy all OpenCV windows
    if (key >= 0) {
        cv::destroyAllWindows(); // Close all open windows
    }
    return 0;
}