#include <opencv2/opencv.hpp>
#include <iostream>

int main() {
    cv::Mat image = cv::imread("D:\\Deep Learning/deep learning in cpp\\lion.png");
    if (image.empty()) {
        std::cerr << "Could not open or find the image!" << std::endl;
        return -1;
    }

    // Preprocessing: Resize and denoise
    cv::resize(image, image, cv::Size(800, 600));
    cv::Mat denoised;
    cv::bilateralFilter(image, denoised, 9, 75, 75);

    // Define ROI
    cv::Rect roi(50, 50, image.cols - 100, image.rows - 100);

    // GrabCut Initialization
    cv::Mat mask = cv::Mat::zeros(image.size(), CV_8UC1);
    cv::Mat bgModel, fgModel;
    mask.setTo(cv::GC_BGD);
    cv::grabCut(denoised, mask, roi, bgModel, fgModel, 5, cv::GC_INIT_WITH_RECT);

    // Refine Mask with Color Segmentation
    cv::Mat hsv, colorMask;
    cv::cvtColor(denoised, hsv, cv::COLOR_BGR2HSV);
    cv::inRange(hsv, cv::Scalar(10, 50, 50), cv::Scalar(30, 255, 255), colorMask);

    // Combine Masks
    cv::compare(mask, cv::GC_PR_FGD, mask, cv::CMP_EQ);
    cv::bitwise_and(mask, colorMask, mask);

    // Postprocess Mask
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5));
    cv::morphologyEx(mask, mask, cv::MORPH_CLOSE, kernel);

    // Apply Mask
    cv::Mat result;
    image.copyTo(result, mask);

    // Display Results
    cv::imshow("Original Image", image);
    cv::imshow("Segmented Lion", result);
    // Wait for a key press and check if it is a valid key
    int key = cv::waitKey(0); // Wait indefinitely for a key press
    // If any key is pressed, destroy all OpenCV windows
    if (key >= 0) {
        cv::destroyAllWindows(); // Close all open windows
    }
    return 0;
}
