#include <opencv2/opencv.hpp>
#include <iostream>

using namespace std;

int main() {
    // Specify the image file path
    string imagePath = "D:\\Deep Learning/deep learning in cpp\\lion.png";
    // Load the image from the specified path
    cv::Mat image = cv::imread(imagePath);
    // Check if the image was successfully loaded
    if (image.empty()) {
        cout << "Error: Could not open or find the image at: " << imagePath << endl;
        return -1; // Return error code if image is not found
    }
    // Resize the image to a fixed size of 400x400 pixels
    cv::Size newSize(200, 200);
    cv::Mat resizedImage;
    cv::resize(image, resizedImage, newSize);
    // Display the resized image in a window
    cv::imshow("Resized Image", resizedImage);
    // Convert the resized image to grayscale
    cv::Mat grayImage;
    cv::cvtColor(resizedImage, grayImage, cv::COLOR_BGR2GRAY);
    // Display the grayscale image
    cv::imshow("Grayscale Image", grayImage);
    // Apply Gaussian blur to the resized image
    cv::Mat blurredImage;
    cv::GaussianBlur(resizedImage, blurredImage, cv::Size(15, 15), 0);
    // Display the blurred image
    cv::imshow("Gaussian Blurred Image", blurredImage);
    // Detect edges using the Canny edge detector
    cv::Mat edges;
    cv::Canny(resizedImage, edges, 100, 200);
    // Display the edge-detected image
    cv::imshow("Edge Detected Image", edges);
    // Wait for a key press and check if it is a valid key
    int key = cv::waitKey(0); // Wait indefinitely for a key press

    // If any key is pressed, destroy all OpenCV windows
    if (key >= 0) {
        cv::destroyAllWindows(); // Close all open windows
    }
    return 0;
}
