#include <opencv2/opencv.hpp>
#include <iostream>

using namespace std;

int main() {
    // Specify the image path correctly
    string imagePath = "D:\\Deep Learning/deep learning in cpp\\new_image.jpg";
    
    // Load the image
    cv::Mat image = cv::imread(imagePath);
    cv::Size newSize(400, 400);
    cv::Mat resizedImage;

    cv::resize(image, resizedImage, newSize);
    
    // Check if the image was loaded successfully
    if (image.empty()) {
        std::cout << "Could not open or find the image at: " << imagePath << std::endl;
        return -1; // Return with error code
    }
     
    // Display the image
    cv::imshow("Display Window", resizedImage);

   //Display gray image  
    cv::Mat grayImage;
    cv::cvtColor(resizedImage, grayImage, cv::COLOR_BGR2GRAY);
    cv::imshow("Grayscale Image", grayImage);

    cv::Mat canvas = cv::Mat::zeros(400, 400, CV_8UC3);
    cv::circle(canvas, cv::Point(200, 200), 100, cv::Scalar(255, 0, 0), -1);

    cv::rectangle(canvas, cv::Point(50, 50), cv::Point(350, 350), cv::Scalar(0, 255, 0), 3);
    cv::line(canvas, cv::Point(0, 0), cv::Point(400, 400), cv::Scalar(0, 0, 255), 2);

    cv::imshow("Canvas", canvas);

    cv::Mat blurredImage;
    cv::GaussianBlur(resizedImage, blurredImage, cv::Size(15, 15), 0);
    cv::imshow("Gausian blurr", blurredImage);

    cv::Mat edges;
    cv::Canny(resizedImage, edges, 100, 200);
    cv::imshow("Edge detector", edges);

    // Wait indefinitely until a key is pressed
    cv::waitKey(1000000);
    
    // Destroy all created windows
    cv::destroyAllWindows();

    return 0;
}

