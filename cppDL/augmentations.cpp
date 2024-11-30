#include <opencv2/opencv.hpp>
#include <iostream>

using namespace cv;
using namespace std;

int main() {
    // load image
    Mat image = imread("images\\elephant.png");
    if (image.empty()) {
        cout << "Could not read the image" << endl;
        return -1;
    }
    // resize images
    Size targetSize(250, 220);
    Mat resizedImage;
    resize(image, resizedImage, targetSize);

    // random horizontal flip
    Mat flippedImage;
    flip(resizedImage, flippedImage, 1);

    // random rotation
    Point center(resizedImage.cols / 2, resizedImage.rows / 2);
    double angle = (rand() % 30) - 15;
    Mat rotationMatrix = getRotationMatrix2D(center, angle, 1.0);
    Mat rotatedImage;
    warpAffine(resizedImage, rotatedImage, rotationMatrix, resizedImage.size());

    // random scaling
    double scale = 1 + (rand() % 50) / 100.0;
    Mat scaledImage;
    resize(resizedImage, scaledImage, Size(), scale, scale);

    // random translation
    int maxShift = 50; // max shift of 50 pixels
    int shiftX = rand() % (2 * maxShift) - maxShift;
    int shiftY = rand() % (2 * maxShift) - maxShift;
    Mat translationMatrix = (Mat_<double>(2,3) << 1, 0, shiftX, 0, 1, shiftY);
    Mat translatedImage;
    warpAffine(resizedImage, translatedImage, translationMatrix, resizedImage.size());

    // display the combined image
    imshow("original", resizedImage);
    imshow("flipped image", flippedImage);
    imshow("rotated image", rotatedImage);
    imshow("scaled image", scaledImage);
    imshow("translated image", translatedImage);
    // Wait for a key press and check if it is a valid key
    int key = waitKey(0); // Wait indefinitely for a key press
    // If any key is pressed, destroy all OpenCV windows
    if (key >= 0) {
        destroyAllWindows(); // Close all open windows
    }
    return 0;
}


