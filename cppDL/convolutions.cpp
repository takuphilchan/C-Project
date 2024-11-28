#include <iostream>
#include <vector>
#include <cmath>
using namespace std;
using Matrix = vector<vector<float>>;
Matrix conv2D(const Matrix& input, const Matrix& kernel, int stride = 1, int padding = 0) {
    int input_size = input.size();
    int kernel_size = kernel.size();
    int output_size = (input_size - kernel_size + 2 * padding) / stride + 1;
    Matrix output(output_size, vector<float>(output_size, 0.0));
    // apply padding to input
    Matrix padded_input = input; 
    if (padding > 0) {
        int padded_size = input_size + 2 * padding;
        padded_input = Matrix(padded_size, vector<float>(padded_size, 0.0));
        for (int i = 0; i < input_size; ++i) {
            for (int j = 0; j < input_size; ++j) {
                padded_input[i + padding][j + padding] = input[i][j];
            }
        }
    }
    // 2d convolution operation
    for (int i = 0; i < output_size; ++i) {
        for (int j = 0; j < output_size; ++j) {
            float sum = 0.0; 
            for (int ki = 0; ki < kernel_size; ++ki) {
                for (int kj = 0; kj < kernel_size; ++kj) {
                    sum += kernel[ki][kj] * padded_input[i * stride + ki][j * stride + kj];                
                }
            } 
            output[i][j] = sum;
        } 
    }
    return output;
}
void printMatrix(const Matrix& mat) {
    for (const auto& row : mat) {
        for (const auto& value : row) cout << value << " "; cout << endl;
    }
}
int main() { 
    Matrix input = {
        {1, 2, 3, 4, 5},
        {6, 7, 8, 9, 10},
        {11, 12, 13, 14, 15},
        {16, 17, 18, 19, 20},
        {21, 22, 23, 24, 25}
    };
    Matrix kernel = {
        {1, 0, -1},
        {1, 0, -1},
        {1, 0, -1}
    };
    // Perform 2D convolution
    Matrix result = conv2D(input, kernel, 1, 1);
    cout << "2D Convolution Result:" << endl;
    printMatrix(result);
    return 0; 
}


