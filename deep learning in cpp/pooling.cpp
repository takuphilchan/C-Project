#include <iostream>
#include <vector>
#include <algorithm> 
#include <cmath>
using namespace std;
// Alias for a 2D matrix of floats, used for images or numerical computations.
using Matrix = vector<vector<float>>;

Matrix maxPool2D(const Matrix& input, int pool_size = 2, int stride = 2) {
    int input_size = input.size();
    int output_size = (input_size - pool_size) / stride + 1;
    // create output matrix
    Matrix output(output_size, vector<float>(output_size, 0.0));
    // performance max pooling
    for (int i = 0; i < output_size; ++i) {
        for (int j = 0; j < output_size; ++j) {
            float max_value = -INFINITY;
            for (int pi = 0; pi < pool_size; ++pi) {
                for (int pj = 0; pj < pool_size; ++pj) {
                    max_value = max(max_value, input[i * stride + pi][j * stride +pj]);
                }
            }
            output[i][j] = max_value;
        }
    }
    return output;
}

Matrix avgPool2D(const Matrix& input, int pool_size = 2, int stride = 2){
    int input_size = input.size();
    int output_size = (input_size - pool_size) / stride + 1; 
    // create output matrix
    Matrix output(output_size, vector<float>(output_size, 0.0));
    // perform average pooling
    for (int i = 0; i < output_size; ++i) {
        for (int j = 0; j < output_size; ++j) {
            float sum = 0.0;
            for (int pi = 0; pi < pool_size; ++pi) {
                for (int pj = 0; pj < pool_size; ++pj) {
                    sum +=input[i * stride + pi][j * stride + pj];
                }

            }
            output[i][j] = sum / (pool_size * pool_size);
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
    // input matrix 
    Matrix input = {
        {1, 2, 3, 4},
        {5, 6, 7, 8},
        {9, 10, 11, 12},
        {13, 14, 15, 16}
    };
    Matrix max_pooled = maxPool2D(input, 2, 2);
    cout << "Max Pooling Output: " << endl; 
    printMatrix(max_pooled); 

    Matrix avg_pooled = avgPool2D(input, 2, 2);
    cout << "Average Pooling Output:" << endl;
    printMatrix(avg_pooled);
    return 0;
}