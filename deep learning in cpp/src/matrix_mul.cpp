#include <pybind11/pybind11.h>
#include <vector>

using namespace std;

vector<vector<float>> matmul(const vector<vector<float>>& A, const vector<vector<float>>& B) {
    // get dimensions
    int rowsA = A.size();
    int colsA = A[0].size();
    int rowsB = B.size();
    int colsB = B[0].size();

    // ensure the matrix dimension match for multiplication
    if (colsA != rowsB) {
        throw invalid_argument("Matrix dimensions must match for multiplication");
    }

    // initialize result matrix with zeros
    vector<vector<float>> C(rowsA, vector<float>(colsB, 0));

    // perform matrix multiplication
    for (int i = 0; i < rowsA; ++i) {
        for (int j = 0; j < colsB; ++j) {
            for (int k = 0; k < colsA; ++k) {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }
    return C;
}

// expose the matmul function to python 
PYBIND11_MODULE(matrix_mul, m) {
    m.def("matmul", &matmul, "Perform matrix multiplication"); 
}