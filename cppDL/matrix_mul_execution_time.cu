#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <chrono>
#include <cmath>

using namespace std;
using namespace chrono;

// CUDA kernel for matrix multiplication using shared memory
__global__ void matrixMultiplySharedMemory(float* matrixA, float* matrixB, float* matrixC, int matrixSize) {
    extern __shared__ float sharedMemory[];

    float* subMatrixA = sharedMemory; // Shared memory for sub-matrix of A
    float* subMatrixB = &sharedMemory[blockDim.x * blockDim.y]; // Shared memory for sub-matrix of B

    int threadX = threadIdx.x;
    int threadY = threadIdx.y;
    int row = blockIdx.y * blockDim.y + threadY;
    int col = blockIdx.x * blockDim.x + threadX;

    float cellValue = 0.0f;

    // Loop over tiles of the input matrices
    for (int tile = 0; tile < (matrixSize + blockDim.x - 1) / blockDim.x; ++tile) {
        if (row < matrixSize && tile * blockDim.x + threadX < matrixSize)
            subMatrixA[threadY * blockDim.x + threadX] = matrixA[row * matrixSize + tile * blockDim.x + threadX];
        else
            subMatrixA[threadY * blockDim.x + threadX] = 0.0f;

        if (col < matrixSize && tile * blockDim.y + threadY < matrixSize)
            subMatrixB[threadY * blockDim.x + threadX] = matrixB[(tile * blockDim.y + threadY) * matrixSize + col];
        else
            subMatrixB[threadY * blockDim.x + threadX] = 0.0f;

        __syncthreads();

        for (int k = 0; k < blockDim.x; ++k) {
            cellValue += subMatrixA[threadY * blockDim.x + k] * subMatrixB[k * blockDim.x + threadX];
        }

        __syncthreads();
    }

    if (row < matrixSize && col < matrixSize) {
        matrixC[row * matrixSize + col] = cellValue;
    }
}

// CPU function for matrix multiplication
void matrixMultiplyCPU(const vector<float>& matrixA, const vector<float>& matrixB, vector<float>& matrixC, int matrixSize) {
    for (int row = 0; row < matrixSize; ++row) {
        for (int col = 0; col < matrixSize; ++col) {
            float sum = 0.0f;
            for (int k = 0; k < matrixSize; ++k) {
                sum += matrixA[row * matrixSize + k] * matrixB[k * matrixSize + col];
            }
            matrixC[row * matrixSize + col] = sum;
        }
    }
}

// Helper function to initialize matrices with random values
void initializeMatrix(vector<float>& matrix, int size) {
    for (int i = 0; i < size * size; ++i) {
        matrix[i] = static_cast<float>(rand()) / RAND_MAX;
    }
}

int main() {
    const int matrixSize = 1024; // Matrix size
    const int blockSize = 16; // Block size for CUDA kernel

    size_t totalBytes = matrixSize * matrixSize * sizeof(float);

    // Allocate host memory
    vector<float> hostMatrixA(matrixSize * matrixSize);
    vector<float> hostMatrixB(matrixSize * matrixSize);
    vector<float> hostMatrixC(matrixSize * matrixSize, 0);
    vector<float> referenceMatrixC(matrixSize * matrixSize, 0);

    initializeMatrix(hostMatrixA, matrixSize);
    initializeMatrix(hostMatrixB, matrixSize);

    // Verify initialization
    cout << "hostMatrixA[0]: " << hostMatrixA[0] << endl;
    cout << "hostMatrixB[0]: " << hostMatrixB[0] << endl;

    // Measure CPU execution time
    auto startCPU = high_resolution_clock::now();
    matrixMultiplyCPU(hostMatrixA, hostMatrixB, referenceMatrixC, matrixSize);
    auto endCPU = high_resolution_clock::now();
    duration<double> cpuDuration = endCPU - startCPU;
    cout << "CPU execution time: " << cpuDuration.count() << " seconds." << endl;

    // Allocate device memory
    float *deviceMatrixA, *deviceMatrixB, *deviceMatrixC;
    cudaMalloc(&deviceMatrixA, totalBytes);
    cudaMalloc(&deviceMatrixB, totalBytes);
    cudaMalloc(&deviceMatrixC, totalBytes);

    // Copy data to device
    cudaMemcpy(deviceMatrixA, hostMatrixA.data(), totalBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(deviceMatrixB, hostMatrixB.data(), totalBytes, cudaMemcpyHostToDevice);

    // Define grid and block dimensions
    dim3 threadsPerBlock(blockSize, blockSize);
    dim3 blocksPerGrid((matrixSize + blockSize - 1) / blockSize, (matrixSize + blockSize - 1) / blockSize);

    // Measure GPU execution time
    cudaEvent_t startEvent, stopEvent;
    cudaEventCreate(&startEvent);
    cudaEventCreate(&stopEvent);

    size_t sharedMemorySize = 2 * blockSize * blockSize * sizeof(float);
    cudaEventRecord(startEvent);
    matrixMultiplySharedMemory<<<blocksPerGrid, threadsPerBlock, sharedMemorySize>>>(deviceMatrixA, deviceMatrixB, deviceMatrixC, matrixSize);
    cudaEventRecord(stopEvent);

    cudaEventSynchronize(stopEvent);
    float gpuDuration = 0;
    cudaEventElapsedTime(&gpuDuration, startEvent, stopEvent);
    cout << "GPU execution time: " << gpuDuration / 1000.0 << " seconds." << endl;

    // Copy result back to host
    cudaMemcpy(hostMatrixC.data(), deviceMatrixC, totalBytes, cudaMemcpyDeviceToHost);

    // Verify results
    bool isMatch = true;
    for (int i = 0; i < matrixSize * matrixSize; ++i) {
        if (fabs(hostMatrixC[i] - referenceMatrixC[i]) > 1e-4) { // Adjusted tolerance for comparison
            cout << "Mismatch at index " << i << ": GPU result = " << hostMatrixC[i] << ", CPU result = " << referenceMatrixC[i] << endl;
            isMatch = false;
            break;
        }
    }

    cout << "Matrix multiplication " << (isMatch ? "PASSED" : "FAILED") << endl;

    // Free device memory
    cudaFree(deviceMatrixA);
    cudaFree(deviceMatrixB);
    cudaFree(deviceMatrixC);

    // Check for CUDA errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        cout << "CUDA error: " << cudaGetErrorString(err) << endl;
    }

    return 0;
}