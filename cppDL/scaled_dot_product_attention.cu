#include <iostream>
#include <cmath>
#include <cuda_runtime.h>

// Kernel for matrix multiplication to compute Q * K^T
__global__ void computeAttentionScores(float* query, float* key, float* attentionScores, int batchSize, int seqLen, int embedDim) {
    int batchIdx = blockIdx.z;                    // Current batch index
    int rowIdx = blockIdx.y * blockDim.y + threadIdx.y;  // Row index in query
    int colIdx = blockIdx.x * blockDim.x + threadIdx.x;  // Column index in key (after transpose)

    if (rowIdx < seqLen && colIdx < seqLen) {
        float dotProduct = 0.0f;
        for (int i = 0; i < embedDim; ++i) {
            dotProduct += query[batchIdx * seqLen * embedDim + rowIdx * embedDim + i] *
                          key[batchIdx * seqLen * embedDim + colIdx * embedDim + i];
        }
        // Store result in attention scores matrix
        attentionScores[batchIdx * seqLen * seqLen + rowIdx * seqLen + colIdx] = dotProduct;
    }
}

// Kernel for scaling and softmax operation on attention scores
__global__ void applySoftmax(float* attentionScores, int batchSize, int seqLen) {
    int batchIdx = blockIdx.z;                     // Current batch index
    int rowIdx = blockIdx.y * blockDim.y + threadIdx.y; // Row index in attention scores

    if (rowIdx < seqLen) {
        // Find the maximum value for numerical stability
        float maxVal = -1e9f;
        for (int i = 0; i < seqLen; ++i) {
            maxVal = fmaxf(maxVal, attentionScores[batchIdx * seqLen * seqLen + rowIdx * seqLen + i]);
        }

        // Compute exponential values and sum them
        float expSum = 0.0f;
        for (int i = 0; i < seqLen; ++i) {
            float expVal = expf(attentionScores[batchIdx * seqLen * seqLen + rowIdx * seqLen + i] - maxVal);
            attentionScores[batchIdx * seqLen * seqLen + rowIdx * seqLen + i] = expVal;
            expSum += expVal;
        }

        // Normalize values to obtain softmax probabilities
        for (int i = 0; i < seqLen; ++i) {
            attentionScores[batchIdx * seqLen * seqLen + rowIdx * seqLen + i] /= expSum;
        }
    }
}

// Kernel for computing the weighted sum: Attention Scores * Value
__global__ void computeWeightedSum(float* attentionScores, float* value, float* output, int batchSize, int seqLen, int embedDim) {
    int batchIdx = blockIdx.z;                     // Current batch index
    int rowIdx = blockIdx.y * blockDim.y + threadIdx.y; // Row index in output
    int colIdx = blockIdx.x * blockDim.x + threadIdx.x; // Column index in output

    if (rowIdx < seqLen && colIdx < embedDim) {
        float weightedSum = 0.0f;
        for (int i = 0; i < seqLen; ++i) {
            weightedSum += attentionScores[batchIdx * seqLen * seqLen + rowIdx * seqLen + i] *
                           value[batchIdx * seqLen * embedDim + i * embedDim + colIdx];
        }
        // Store the weighted sum in the output matrix
        output[batchIdx * seqLen * embedDim + rowIdx * embedDim + colIdx] = weightedSum;
    }
}

int main() {
    // Hyperparameters
    const int batchSize = 1;  // Number of batches
    const int seqLen = 4;     // Sequence length
    const int embedDim = 4;   // Dimension of embedding

    // Host memory allocation for input tensors
    float h_query[batchSize * seqLen * embedDim] = {1.0, 0.0, 0.0, 1.0,   0.0, 1.0, 0.0, 1.0,   1.0, 0.0, 1.0, 0.0,   0.0, 1.0, 0.0, 1.0};
    float h_key[batchSize * seqLen * embedDim]   = {1.0, 0.0, 0.0, 1.0,   0.0, 1.0, 0.0, 1.0,   1.0, 0.0, 1.0, 0.0,   0.0, 1.0, 0.0, 1.0};
    float h_value[batchSize * seqLen * embedDim] = {0.1, 0.2, 0.3, 0.4,   0.5, 0.6, 0.7, 0.8,   0.9, 1.0, 1.1, 1.2,   1.3, 1.4, 1.5, 1.6};

    // Device memory allocation
    float *d_query, *d_key, *d_value, *d_attentionScores, *d_output;
    cudaMalloc(&d_query, batchSize * seqLen * embedDim * sizeof(float));
    cudaMalloc(&d_key, batchSize * seqLen * embedDim * sizeof(float));
    cudaMalloc(&d_value, batchSize * seqLen * embedDim * sizeof(float));
    cudaMalloc(&d_attentionScores, batchSize * seqLen * seqLen * sizeof(float));
    cudaMalloc(&d_output, batchSize * seqLen * embedDim * sizeof(float));

    // Copy input data from host to device
    cudaMemcpy(d_query, h_query, batchSize * seqLen * embedDim * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_key, h_key, batchSize * seqLen * embedDim * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_value, h_value, batchSize * seqLen * embedDim * sizeof(float), cudaMemcpyHostToDevice);

    // Define CUDA grid and block dimensions
    dim3 blockDim(16, 16); // Threads per block
    dim3 gridDim((seqLen + 15) / 16, (seqLen + 15) / 16, batchSize); // Blocks per grid

    // Launch kernels for each step of scaled dot-product attention
    computeAttentionScores<<<gridDim, blockDim>>>(d_query, d_key, d_attentionScores, batchSize, seqLen, embedDim);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA kernel launch error in computeAttentionScores: " << cudaGetErrorString(err) << std::endl;
    }

    applySoftmax<<<gridDim, blockDim>>>(d_attentionScores, batchSize, seqLen);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA kernel launch error in applySoftmax: " << cudaGetErrorString(err) << std::endl;
    }

    computeWeightedSum<<<gridDim, blockDim>>>(d_attentionScores, d_value, d_output, batchSize, seqLen, embedDim);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA kernel launch error in computeWeightedSum: " << cudaGetErrorString(err) << std::endl;
    }

    // Copy the output back to host
    float h_output[batchSize * seqLen * embedDim];
    cudaMemcpy(h_output, d_output, batchSize * seqLen * embedDim * sizeof(float), cudaMemcpyDeviceToHost);

    // Print the attention output
    std::cout << "Attention Output:\n";
    for (int i = 0; i < batchSize * seqLen * embedDim; ++i) {
        std::cout << h_output[i] << " ";
        if ((i + 1) % embedDim == 0) std::cout << "\n";
    }

    // Free device memory
    cudaFree(d_query);
    cudaFree(d_key);
    cudaFree(d_value);
    cudaFree(d_attentionScores);
    cudaFree(d_output);

    return 0;
}
