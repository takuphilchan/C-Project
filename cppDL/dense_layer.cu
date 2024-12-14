#include <iostream>
#include <cuda_runtime.h>

// Use standard namespace
using namespace std;

// Define activation function (ReLU)
__global__ void relu(float* output_matrix, int num_samples, int num_neurons) {
    int thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = num_samples * num_neurons;

    if (thread_idx < total_elements) {
        output_matrix[thread_idx] = fmaxf(0.0f, output_matrix[thread_idx]); // Apply ReLU element-wise
    }
}

// Matrix multiplication kernel: C = A * B
__global__ void matmul(const float* input_matrix, const float* weight_matrix, float* output_matrix, int num_samples, int num_features, int num_neurons) {
    int row_idx = blockIdx.y * blockDim.y + threadIdx.y;
    int col_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (row_idx < num_samples && col_idx < num_neurons) {
        float accumulated_value = 0.0f;
        for (int feature_idx = 0; feature_idx < num_features; ++feature_idx) {
            accumulated_value += input_matrix[row_idx * num_features + feature_idx] * weight_matrix[feature_idx * num_neurons + col_idx];
        }
        output_matrix[row_idx * num_neurons + col_idx] = accumulated_value;
    }
}

// Add bias kernel
__global__ void add_bias(float* output_matrix, const float* bias_vector, int num_samples, int num_neurons) {
    int row_idx = blockIdx.y * blockDim.y + threadIdx.y;
    int col_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (row_idx < num_samples && col_idx < num_neurons) {
        output_matrix[row_idx * num_neurons + col_idx] += bias_vector[col_idx]; // Add bias for each column
    }
}

int main() {
    // Define layer dimensions
    const int num_samples = 2; // Batch size
    const int num_features = 3; // Input features
    const int num_neurons = 4; // Output neurons

    // Host memory allocation
    float host_input_matrix[num_samples * num_features] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f}; // Input matrix
    float host_weight_matrix[num_features * num_neurons] = { // Weight matrix
        0.1f, 0.2f, 0.3f, 0.4f,
        0.5f, 0.6f, 0.7f, 0.8f,
        0.9f, 1.0f, 1.1f, 1.2f
    };
    float host_bias_vector[num_neurons] = {0.1f, 0.2f, 0.3f, 0.4f}; // Bias vector
    float host_output_matrix[num_samples * num_neurons]; // Output matrix

    // Device memory allocation
    float *device_input_matrix, *device_weight_matrix, *device_bias_vector, *device_output_matrix;
    cudaMalloc(&device_input_matrix, num_samples * num_features * sizeof(float));
    cudaMalloc(&device_weight_matrix, num_features * num_neurons * sizeof(float));
    cudaMalloc(&device_bias_vector, num_neurons * sizeof(float));
    cudaMalloc(&device_output_matrix, num_samples * num_neurons * sizeof(float));

    // Copy data to device
    cudaMemcpy(device_input_matrix, host_input_matrix, num_samples * num_features * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(device_weight_matrix, host_weight_matrix, num_features * num_neurons * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(device_bias_vector, host_bias_vector, num_neurons * sizeof(float), cudaMemcpyHostToDevice);

    // Define grid and block sizes
    dim3 block_size(16, 16);
    dim3 grid_size((num_neurons + block_size.x - 1) / block_size.x, (num_samples + block_size.y - 1) / block_size.y);

    // Perform matrix multiplication
    matmul<<<grid_size, block_size>>>(device_input_matrix, device_weight_matrix, device_output_matrix, num_samples, num_features, num_neurons);

    // Add biases
    add_bias<<<grid_size, block_size>>>(device_output_matrix, device_bias_vector, num_samples, num_neurons);

    // Apply activation function (ReLU)
    int total_threads = (num_samples * num_neurons + 255) / 256; // Use 256 threads per block
    relu<<<total_threads, 256>>>(device_output_matrix, num_samples, num_neurons);

    // Copy result back to host
    cudaMemcpy(host_output_matrix, device_output_matrix, num_samples * num_neurons * sizeof(float), cudaMemcpyDeviceToHost);

    // Print the result
    cout << "Output (Z):\n";
    for (int sample_idx = 0; sample_idx < num_samples; ++sample_idx) {
        for (int neuron_idx = 0; neuron_idx < num_neurons; ++neuron_idx) {
            cout << host_output_matrix[sample_idx * num_neurons + neuron_idx] << " ";
        }
        cout << "\n";
    }

    // Free device memory
    cudaFree(device_input_matrix);
    cudaFree(device_weight_matrix);
    cudaFree(device_bias_vector);
    cudaFree(device_output_matrix);

    return 0;
}
