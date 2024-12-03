#include <iostream>
#include <cuda_fp16.h> // For half-precision floating-point operations (__half)

using namespace std;

// Macro for error checking CUDA calls
#define CUDA_CHECK(call) \
    { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            cerr << "CUDA error: " << cudaGetErrorString(err) << " at " << __FILE__ << ":" << __LINE__ << endl; \
            exit(err); \
        } \
    }

// CUDA kernel for element-wise addition of two half-precision arrays (mixed precision)
__global__ void half_precision_add_mixed_precision(const __half* input1, const __half* input2, __half* output, int size) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index < size) {
        // Convert half-precision inputs to float for accurate addition
        float temp1 = __half2float(input1[index]);
        float temp2 = __half2float(input2[index]);
        
        // Perform addition in full precision (float)
        float sum = temp1 + temp2;

        // Convert result back to half precision and store
        output[index] = __float2half(sum); // Convert sum back to half-precision
    }
}

int main() {
    // Number of elements in the arrays
    const int num_elements = 1024;

    // Calculate the size of arrays in bytes
    size_t array_size_in_bytes = num_elements * sizeof(__half);

    // Allocate host memory for input and output arrays
    __half* host_input1 = new __half[num_elements];
    __half* host_input2 = new __half[num_elements];
    __half* host_output = new __half[num_elements];

    // Initialize the input arrays on the host with a mix of large and small values
    for (int i = 0; i < num_elements; i++) {
        if (i % 2 == 0) {
            host_input1[i] = __float2half(1.0f);  // Large value
            host_input2[i] = __float2half(0.0001f);  // Small value
        } else {
            host_input1[i] = __float2half(100.0f);  // Larger value
            host_input2[i] = __float2half(0.00001f);  // Very small value
        }
    }

    // Allocate device memory for input and output arrays
    __half* device_input1;
    __half* device_input2;
    __half* device_output;
    CUDA_CHECK(cudaMalloc((void**)&device_input1, array_size_in_bytes));
    CUDA_CHECK(cudaMalloc((void**)&device_input2, array_size_in_bytes));
    CUDA_CHECK(cudaMalloc((void**)&device_output, array_size_in_bytes));

    // Copy the input arrays from host to device
    CUDA_CHECK(cudaMemcpy(device_input1, host_input1, array_size_in_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(device_input2, host_input2, array_size_in_bytes, cudaMemcpyHostToDevice));

    // Define the grid and block dimensions for the CUDA kernel launch
    int threads_per_block = 256;
    int blocks_per_grid = (num_elements + threads_per_block - 1) / threads_per_block;

    // Launch the CUDA kernel to perform mixed-precision addition
    half_precision_add_mixed_precision<<<blocks_per_grid, threads_per_block>>>(device_input1, device_input2, device_output, num_elements);
    CUDA_CHECK(cudaGetLastError()); // Check for kernel launch errors

    // Copy results from device to host
    CUDA_CHECK(cudaMemcpy(host_output, device_output, array_size_in_bytes, cudaMemcpyDeviceToHost));

    // Print the first few results
    for (int i = 0; i < 10; i++) {
        cout << "Result[" << i << "] = " << __half2float(host_output[i]) << endl; // Convert half to float for better readability
    }

    // Free host memory
    delete[] host_input1;
    delete[] host_input2;
    delete[] host_output;

    // Free device memory
    CUDA_CHECK(cudaFree(device_input1));
    CUDA_CHECK(cudaFree(device_input2));
    CUDA_CHECK(cudaFree(device_output));

    return 0;
}



