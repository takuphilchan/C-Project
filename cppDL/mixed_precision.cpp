#include <cuda_fp16.h>
#include <iostream>
#include <vector>

using namespace std;  // Using the standard namespace to avoid having to prefix std::

// CUDA kernel for half-precision floating-point addition
__global__ void half_precision_add(const __half* input1, const __half* input2, __half* output, int num_elements) {
    // Calculate the global thread index for this thread
    int index = threadIdx.x + blockIdx.x * blockDim.x;

    // Perform the addition only for valid indices within the array bounds
    if (index < num_elements) {
        // Add the two input elements (in half precision) and store the result
        output[index] = __hadd(input1[index], input2[index]);
    }
}

int main() {
    const int num_elements = 1024;  // Number of elements in the array
    const int array_size_in_bytes = num_elements * sizeof(__half);  // Size in bytes for memory allocation

    // Initialize host input arrays with values (1.0 and 2.0) in half precision
    vector<half> host_input1(num_elements, __float2half(1.0f));
    vector<half> host_input2(num_elements, __float2half(2.0f));
    vector<half> host_output(num_elements);  // Array to store results

    // Declare device pointers for arrays
    __half *device_input1, *device_input2, *device_output;

    // Allocate memory on the device (GPU) for input and output arrays
    cudaMalloc(&device_input1, array_size_in_bytes);
    cudaMalloc(&device_input2, array_size_in_bytes);
    cudaMalloc(&device_output, array_size_in_bytes);

    // Copy input data from host to device
    cudaMemcpy(device_input1, host_input1.data(), array_size_in_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(device_input2, host_input2.data(), array_size_in_bytes, cudaMemcpyHostToDevice);

    // Define kernel launch configuration (number of threads and blocks)
    const int threads_per_block = 256;  // Number of threads per block
    const int blocks_per_grid = (num_elements + threads_per_block - 1) / threads_per_block;  // Number of blocks needed

    // Launch the CUDA kernel for the element-wise addition
    half_precision_add<<<blocks_per_grid, threads_per_block>>>(device_input1, device_input2, device_output, num_elements);

    // Copy the result from the device (GPU) back to the host (CPU)
    cudaMemcpy(host_output.data(), device_output, array_size_in_bytes, cudaMemcpyDeviceToHost);

    // Verify the results: Check if the sum of corresponding elements is correct
    bool success = true;
    for (int i = 0; i < num_elements; i++) {
        // Convert half precision to float for comparison
        float expected = __half2float(host_input1[i]) + __half2float(host_input2[i]);
        float result = __half2float(host_output[i]);

        // If the result is significantly different from the expected, print the mismatch
        if (abs(result - expected) > 1e-3f) {
            success = false;
            cout << "Mismatch at index " << i << ": expected " << expected << ", got " << result << endl;
            break;  // Stop at the first mismatch
        }
    }

    // If no mismatches were found, print success message
    if (success) {
        cout << "All results are correct!" << endl;
    }

    // Free device memory after computation
    cudaFree(device_input1);
    cudaFree(device_input2);
    cudaFree(device_output);

    return 0;
}
