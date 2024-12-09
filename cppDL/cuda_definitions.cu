// Basic Functionality and Memory Management

cudaMalloc(), cudaFree(): /* Used to allocate and free memory on the GPU. 
                             cudaMalloc reserves space, and cudaFree releases it when done. */

cudaMallocHost(), cudaFreeHost(): /* Allocates and frees memory on the host using cudaMallocHost, 
                                     ensuring proper deallocation and avoiding memory leaks. */

cudaMemcpy(): /* Copies data between CPU and GPU memory. 
                 Use it to send data to the GPU or retrieve results from the GPU. */

cudaMemcpyAsync(): /* Asynchronous memory copy function, which allows overlap
                      of data transfer and computation. It doesn't block the CPU 
                      while waiting for the data transfer to complete. */

// Memory and Execution Configuration

cudaStream_t: /* Represents a stream of operations on the GPU. 
                 Streams allow for concurrent execution of multiple operations, 
                 improving overall performance by overlapping computation and communication. */

cudaEvent_t: /* Represents an event in GPU code. Used to record timestamps or 
                mark the completion of operations in the stream, 
                allowing synchronization and profiling. */

cudaLaunchKernel(): /* A function that allows a kernel to be launched 
                       on the GPU from the CPU with specific configurations 
                       for grid and block dimensions. */

cudaOccupancyMaxPotentialBlockSize(): /* A helper function to determine the maximum 
                                         block size and occupancy for a kernel based
                                         on available resources on the GPU. */

cudaDeviceSynchronize(): /* Forces the CPU to wait until the GPU finishes all
                            its tasks before proceeding with further operations. */

cudaThreadSynchronize(): /* Ensures that the CPU waits for the current GPU thread 
                            to complete before proceeding, similar to cudaDeviceSynchronize(), 
                            but specifically for thread synchronization. */

// GPU and CPU Specific Code Declarations

__device__: /* Defines a function that only runs on the GPU and 
               can only be called by other GPU functions or kernels. */

__host__: /* Defines a function that only runs on the CPU and cannot be called from the GPU. */

__device__ vs __host__: /* __device__ functions can be called only from GPU, 
                           while __host__ functions are designed to run on the CPU and 
                           cannot be called from GPU. When used together in a function declaration, 
                           both host and device code can run, 
                           making the function callable from both CPU and GPU. */

__global__: /* Marks a function as a "kernel" that can be executed in 
               parallel by many threads on the GPU. */

__shared__: /* Defines memory shared by all threads in the same block. 
               Useful for fast communication between threads in a block. */

// Thread and Block Management

threadIdx: /* A built-in variable that provides the index of the current 
              thread in its block. It helps identify which data each thread works on. */

blockIdx: /* A built-in variable that gives the index of the current block in the grid. 
             It helps identify which block is working on which data. */

blockDim: /* A built-in variable that gives the number of threads in each block. 
             It tells you how many threads are in the block. */

gridDim: /* A built-in variable that gives the total number of blocks in the grid. 
            It helps understand the size of the entire grid. */

// Synchronization and Atomic Operations

__syncthreads(): /* Ensures all threads in a block reach the same point before continuing, 
                    which is important for proper shared memory usage. */

atomicAdd(): /* Safely adds a value to a variable, ensuring no race conditions happen when 
                multiple threads try to modify the same variable. */

// Device Information and Error Handling

cudaError_t: /* Defines a type used for error checking in CUDA functions. 
                It's used to capture and handle error states from CUDA runtime functions. */

cudaGetDeviceProperties(): /* Queries and returns properties of a specific GPU device, 
                              such as memory size, compute capability, and more. */

cudaDeviceReset(): /* Resets the current device, releasing all allocated resources and 
                      restoring the device to its initial state. */





                      