import numpy as np
import cupy as cp
import time

# Define the size of the matrix
MATRIX_SIZE = 12000

# Initialize matrices A and B with random values (single precision)
matrixA = np.random.rand(MATRIX_SIZE, MATRIX_SIZE).astype(np.float32)
matrixB = np.random.rand(MATRIX_SIZE, MATRIX_SIZE).astype(np.float32)

# Initialize matrix C for CPU result
cpuResultMatrix = np.zeros((MATRIX_SIZE, MATRIX_SIZE), dtype=np.float32)

# CPU Matrix Multiplication using np.dot
start_cpu = time.time()
cpuResultMatrix = np.dot(matrixA, matrixB)
end_cpu = time.time()
print(f"CPU execution time: {end_cpu - start_cpu:.6f} seconds.")

# GPU Matrix Multiplication using CuPy
matrixA_gpu = cp.array(matrixA)
matrixB_gpu = cp.array(matrixB)

start_gpu = time.time()
gpuResultMatrix = cp.dot(matrixA_gpu, matrixB_gpu)  # Optimized GPU multiplication
end_gpu = time.time()
print(f"GPU execution time: {end_gpu - start_gpu:.6f} seconds.")

# Copy GPU result back to host and compare with CPU result
gpuResultMatrix_host = gpuResultMatrix.get()

# Verify results
if np.allclose(cpuResultMatrix, gpuResultMatrix_host, atol=1e-4):
    print("Matrix multiplication PASSED")
else:
    print("Matrix multiplication FAILED")
