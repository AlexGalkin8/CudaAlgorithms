#include <cuda_runtime.h>

__global__ void vectorAddKernel(const float* A, const float* B, float* result, int vectorLength)
{
    const int workIndex = threadIdx.x + blockIdx.x * blockDim.x;
    if (workIndex < vectorLength)
    {
        result[workIndex] = A[workIndex] + B[workIndex];
    }
}

cudaError_t vectorAdd(const float* A, const float* B, float* result, int vectorLength)
{
    if (vectorLength < 0 || (vectorLength > 0 && (A == nullptr || B == nullptr || result == nullptr)))
    {
        return cudaErrorInvalidValue;
    }
    if (vectorLength == 0)
    {
        return cudaSuccess;
    }

    float* gpuMemA = nullptr;
    float* gpuMemB = nullptr;
    float* gpuMemResult = nullptr;
    cudaError_t error = cudaSuccess;
    const size_t byteCount = static_cast<size_t>(vectorLength) * sizeof(float);

    error = cudaMalloc(&gpuMemA, byteCount);
    if (error != cudaSuccess) goto cleanup;

    error = cudaMalloc(&gpuMemB, byteCount);
    if (error != cudaSuccess) goto cleanup;

    error = cudaMalloc(&gpuMemResult, byteCount);
    if (error != cudaSuccess) goto cleanup;

    error = cudaMemcpy(gpuMemA, A, byteCount, cudaMemcpyHostToDevice);
    if (error != cudaSuccess) goto cleanup;

    error = cudaMemcpy(gpuMemB, B, byteCount, cudaMemcpyHostToDevice);
    if (error != cudaSuccess) goto cleanup;

    {
        constexpr int threadsPerBlock = 256;
        const int blocks = (vectorLength + threadsPerBlock - 1) / threadsPerBlock;
        vectorAddKernel<<<blocks, threadsPerBlock>>>(gpuMemA, gpuMemB, gpuMemResult, vectorLength);
    }

    error = cudaGetLastError();
    if (error != cudaSuccess) goto cleanup;

    error = cudaDeviceSynchronize();
    if (error != cudaSuccess) goto cleanup;

    error = cudaMemcpy(result, gpuMemResult, byteCount, cudaMemcpyDeviceToHost);

cleanup:
    if (gpuMemResult != nullptr)
    {
        const cudaError_t freeError = cudaFree(gpuMemResult);
        if (error == cudaSuccess) error = freeError;
    }
    if (gpuMemB != nullptr)
    {
        const cudaError_t freeError = cudaFree(gpuMemB);
        if (error == cudaSuccess) error = freeError;
    }
    if (gpuMemA != nullptr)
    {
        const cudaError_t freeError = cudaFree(gpuMemA);
        if (error == cudaSuccess) error = freeError;
    }

    return error;
}
