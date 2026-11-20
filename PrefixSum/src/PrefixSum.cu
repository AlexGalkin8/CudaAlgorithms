#include <cuda_runtime.h>

static const int threadsPerBlock = 256;

__global__ void blockExclusiveScanKernel(const int* input, int* output, int* blockSums, int length)
{
	__shared__ int values[2][threadsPerBlock];
	const int workIndex = threadIdx.x + blockIdx.x * blockDim.x;
	
	int currentValue = 0;
	if (workIndex < length)
	{
		currentValue = input[workIndex];
	}

	values[0][threadIdx.x] = currentValue;

	__syncthreads();

	int readIndex = 0;
	for (unsigned int distance = 1; distance < threadsPerBlock; distance *= 2)
	{
		const int writeIndex = 1 - readIndex;

		if (distance <= threadIdx.x)
		{
			int currentIdx = threadIdx.x - distance;
			currentValue += values[readIndex][currentIdx];
		}

		values[writeIndex][threadIdx.x] = currentValue;

		__syncthreads();

		readIndex = writeIndex;
	}

	if (workIndex < length)
	{
		output[workIndex] = threadIdx.x == 0 ? 0 : values[readIndex][threadIdx.x - 1];
	}

	if (blockSums != nullptr && threadIdx.x == threadsPerBlock - 1)
	{
		blockSums[blockIdx.x] = values[readIndex][threadsPerBlock - 1];
	}
}

__global__ void addBlockOffsetsKernel(int* output, const int* blockOffsets, int length)
{
	const int workIndex = threadIdx.x + blockIdx.x * blockDim.x;
	if (workIndex < length)
	{
		output[workIndex] += blockOffsets[blockIdx.x];
	}
}

namespace
{
	// Device pointers stay on the GPU throughout the recursive hierarchy.
	cudaError_t exclusiveScanDevice(const int* input, int* output, int length)
	{
		const int blocks = 1 + (length - 1) / threadsPerBlock;
		if (blocks == 1)
		{
			blockExclusiveScanKernel<<<1, threadsPerBlock>>>(input, output, nullptr, length);
			return cudaGetLastError();
		}

		int* blockSums = nullptr;
		int* blockOffsets = nullptr;
		const size_t bytes = static_cast<size_t>(blocks) * sizeof(int);
		cudaError_t error = cudaMalloc(&blockSums, bytes);
		if (error != cudaSuccess) goto cleanup;

		error = cudaMalloc(&blockOffsets, bytes);
		if (error != cudaSuccess) goto cleanup;

		blockExclusiveScanKernel<<<blocks, threadsPerBlock>>>(input, output, blockSums, length);
		error = cudaGetLastError();
		if (error != cudaSuccess) goto cleanup;

		// Recursion produces complete offsets, even when the sums span multiple blocks.
		error = exclusiveScanDevice(blockSums, blockOffsets, blocks);
		if (error != cudaSuccess) goto cleanup;

		addBlockOffsetsKernel<<<blocks, threadsPerBlock>>>(output, blockOffsets, length);
		error = cudaGetLastError();

	cleanup:
		// Complete queued uses before releasing this level's temporary buffers.
		{
			const cudaError_t syncError = cudaDeviceSynchronize();
			if (error == cudaSuccess) error = syncError;
		}
		if (blockOffsets != nullptr)
		{
			const cudaError_t freeError = cudaFree(blockOffsets);
			if (error == cudaSuccess) error = freeError;
		}
		if (blockSums != nullptr)
		{
			const cudaError_t freeError = cudaFree(blockSums);
			if (error == cudaSuccess) error = freeError;
		}
		return error;
	}
}

cudaError_t prefixSum(const int* input, int* output, int length)
{
	if (length < 0 || (length > 0 && (input == nullptr || output == nullptr)))
		return cudaErrorInvalidValue;
	if (length == 0) return cudaSuccess;

	int* deviceInput = nullptr;
	int* deviceOutput = nullptr;
	const size_t bytes = static_cast<size_t>(length) * sizeof(int);
	cudaError_t error = cudaMalloc(&deviceInput, bytes);
	if (error != cudaSuccess) goto cleanup;
	error = cudaMalloc(&deviceOutput, bytes);
	if (error != cudaSuccess) goto cleanup;
	error = cudaMemcpy(deviceInput, input, bytes, cudaMemcpyHostToDevice);
	if (error != cudaSuccess) goto cleanup;

	error = exclusiveScanDevice(deviceInput, deviceOutput, length);
	if (error != cudaSuccess) goto cleanup;
	error = cudaDeviceSynchronize();
	if (error != cudaSuccess) goto cleanup;
	error = cudaMemcpy(output, deviceOutput, bytes, cudaMemcpyDeviceToHost);

cleanup:
	if (deviceOutput != nullptr)
	{
		const cudaError_t freeError = cudaFree(deviceOutput);
		if (error == cudaSuccess) error = freeError;
	}
	if (deviceInput != nullptr)
	{
		const cudaError_t freeError = cudaFree(deviceInput);
		if (error == cudaSuccess) error = freeError;
	}
	return error;
}
