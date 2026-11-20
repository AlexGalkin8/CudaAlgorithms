#pragma once

#include <cuda_runtime.h>
#include <iostream>
#include <vector>

using PrefixSumFunction = cudaError_t (*)(const int* input, int* output, int length);

cudaError_t prefixSum(const int* input, int* output, int length);

inline bool runPrefixSumTest(const char* testName, const std::vector<int>& input,
							 const std::vector<int>& expected, PrefixSumFunction scan)
{
	if (input.size() != expected.size())
	{
		std::cerr << testName << " has invalid test data\n";
		return false;
	}

	std::vector<int> output(input.size(), 0);
	const cudaError_t error = scan(input.data(), output.data(), static_cast<int>(input.size()));
	if (error != cudaSuccess)
	{
		std::cerr << testName << " failed: " << cudaGetErrorString(error) << '\n';
		return false;
	}

	for (size_t i = 0; i < expected.size(); ++i)
	{
		if (output[i] != expected[i])
		{
			std::cerr << testName << " failed at index " << i
					  << ": expected " << expected[i] << ", got " << output[i] << '\n';
			return false;
		}
	}

	std::cout << testName << " passed (" << input.size() << " elements)\n";
	return true;
}

inline int runPrefixSumTests(PrefixSumFunction scan)
{
	bool allTestsPassed = true;
	allTestsPassed = runPrefixSumTest("Empty input", {}, {}, scan) && allTestsPassed;
	allTestsPassed = runPrefixSumTest("Single element", { 5 }, { 0 }, scan) && allTestsPassed;
	allTestsPassed = runPrefixSumTest("Known values", { 1, 2, 3, 4 }, { 0, 1, 3, 6 }, scan) && allTestsPassed;
	allTestsPassed = runPrefixSumTest("Negative values", { 4, -7, 2, -1 }, { 0, 4, -3, -1 }, scan) && allTestsPassed;
	allTestsPassed = runPrefixSumTest("Zero values", { 0, 0, 0, 0 }, { 0, 0, 0, 0 }, scan) && allTestsPassed;

	constexpr int sizes[] = { 255, 256, 257, 513, 1024, 65535, 65536, 65537, 1000000, 16777217 };
	for (const int size : sizes)
	{
		std::vector<int> input(size);
		std::vector<int> expected(size);
		int sum = 0;
		for (int i = 0; i < size; ++i)
		{
			input[i] = (i % 5) - 2;
			expected[i] = sum;
			sum += input[i];
		}

		allTestsPassed = runPrefixSumTest("Boundary/large input", input, expected, scan) && allTestsPassed;
	}

	return allTestsPassed ? 0 : 1;
}

