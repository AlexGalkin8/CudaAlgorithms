#pragma once

#include <cmath>
#include <iostream>
#include <vector>
#include <cuda_runtime.h>

cudaError_t vectorAdd(const float* A, const float* B, float* result, int vectorLength);

inline bool runVectorAddTest(const char* testName, int vectorLength)
{
	std::vector<float> A(vectorLength);
	std::vector<float> B(vectorLength);
	std::vector<float> result(vectorLength, 0.0f);

	for (int i = 0; i < vectorLength; ++i)
	{
		A[i] = static_cast<float>(i) * 1.25f - 3.0f;
		B[i] = static_cast<float>(i) * -0.5f + 2.0f;
	}

	const cudaError_t error = vectorAdd(A.data(), B.data(), result.data(), vectorLength);
	if (error != cudaSuccess)
	{
		std::cerr << testName << " failed: " << cudaGetErrorString(error) << '\n';
		return false;
	}

	constexpr float tolerance = 1e-5f;
	for (int i = 0; i < vectorLength; ++i)
	{
		const float expected = A[i] + B[i];
		if (std::fabs(result[i] - expected) > tolerance)
		{
			std::cerr << testName << " failed at index " << i
					  << ": expected " << expected << ", got " << result[i] << '\n';
			return false;
		}
	}

	std::cout << testName << " passed (" << vectorLength << " elements)\n";
	return true;
}

inline int runVectorAddTests()
{
	bool allTestsPassed = true;
	allTestsPassed = runVectorAddTest("Empty vector", 0) && allTestsPassed;
	allTestsPassed = runVectorAddTest("Single element", 1) && allTestsPassed;
	allTestsPassed = runVectorAddTest("Below block boundary", 255) && allTestsPassed;
	allTestsPassed = runVectorAddTest("At block boundary", 256) && allTestsPassed;
	allTestsPassed = runVectorAddTest("Above block boundary", 257) && allTestsPassed;

	return allTestsPassed ? 0 : 1;
}
