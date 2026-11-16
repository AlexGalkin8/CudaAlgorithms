#pragma once

#include <algorithm>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

namespace benchmark
{
	struct Result
	{
		bool succeeded = false;
		double medianMilliseconds = 0.0;
		double minimumMilliseconds = 0.0;
		double averageMilliseconds = 0.0;
	};

	template <typename Operation>
	Result measure(const char* name, Operation&& operation, int warmupCount = 2, int sampleCount = 10)
	{
		if (warmupCount < 0 || sampleCount <= 0)
		{
			std::cerr << "Invalid profiling configuration for " << name << '\n';
			return {};
		}

		for (int i = 0; i < warmupCount; ++i)
		{
			if (!operation())
			{
				std::cerr << name << " failed during warm-up\n";
				return {};
			}
		}

		std::vector<double> samples;
		samples.reserve(sampleCount);
		for (int i = 0; i < sampleCount; ++i)
		{
			const auto start = std::chrono::steady_clock::now();
			if (!operation())
			{
				std::cerr << name << " failed during profiling\n";
				return {};
			}
			const auto stop = std::chrono::steady_clock::now();
			samples.push_back(std::chrono::duration<double, std::milli>(stop - start).count());
		}

		std::sort(samples.begin(), samples.end());
		const size_t middle = samples.size() / 2;
		const double median = samples.size() % 2 == 0
			? (samples[middle - 1] + samples[middle]) / 2.0
			: samples[middle];

		return {
			true,
			median,
			samples.front(),
			std::accumulate(samples.begin(), samples.end(), 0.0) / samples.size()
		};
	}

	inline void printComparison(const char* algorithmName, int elementCount, const Result& cpu, const Result& gpu)
	{
		if (!cpu.succeeded || !gpu.succeeded)
		{
			return;
		}

		std::cout << "\n" << algorithmName << " - " << elementCount << " elements\n"
				  << std::fixed << std::setprecision(4)
				  << "  CPU median: " << cpu.medianMilliseconds << " ms"
				  << " (min " << cpu.minimumMilliseconds << " ms, average " << cpu.averageMilliseconds << " ms)\n"
				  << "  GPU median: " << gpu.medianMilliseconds << " ms"
				  << " (min " << gpu.minimumMilliseconds << " ms, average " << gpu.averageMilliseconds << " ms)\n";

		if (gpu.medianMilliseconds > 0.0)
		{
			const double speedup = cpu.medianMilliseconds / gpu.medianMilliseconds;
			if (speedup >= 1.0)
			{
				std::cout << "  GPU speedup: " << speedup << "x\n";
			}
			else
			{
				std::cout << "  CPU speedup: " << 1.0 / speedup << "x\n";
			}
		}
	}
}
