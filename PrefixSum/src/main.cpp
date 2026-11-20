#include "Tests.h"
#include "..\..\Common\Benchmark.h"

#include <vector>
#include <string_view>

namespace
{
	cudaError_t prefixSumCpu(const int* input, int* output, int length)
	{
		if (length < 0 || (length > 0 && (input == nullptr || output == nullptr)))
		{
			return cudaErrorInvalidValue;
		}

		int sum = 0;
		for (int i = 0; i < length; ++i)
		{
			output[i] = sum;
			sum += input[i];
		}
		return cudaSuccess;
	}

	bool runPrefixSumBenchmark(int length)
	{
		std::vector<int> input(length);
		std::vector<int> cpuOutput(length, 0);
		std::vector<int> gpuOutput(length, 0);
		for (int i = 0; i < length; ++i)
		{
			input[i] = (i % 5) - 2;
		}

		cudaError_t error = prefixSumCpu(input.data(), cpuOutput.data(), length);
		if (error != cudaSuccess)
		{
			std::cerr << "CPU benchmark setup failed: " << cudaGetErrorString(error) << '\n';
			return false;
		}

		error = prefixSum(input.data(), gpuOutput.data(), length);
		if (error != cudaSuccess)
		{
			std::cerr << "GPU benchmark setup failed: " << cudaGetErrorString(error) << '\n';
			return false;
		}

		for (int i = 0; i < length; ++i)
		{
			if (cpuOutput[i] != gpuOutput[i])
			{
				std::cerr << "CPU/GPU results differ at index " << i
						  << ": CPU " << cpuOutput[i] << ", GPU " << gpuOutput[i] << '\n';
				return false;
			}
		}

		const benchmark::Result cpu = benchmark::measure("CPU PrefixSum", [&]()
		{
			return prefixSumCpu(input.data(), cpuOutput.data(), length) == cudaSuccess;
		});
		const benchmark::Result gpu = benchmark::measure("GPU PrefixSum", [&]()
		{
			const cudaError_t scanError = prefixSum(input.data(), gpuOutput.data(), length);
			if (scanError != cudaSuccess)
			{
				std::cerr << "GPU benchmark failed: " << cudaGetErrorString(scanError) << '\n';
				return false;
			}
			return true;
		});

		if (!cpu.succeeded || !gpu.succeeded)
		{
			return false;
		}

		for (int i = 0; i < length; ++i)
		{
			if (cpuOutput[i] != gpuOutput[i])
			{
				std::cerr << "CPU/GPU results differ after profiling at index " << i << '\n';
				return false;
			}
		}

		benchmark::printComparison("Exclusive prefix sum", length, cpu, gpu);
		return true;
	}

	int runPrefixSumBenchmarks()
	{
		constexpr int sizes[] = { 65536, 1048576, 4194304, 16777216, 67108864 };
		for (const int size : sizes)
		{
			if (!runPrefixSumBenchmark(size))
			{
				return 1;
			}
		}
		return 0;
	}
}

// Diagnostic mode: isolate CPU timing from CUDA initialization and transfers.
int runCpuOnlyBenchmarks()
{
    constexpr int sizes[] = { 1048576, 1572864, 2097152, 2621440, 3145728, 3670016, 4194304 };
    for (int length : sizes)
    {
        std::vector<int> input(length);
        std::vector<int> output(length);
        for (int i = 0; i < length; ++i) input[i] = (i % 5) - 2;
        const auto timing = benchmark::measure("CPU-only PrefixSum", [&]()
        {
            return prefixSumCpu(input.data(), output.data(), length) == cudaSuccess;
        }, 5, 20);
        if (!timing.succeeded) return 1;
        int expected = 0;
        for (int i = 0; i < length; ++i)
        {
            if (output[i] != expected) return 1;
            expected += input[i];
        }
        std::cout << length << " elements: median " << timing.medianMilliseconds
                  << " ms, min " << timing.minimumMilliseconds << " ms\n";
    }
    return 0;
}

int main(int argc, char** argv)
{
    if (argc == 2 && std::string_view(argv[1]) == "--cpu-only")
        return runCpuOnlyBenchmarks();
    if (runPrefixSumTests(prefixSum) != 0) return 1;
    return runPrefixSumBenchmarks();
}
