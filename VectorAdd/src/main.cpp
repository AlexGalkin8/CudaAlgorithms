#include "Tests.h"
#include "..\..\Common\Benchmark.h"

#include <vector>

namespace
{
    bool vectorAddCpu(const float* A, const float* B, float* result, int vectorLength)
    {
        for (int i = 0; i < vectorLength; ++i)
        {
            result[i] = A[i] + B[i];
        }
        return true;
    }

    bool runVectorAddBenchmark(int vectorLength)
    {
        std::vector<float> A(vectorLength);
        std::vector<float> B(vectorLength);
        std::vector<float> cpuResult(vectorLength, 0.0f);
        std::vector<float> gpuResult(vectorLength, 0.0f);

        for (int i = 0; i < vectorLength; ++i)
        {
            A[i] = static_cast<float>(i) * 1.25f - 3.0f;
            B[i] = static_cast<float>(i) * -0.5f + 2.0f;
        }

        if (!vectorAddCpu(A.data(), B.data(), cpuResult.data(), vectorLength))
        {
            return false;
        }

        const cudaError_t gpuError = vectorAdd(A.data(), B.data(), gpuResult.data(), vectorLength);
        if (gpuError != cudaSuccess)
        {
            std::cerr << "GPU benchmark setup failed: " << cudaGetErrorString(gpuError) << '\n';
            return false;
        }

        for (int i = 0; i < vectorLength; ++i)
        {
            if (std::fabs(cpuResult[i] - gpuResult[i]) > 1e-5f)
            {
                std::cerr << "CPU/GPU results differ at index " << i << '\n';
                return false;
            }
        }

        const benchmark::Result cpu = benchmark::measure("CPU vectorAdd", [&]()
        {
            return vectorAddCpu(A.data(), B.data(), cpuResult.data(), vectorLength);
        });
        const benchmark::Result gpu = benchmark::measure("GPU vectorAdd", [&]()
        {
            const cudaError_t error = vectorAdd(A.data(), B.data(), gpuResult.data(), vectorLength);
            if (error != cudaSuccess)
            {
                std::cerr << "GPU benchmark failed: " << cudaGetErrorString(error) << '\n';
                return false;
            }
            return true;
        });

        if (!cpu.succeeded || !gpu.succeeded)
        {
            return false;
        }

        benchmark::printComparison("Vector addition", vectorLength, cpu, gpu);
        return true;
    }

    int runVectorAddBenchmarks()
    {
        constexpr int sizes[] = { 1024, 65536, 1048576 };
        for (const int size : sizes)
        {
            if (!runVectorAddBenchmark(size))
            {
                return 1;
            }
        }
        return 0;
    }
}

int main()
{
    if (runVectorAddTests() != 0)
    {
        return 1;
    }

    return runVectorAddBenchmarks();
}
