# VectorAdd

A small CUDA learning project that compares element-wise vector addition on the CPU and GPU. The C++ host code creates and validates input data, while a CUDA kernel computes `result[i] = A[i] + B[i]` in parallel.

The project is intended to demonstrate the basic CUDA workflow: allocating device memory, transferring data between host and device, launching a kernel, checking errors, and copying results back to the host.

## Requirements

- Windows
- Visual Studio 2026 with the Desktop development with C++ workload
- NVIDIA CUDA Toolkit 13.5 with Visual Studio integration
- An NVIDIA GPU and a compatible NVIDIA driver

The project currently targets `compute_86,sm_86` (NVIDIA Ampere, including the GeForce RTX 3070 Ti). Change **CUDA C/C++ → Device → Code Generation** in the project properties if building for a different GPU architecture.

## Build and run

1. Open `VectorAdd.vcxproj` in Visual Studio, or add it to the parent solution.
2. Select the `x64` platform and either `Debug` or `Release` configuration.
3. Build and run the `VectorAdd` project.

The project uses the Visual Studio CUDA build customization; it does not require CMake.

## Implementation overview

- `src/main.cpp` creates input vectors, runs the CPU reference implementation, validates the GPU output, and reports benchmark results.
- `src/VectorAdd.cu` implements the host-side `vectorAdd` function and the `vectorAddKernel` CUDA kernel.
- `src/Tests.h` contains small correctness checks, including empty input and sizes around the 256-thread block boundary.
- `../../Common/Benchmark.h` provides the shared CPU-side timing helper.

`vectorAdd` allocates three device buffers, copies both inputs from host to device, launches the kernel, synchronizes and checks for execution errors, copies the output back, and frees the buffers. The kernel assigns one element to each thread and guards against indices outside the input length.

## Benchmarks

The executable compares the CPU reference with a complete GPU call for 1,024, 65,536, and 1,048,576 elements. Each measurement includes device allocation, host-to-device transfers, kernel launch and execution, device-to-host transfer, and device-buffer cleanup. This is an end-to-end comparison; it does not measure kernel execution in isolation or reuse GPU buffers between iterations.

Results depend on the GPU, driver, system load, and build configuration. For small inputs, transfer and launch overhead can make the GPU path slower than the CPU path.

## Learning goals

- Understand the host/device memory boundary.
- Practice CUDA memory allocation and data transfers.
- Map CUDA threads to array elements and handle a partial final block.
- Check CUDA API, kernel launch, and execution errors.
- Compare end-to-end CPU and GPU timings while accounting for transfer overhead.
