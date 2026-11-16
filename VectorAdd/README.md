# VectorAdd

A small CUDA C++ project that adds two arrays element by element on the CPU and GPU.

It demonstrates the basic CUDA workflow:

1. Allocate memory on the GPU.
2. Copy input data from the CPU to the GPU.
3. Launch a kernel to add the elements.
4. Copy the result back and check it against the CPU result.

## Build

Open `VectorAdd.vcxproj` in Visual Studio 2026, select `x64` and `Debug` or `Release`, then build and run.

Requirements:

- Visual Studio 2026 with **Desktop development with C++**
- NVIDIA CUDA Toolkit 13.5 with Visual Studio integration
- A compatible NVIDIA GPU and driver

The project targets `compute_86,sm_86`. Change **CUDA C/C++ → Device → Code Generation** in the project properties to target another GPU architecture.

## Tests and benchmark

The program checks empty input, a single element, and array sizes around the 256-thread block boundary. It then compares CPU and GPU results for several larger arrays.

GPU timings include memory allocation, data transfers, kernel execution, and cleanup. For small arrays, this overhead can make the GPU version slower than the CPU version.
