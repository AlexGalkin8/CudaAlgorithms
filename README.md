# CudaAlgorithms

A collection of small CUDA C++ projects for learning GPU programming, parallel algorithms, and performance optimization.

## Build

Open `CudaAlgorithms.slnx` in Visual Studio 2026. Select a project, choose the `x64` platform and a configuration, then build and run.

Requirements:

- Visual Studio 2026 with **Desktop development with C++**
- NVIDIA CUDA Toolkit with Visual Studio integration
- A compatible NVIDIA GPU and driver

The projects use Visual Studio `.vcxproj` files and do not require CMake.

## Projects

| Project | Description | Status |
|---|---|---|
| [VectorAdd](VectorAdd/README.md) | Element-wise vector addition on the CPU and GPU | Complete |
| [PrefixSum](PrefixSum/README.md) | Hierarchical exclusive prefix sum | Complete |
| `MemoryPatterns` | GPU memory access and shared memory | Planned |
| `StreamCompaction` | Filter elements into a compact array using scan | Next |

Each project has its own folder and Visual Studio project file. Shared code is kept in `Common/`.

See [VectorAdd/README.md](VectorAdd/README.md) for details about the first project.
