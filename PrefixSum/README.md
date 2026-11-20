# PrefixSum

A CUDA C++ implementation of exclusive prefix sum for integer arrays.

```text
Input:  [2, 5, 1, 3]
Output: [0, 2, 7, 8]
```

## Implementation

Each block scans up to 256 elements using a Hillis–Steele scan and two shared-memory buffers. The algorithm recursively scans block sums, then adds the resulting offsets to each block's local output.

The host function copies input to the GPU and returns the result to CPU memory. Temporary device buffers are allocated per call. All intermediate sums must fit in `int`.

## Build and run

Open `../CudaAlgorithms.slnx` in Visual Studio 2026 and select PrefixSum as the startup project. Build for x64; use Release for benchmarks.

Requires CUDA Toolkit 13.5 with Visual Studio integration and a compatible NVIDIA driver. The project targets `compute_86,sm_86`.

## Tests and benchmarks

Tests compare results with a CPU reference, covering empty input, negative values, block boundaries, and multiple hierarchy levels, including 16,777,217 elements.

Benchmarks use arrays from 65,536 to 67,108,864 elements. GPU timings include allocation, transfers, all scan stages, synchronization, and cleanup. Reported speedups compare this implementation with the included sequential CPU baseline and depend on the system and build configuration.

Use `--cpu-only` to run a separate CPU diagnostic benchmark. A sharp CPU timing increase between roughly 1.5 and 2 million elements was observed on the development machine; its cause has not been established.
