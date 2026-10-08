# FPcompress CUDA library

The CMake build provides all four GPU codecs as a reusable static library.
Include `<fpcompress/fpcompress.h>` and link `FPcompress::fpcompress` (or its
`FPcompress::cuda` alias). The public API has C linkage and can be called from
ordinary C/C++ programs; application code does not need to be compiled by nvcc.

## Build and install

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel
cmake --install build --prefix /path/to/install
```

CUDA and a compatible host compiler are required. Set
`-DCMAKE_CUDA_ARCHITECTURES=...` to the architecture(s) of your deployment GPUs;
the default is `80` (compute capability 8.0). These kernels require `80` or newer
because they use `__reduce_add_sync`.
The `CUDAARCHS` environment variable is also respected. Configure a separate build
directory when changing compilers, and use `CMAKE_CUDA_HOST_COMPILER` if nvcc needs
a specific host compiler.

## Link from another project

Use an installed package (set `CMAKE_PREFIX_PATH` to the installation prefix):

```cmake
find_package(FPcompress CONFIG REQUIRED)
target_link_libraries(my_program PRIVATE FPcompress::fpcompress)
```

Or include this repository directly:

```cmake
add_subdirectory(path/to/FPcompress)
target_link_libraries(my_program PRIVATE FPcompress::fpcompress)
```

## Device-buffer interface

Select `FPCOMPRESS_SP_SPEED`, `FPCOMPRESS_SP_RATIO`, `FPCOMPRESS_DP_SPEED`, or
`FPCOMPRESS_DP_RATIO`. Sizes are **bytes**, including for floating-point arrays.

1. Call `fpcompress_compress_bound` for the maximum encoded buffer size.
2. Query `fpcompress_cuda_compress_workspace_size` and
   `fpcompress_cuda_decompress_workspace_size`, passing the original byte count.
3. Allocate device input/output buffers, workspace, and a device `size_t` for the
   compressed size. `cudaMalloc` satisfies their alignment requirements.
4. Call `fpcompress_cuda_compress_async` on your stream. It writes the exact
   compressed byte count to the device size result. A stream-ordered copy and
   synchronization can make that count available on the host.
5. Call `fpcompress_cuda_decompress_async` with that encoded length and the
   original byte count. The original byte count must match the stream.

The library uses the caller's current CUDA device and stream. Calls allocate no
memory, perform no host transfers, and do not synchronize. Reuse workspace in
stream order; overlapping operations require separate workspaces and outputs.
Keep all buffers alive until the operation completes. Return values report
argument and launch errors; check stream completion for execution errors.

All input/output buffers must be 8-byte aligned; workspace must be 256-byte
aligned, and buffers must not overlap. Empty inputs are supported. DP_RATIO
requires a multiple of eight input bytes, expands its intermediate data to twice
the original size, and reserves preprocessing/sorting scratch accordingly. The
other codecs accept byte tails. The current kernels limit working data to
slightly below 2 GiB, so DP_RATIO's original input is limited to roughly 1 GiB.
Split larger inputs into independent calls.

Encoded data retains the original GPU codecs' native-endian format: a 32-bit
working-size header for SP_SPEED, SP_RATIO, and DP_SPEED, or a 64-bit header for
DP_RATIO; then 16-bit chunk lengths and chunk payloads. It does not identify the
codec. Retain the codec and original byte count with each stream. The helper
`fpcompress_decompressed_size` can inspect a **host copy** of encoded data and
validate its outer framing. Payload decoders expect trusted streams from the
matching codec; they are not validators for arbitrary or corrupted input.

# Artifact Appendix
### Abstract
This artifact contains the code and script to generate compression-ratio and throughput results for the 4 algorithms the paper introduces. The results should be similar to the numbers shown in Figures 8 through 19 for SPratio, SPspeed, DPratio, and DPspeed, that is, the compression ratios should match exactly but the compression and decompression throughputs are system dependent.

### Artifact check-list (meta-information)
  - Algorithm: SPratio, SPspeed, DPratio, and DPspeed
  - Compilation: g++ and nvcc
  - Data set: SDRBench
  - Hardware: CPU and GPU
  - Execution: Parallel
  - Metrics: Compression ratio and throughput
  - Output: Compression ratio vs. throughput scatter plots
  - How much disk space required (approximately)?: 100 GB
  - How much time is needed to prepare workflow (approximately)?: 180 minutes to download inputs
  - How much time is needed to complete experiments (approximately)?: 100 minutes for the GPU and 200 minutes for the CPU
  - Publicly available?: Yes
  - Code licenses (if publicly available)?: BSD 3-Clause License
  - Workflow automation framework used?: Python scripts
  - Archived (provide DOI)?: https://doi.org/10.5281/zenodo.14061031


### Description
###### How to access
The artifact can be found at https://github.com/burtscher/FPcompress.

###### Hardware dependencies
The hardware required for this artifact is an x86 multi-core CPU and a CUDA-capable GPU. We used a 32-core Intel Xeon Gold 6226R CPU @ 2.9 GHz with hyperthreading enabled to run the CPU codes. To run the GPU codes, we used an NVIDIA RTX 4090. Using similar hardware should result in throughputs similar to those reported in the paper.

###### Software dependencies
The required software includes:
- The computational artifact from https://github.com/burtscher/FPcompress
- GCC 7.5.0 or higher
- OpenMP 3.1 or higher
- CUDA 11.0 or higher
- Python v3.4 or higher
- Matplotlib v3.6 or higher


###### Data sets
The data sets used in the artifact are downloaded as part of the installation process and can be found at https://sdrbench.github.io.

###### Installation
To install the artifact
- Clone the repository from https://github.com/burtscher/FPcompress
- Run './compile.py' to compile SPratio, SPspeed, DPratio, and DPspeed

> **Note:**  
> If you encounter an error related to missing dependencies, add the flag `-ltbb` at the very end of the compilation instructions in `compile.py`.

###### Experiment workflow
1. Clone the repository from https://github.com/burtscher/FPcompress
2. Run './get\_inputs\_double.py' and './get\_inputs\_single.py' to download and set up the inputs used by the artifact
3. Run './compile.py' to compile SPratio, SPspeed, DPratio, and DPspeed
4. Run './run\_experiments\_double.py' and './run\_experiments\_single.py' to produce the intermediate experimental results
5. Run './chart\_double.py' and './chart\_single.py' to generate compression and decompression charts that look like Figures 8 through 19 but without the results for the third-party codes.
6. View the charts, which can be found in the current directory under 'double_charts.png' and 'single_charts.png'

###### Evaluation and expected results

The evaluation of the results is accomplished by comparing the figures generated using this artifact to the SPratio, SPspeed, DPratio, and DPspeed results listed in Figures 8 through 19. The absolute values of the throughputs and the relative positions may be different based on the CPU and GPU used, but the compression ratios should be the same.

###### Methodology

Submission, reviewing and badging methodology:
- https://www.acm.org/publications/policies/artifact-review-and-badging-current
- https://cTuning.org/ae
