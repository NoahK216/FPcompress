#pragma once
#include <cuda_runtime.h>
#include <fpcompress/fpcompress.h>

namespace fpcompress_detail {

inline size_t pad(size_t bytes) { return (bytes + 255) & ~size_t(255); }

inline cudaError_t launch_blocks(int *blocks) {
  int device;
  cudaDeviceProp properties;
  cudaError_t error = cudaGetDevice(&device);
  if (error != cudaSuccess)
    return error;
  error = cudaGetDeviceProperties(&properties, device);
  if (error != cudaSuccess)
    return error;
  *blocks = properties.multiProcessorCount *
            (properties.maxThreadsPerMultiProcessor / 512);
  return *blocks > 0 ? cudaSuccess : cudaErrorInvalidConfiguration;
}

template <class T>
static __global__ void publish_size(const T *size, size_t *result) {
  *result = static_cast<size_t>(*size);
}

// Each launch function is defined beside its kernel in the original .cu file.
cudaError_t sp_speed_encode(const unsigned char *input, size_t bytes,
                            unsigned char *output, size_t *output_bytes,
                            void *workspace, cudaStream_t stream);
cudaError_t sp_ratio_encode(const unsigned char *input, size_t bytes,
                            unsigned char *output, size_t *output_bytes,
                            void *workspace, cudaStream_t stream);
cudaError_t dp_speed_encode(const unsigned char *input, size_t bytes,
                            unsigned char *output, size_t *output_bytes,
                            void *workspace, cudaStream_t stream);
cudaError_t dp_ratio_encode(const unsigned char *input, size_t bytes,
                            unsigned char *output, size_t *output_bytes,
                            void *workspace, cudaStream_t stream);

cudaError_t sp_speed_decode(const unsigned char *input, size_t bytes,
                            unsigned char *output, void *workspace,
                            cudaStream_t stream);
cudaError_t sp_ratio_decode(const unsigned char *input, size_t bytes,
                            unsigned char *output, void *workspace,
                            cudaStream_t stream);
cudaError_t dp_speed_decode(const unsigned char *input, size_t bytes,
                            unsigned char *output, void *workspace,
                            cudaStream_t stream);
cudaError_t dp_ratio_decode(const unsigned char *input, size_t bytes,
                            unsigned char *output, void *workspace,
                            cudaStream_t stream);

cudaError_t dp_ratio_compress_workspace_size(size_t bytes, size_t *size);

} // namespace fpcompress_detail
