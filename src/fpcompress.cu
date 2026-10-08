#include "fpcompress_internal.cuh"
#include <algorithm>
#include <climits>
#include <cstdint>
#include <cstring>

using namespace fpcompress_detail;
namespace {
constexpr size_t chunk_size = 16384;
inline bool valid_codec(fpcompress_codec c) {
  return c >= FPCOMPRESS_SP_SPEED && c <= FPCOMPRESS_DP_RATIO;
}
inline size_t header_size(fpcompress_codec c) {
  return c == FPCOMPRESS_DP_RATIO ? 8 : 4;
}
inline bool valid_size(fpcompress_codec c, size_t n) {
  if (!valid_codec(c))
    return false;
  const size_t limit =
      (static_cast<size_t>(INT_MAX) / (chunk_size + 2) - 1) * chunk_size;
  return c == FPCOMPRESS_DP_RATIO ? n % 8 == 0 && n <= limit / 2 : n <= limit;
}
inline size_t working_size(fpcompress_codec c, size_t n) {
  return c == FPCOMPRESS_DP_RATIO ? 2 * n : n;
}
inline bool aligned(const void *p) {
  return reinterpret_cast<uintptr_t>(p) % 8 == 0;
}
static __global__ void empty_stream(unsigned char *output, size_t header,
                                    size_t *result) {
  for (size_t i = 0; i < header; ++i)
    output[i] = 0;
  *result = header;
}
} // namespace

extern "C" const char *fpcompress_status_string(fpcompress_status status) {
  switch (status) {
  case FPCOMPRESS_SUCCESS:
    return "success";
  case FPCOMPRESS_INVALID_ARGUMENT:
    return "invalid argument";
  case FPCOMPRESS_BUFFER_TOO_SMALL:
    return "buffer too small";
  case FPCOMPRESS_INVALID_DATA:
    return "invalid encoded data";
  case FPCOMPRESS_OUT_OF_MEMORY:
    return "out of memory";
  default:
    return "internal codec error";
  }
}

extern "C" fpcompress_status
fpcompress_compress_bound(fpcompress_codec codec, size_t bytes, size_t *bound) {
  if (!bound)
    return FPCOMPRESS_INVALID_ARGUMENT;
  *bound = 0;
  if (!valid_size(codec, bytes))
    return FPCOMPRESS_INVALID_ARGUMENT;
  const size_t work = working_size(codec, bytes);
  *bound =
      header_size(codec) + 2 * ((work + chunk_size - 1) / chunk_size) + work;
  return FPCOMPRESS_SUCCESS;
}

extern "C" fpcompress_status
fpcompress_decompressed_size(fpcompress_codec codec, const void *input,
                             size_t bytes, size_t *size) {
  if (!size)
    return FPCOMPRESS_INVALID_ARGUMENT;
  *size = 0;
  if (!valid_codec(codec) || !input)
    return FPCOMPRESS_INVALID_ARGUMENT;
  const size_t header = header_size(codec);
  if (bytes < header)
    return FPCOMPRESS_INVALID_DATA;
  uint64_t work = 0;
  std::memcpy(&work, input, header);
  if (work > static_cast<uint64_t>(INT_MAX) - chunk_size ||
      (codec == FPCOMPRESS_DP_RATIO && work % 16))
    return FPCOMPRESS_INVALID_DATA;
  const size_t chunks = (work + chunk_size - 1) / chunk_size;
  size_t offset = header + 2 * chunks;
  if (offset > bytes)
    return FPCOMPRESS_INVALID_DATA;
  const auto *data = static_cast<const unsigned char *>(input);
  for (size_t i = 0; i < chunks; ++i) {
    uint16_t length;
    std::memcpy(&length, data + header + 2 * i, 2);
    if (!length ||
        length > std::min<uint64_t>(chunk_size, work - i * chunk_size) ||
        length > bytes - offset)
      return FPCOMPRESS_INVALID_DATA;
    offset += length;
  }
  if (offset != bytes)
    return FPCOMPRESS_INVALID_DATA;
  *size = codec == FPCOMPRESS_DP_RATIO ? work / 2 : work;
  return FPCOMPRESS_SUCCESS;
}

namespace {
using Encoder = cudaError_t (*)(const unsigned char *, size_t, unsigned char *,
                                size_t *, void *, cudaStream_t);
using Decoder = cudaError_t (*)(const unsigned char *, size_t, unsigned char *,
                                void *, cudaStream_t);
const Encoder encoders[] = {sp_speed_encode, sp_ratio_encode, dp_speed_encode,
                            dp_ratio_encode};
const Decoder decoders[] = {sp_speed_decode, sp_ratio_decode, dp_speed_decode,
                            dp_ratio_decode};
bool workspace_valid(void *workspace, size_t bytes, size_t needed) {
  return workspace && reinterpret_cast<uintptr_t>(workspace) % 256 == 0 &&
         bytes >= needed;
}
bool overlap(const void *a, size_t an, const void *b, size_t bn) {
  if (!an || !bn)
    return false;
  const uintptr_t av = reinterpret_cast<uintptr_t>(a),
                  bv = reinterpret_cast<uintptr_t>(b);
  return av <= bv ? bv - av < an : av - bv < bn;
}
bool buffers_valid(const void *input, size_t bytes, void *output,
                   size_t capacity, void *workspace, size_t workspace_bytes) {
  return !overlap(input, bytes, output, capacity) &&
         !overlap(input, bytes, workspace, workspace_bytes) &&
         !overlap(output, capacity, workspace, workspace_bytes);
}
} // namespace

extern "C" cudaError_t
fpcompress_cuda_compress_workspace_size(fpcompress_codec codec, size_t bytes,
                                        size_t *size) {
  if (!size)
    return cudaErrorInvalidValue;
  *size = 0;
  if (!valid_size(codec, bytes))
    return cudaErrorInvalidValue;
  if (codec == FPCOMPRESS_DP_RATIO)
    return dp_ratio_compress_workspace_size(bytes, size);
  const size_t chunks = (bytes + chunk_size - 1) / chunk_size;
  *size = 256 + pad(chunks * sizeof(int));
  return cudaSuccess;
}

extern "C" cudaError_t
fpcompress_cuda_decompress_workspace_size(fpcompress_codec codec, size_t bytes,
                                          size_t *size) {
  if (!size)
    return cudaErrorInvalidValue;
  *size = 0;
  if (!valid_size(codec, bytes))
    return cudaErrorInvalidValue;
  *size = codec == FPCOMPRESS_DP_RATIO ? 256 + pad(2 * bytes) : 256;
  return cudaSuccess;
}

extern "C" cudaError_t
fpcompress_cuda_compress_async(fpcompress_codec codec, const void *input,
                               size_t bytes, void *output, size_t capacity,
                               size_t *result, void *workspace,
                               size_t workspace_bytes, cudaStream_t stream) {
  size_t needed, bound;
  cudaError_t error =
      fpcompress_cuda_compress_workspace_size(codec, bytes, &needed);
  if (error != cudaSuccess)
    return error;
  if (fpcompress_compress_bound(codec, bytes, &bound) != FPCOMPRESS_SUCCESS ||
      !output || !aligned(output) || !result || !aligned(result) ||
      capacity < bound || (bytes && (!input || !aligned(input))) ||
      !workspace_valid(workspace, workspace_bytes, needed) ||
      !buffers_valid(input, bytes, output, capacity, workspace,
                     workspace_bytes) ||
      overlap(result, sizeof(*result), input, bytes) ||
      overlap(result, sizeof(*result), output, capacity) ||
      overlap(result, sizeof(*result), workspace, workspace_bytes))
    return cudaErrorInvalidValue;
  if (!bytes) {
    empty_stream<<<1, 1, 0, stream>>>(static_cast<unsigned char *>(output),
                                      header_size(codec), result);
    return cudaGetLastError();
  }
  return encoders[codec](static_cast<const unsigned char *>(input), bytes,
                         static_cast<unsigned char *>(output), result,
                         workspace, stream);
}

extern "C" cudaError_t
fpcompress_cuda_decompress_async(fpcompress_codec codec, const void *input,
                                 size_t bytes, size_t original, void *output,
                                 size_t capacity, void *workspace,
                                 size_t workspace_bytes, cudaStream_t stream) {
  size_t needed;
  cudaError_t error =
      fpcompress_cuda_decompress_workspace_size(codec, original, &needed);
  if (error != cudaSuccess)
    return error;
  const size_t work = working_size(codec, original);
  const size_t chunks = (work + chunk_size - 1) / chunk_size;
  if (!input || !aligned(input) ||
      bytes < header_size(codec) + 2 * chunks + chunks ||
      (original && (!output || !aligned(output))) || capacity < original ||
      !workspace_valid(workspace, workspace_bytes, needed) ||
      !buffers_valid(input, bytes, output, capacity, workspace,
                     workspace_bytes))
    return cudaErrorInvalidValue;
  if (!original)
    return cudaSuccess;
  return decoders[codec](static_cast<const unsigned char *>(input), original,
                         static_cast<unsigned char *>(output), workspace,
                         stream);
}
