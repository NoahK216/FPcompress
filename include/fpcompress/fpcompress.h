#ifndef FPCOMPRESS_H
#define FPCOMPRESS_H

#include <cuda_runtime_api.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum fpcompress_codec {
  FPCOMPRESS_SP_SPEED = 0,
  FPCOMPRESS_SP_RATIO = 1,
  FPCOMPRESS_DP_SPEED = 2,
  FPCOMPRESS_DP_RATIO = 3
} fpcompress_codec;

typedef enum fpcompress_status {
  FPCOMPRESS_SUCCESS = 0,
  FPCOMPRESS_INVALID_ARGUMENT,
  FPCOMPRESS_BUFFER_TOO_SMALL,
  FPCOMPRESS_INVALID_DATA,
  FPCOMPRESS_OUT_OF_MEMORY,
  FPCOMPRESS_INTERNAL_ERROR
} fpcompress_status;

const char *fpcompress_status_string(fpcompress_status status);

/* All lengths are bytes. DP_RATIO requires a multiple of 8 input bytes.
 * The current implementation limits the codec's working data to slightly below
 * 2 GiB (DP_RATIO working data is twice the original size). Encoded streams use
 * a native-endian header and chunk format and do not identify their codec.
 * Retain the codec alongside the encoded stream.
 */
fpcompress_status fpcompress_compress_bound(fpcompress_codec codec,
                                            size_t input_bytes,
                                            size_t *bound_bytes);

/* Reads and checks the outer header and chunk lengths. Decoders are intended
 * for trusted streams produced by the matching codec; payloads are not fully
 * validated. The returned size excludes DP_RATIO's internal expansion.
 */
fpcompress_status fpcompress_decompressed_size(fpcompress_codec codec,
                                               const void *input,
                                               size_t input_bytes,
                                               size_t *output_bytes);

/* Host queries. original_bytes is the size BEFORE DP_RATIO preprocessing.
 * Workspace is device memory aligned to 256 bytes (cudaMalloc satisfies this).
 */
cudaError_t fpcompress_cuda_compress_workspace_size(fpcompress_codec codec,
                                                    size_t original_bytes,
                                                    size_t *workspace_bytes);
cudaError_t fpcompress_cuda_decompress_workspace_size(fpcompress_codec codec,
                                                      size_t original_bytes,
                                                      size_t *workspace_bytes);

/* Asynchronous, on the caller's current device and stream. All buffer pointers
 * (including compressed_bytes) are device pointers; host size arguments are
 * byte counts. Input/output must be 8-byte aligned. Buffers and workspace must
 * be disjoint and live until the stream completes. Each overlapping call needs
 * its own workspace, output, and size result. No allocations, host transfers,
 * device selection, or synchronization occur inside these calls.
 *
 * compressed_bytes is a device size_t written by compression. Synchronize or
 * use a stream-ordered copy before reading it. Return values report argument
 * and launch failures; execution errors arrive when synchronizing the stream.
 */
cudaError_t fpcompress_cuda_compress_async(
    fpcompress_codec codec, const void *input, size_t input_bytes, void *output,
    size_t output_capacity, size_t *compressed_bytes, void *workspace,
    size_t workspace_bytes, cudaStream_t stream);

/* Decode only trusted streams from the same codec. original_bytes MUST match
 * the stream's uncompressed size. input_bytes is the exact compressed size.
 * No device-header validation or host readback is performed here.
 */
cudaError_t fpcompress_cuda_decompress_async(
    fpcompress_codec codec, const void *input, size_t input_bytes,
    size_t original_bytes, void *output, size_t output_capacity,
    void *workspace, size_t workspace_bytes, cudaStream_t stream);

#ifdef __cplusplus
}
#endif
#endif
