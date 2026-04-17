#include <ATen/cuda/Exceptions.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <torch/all.h>
#include <torch/cuda.h>
#include <cuda_runtime.h>

namespace {

int64_t product_from_dim(const torch::Tensor& t, int64_t start_dim) {
  int64_t out = 1;
  for (int64_t i = start_dim; i < t.dim(); ++i) {
    out *= t.size(i);
  }
  return out;
}

int64_t infer_block_dim(const torch::Tensor& kv_cache, int64_t block_dim) {
  if (block_dim == 0 || block_dim == 1) {
    return block_dim;
  }
  TORCH_CHECK(false,
              "block_dim must be 0 or 1 for KV cache migration, got ",
              block_dim);
}

}  // namespace

// This function assumes that `cpu_tensor` is a CPU tensor allocated with pinned
// memory, and that UVA (Unified Virtual Addressing) is enabled.
torch::Tensor get_cuda_view_from_cpu_tensor(torch::Tensor& cpu_tensor) {
  TORCH_CHECK(cpu_tensor.device().is_cpu(), "Input tensor must be on CPU");

  // Get raw host pointer from CPU tensor
  void* host_ptr = cpu_tensor.data_ptr();

  // Get a device pointer corresponding to the pinned host memory
  void* device_ptr = nullptr;
  cudaError_t err = cudaHostGetDevicePointer(&device_ptr, host_ptr, 0);
  TORCH_CHECK(err == cudaSuccess,
              "cudaHostGetDevicePointer failed: ", cudaGetErrorString(err));

  auto sizes = cpu_tensor.sizes();
  auto strides = cpu_tensor.strides();
  auto options = cpu_tensor.options().device(torch::kCUDA);

  auto deleter = [](void*) {
    // no-op, since the memory is owned by the original CPU tensor
  };

  torch::Tensor cuda_tensor =
      torch::from_blob(device_ptr, sizes, strides, deleter, options);

  TORCH_CHECK(cuda_tensor.device().is_cuda(),
              "Resulting tensor is not on CUDA device");

  return cuda_tensor;
}

torch::Tensor get_cuda_view_from_ptr_like(int64_t device_ptr,
                                          torch::Tensor& like_tensor) {
  TORCH_CHECK(like_tensor.device().is_cuda(),
              "like_tensor must be on CUDA device");
  TORCH_CHECK(device_ptr != 0, "device_ptr must be non-zero");

  void* ptr = reinterpret_cast<void*>(device_ptr);
  auto sizes = like_tensor.sizes();
  auto strides = like_tensor.strides();
  auto options = like_tensor.options();

  auto deleter = [](void*) {
    // no-op: ownership is managed by CUDA IPC handle lifetime.
  };

  return torch::from_blob(ptr, sizes, strides, deleter, options);
}

torch::Tensor get_cuda_view_from_ptr_shape_stride(
    int64_t device_ptr, const std::vector<int64_t>& sizes,
    const std::vector<int64_t>& strides, torch::Tensor& like_tensor) {
  TORCH_CHECK(like_tensor.device().is_cuda(),
              "like_tensor must be on CUDA device");
  TORCH_CHECK(device_ptr != 0, "device_ptr must be non-zero");
  TORCH_CHECK(!sizes.empty(), "sizes must be non-empty");
  TORCH_CHECK(sizes.size() == strides.size(),
              "sizes and strides must have the same length");

  for (size_t i = 0; i < sizes.size(); ++i) {
    TORCH_CHECK(sizes[i] >= 0, "size must be non-negative at dim ", i,
                ", got ", sizes[i]);
    TORCH_CHECK(strides[i] >= 0, "stride must be non-negative at dim ", i,
                ", got ", strides[i]);
  }

  void* ptr = reinterpret_cast<void*>(device_ptr);
  auto options = like_tensor.options();

  auto deleter = [](void*) {
    // no-op: ownership is managed by CUDA IPC handle lifetime.
  };

  torch::Tensor out = torch::from_blob(ptr, sizes, strides, deleter, options);
  TORCH_CHECK(out.device().is_cuda(),
              "Resulting tensor is not on CUDA device");
  return out;
}

void migrate_kv_cache_blocks(torch::Tensor& dst_cache,
                             torch::Tensor& src_cache,
                             torch::Tensor& src_block_ids,
                             torch::Tensor& dst_block_ids,
                             int64_t block_dim) {
  TORCH_CHECK(dst_cache.is_cuda() && src_cache.is_cuda(),
              "migrate_kv_cache_blocks expects CUDA tensors");
  TORCH_CHECK(dst_cache.scalar_type() == src_cache.scalar_type(),
              "src and dst dtypes must match");
  TORCH_CHECK(dst_cache.dim() == src_cache.dim(),
              "src and dst cache dims must match");
  TORCH_CHECK(dst_cache.dim() >= 3,
              "KV cache dim must be >= 3, got ", dst_cache.dim());
  TORCH_CHECK(src_block_ids.numel() == dst_block_ids.numel(),
              "src_block_ids and dst_block_ids must have same length");
  TORCH_CHECK(src_block_ids.scalar_type() == torch::kInt64,
              "src_block_ids must be int64");
  TORCH_CHECK(dst_block_ids.scalar_type() == torch::kInt64,
              "dst_block_ids must be int64");

  block_dim = infer_block_dim(src_cache, block_dim);

  auto src_ids = src_block_ids.to(torch::kCPU).contiguous();
  auto dst_ids = dst_block_ids.to(torch::kCPU).contiguous();
  const int64_t* src_ids_ptr = src_ids.data_ptr<int64_t>();
  const int64_t* dst_ids_ptr = dst_ids.data_ptr<int64_t>();
  const int64_t num_pairs = src_ids.numel();
  if (num_pairs == 0) {
    return;
  }

  const auto elem_size = static_cast<int64_t>(dst_cache.element_size());
  const int64_t src_num_blocks = src_cache.size(block_dim);
  const int64_t dst_num_blocks = dst_cache.size(block_dim);

  const int64_t row_dim = (block_dim == 0) ? 1 : 0;
  const int64_t rows = src_cache.size(row_dim);
  TORCH_CHECK(rows > 0, "rows must be positive for block migration");
  TORCH_CHECK(dst_cache.size(row_dim) == rows,
              "src and dst row dims must match: src rows=",
              rows,
              ", dst rows=",
              dst_cache.size(row_dim));

  for (int64_t dim = 0; dim < src_cache.dim(); ++dim) {
    if (dim == block_dim) {
      continue;
    }
    TORCH_CHECK(src_cache.size(dim) == dst_cache.size(dim),
                "src/dst cache shape mismatch at dim ",
                dim,
                ": src=",
                src_cache.size(dim),
                ", dst=",
                dst_cache.size(dim));
  }

  // Each row copies all trailing dimensions after [row_dim, block_dim].
  const int64_t src_width_elems = product_from_dim(src_cache, 2);
  const int64_t dst_width_elems = product_from_dim(dst_cache, 2);
  TORCH_CHECK(src_width_elems > 0,
              "width in elements must be positive for block migration");
  TORCH_CHECK(src_width_elems == dst_width_elems,
              "src and dst width elements must match, src=",
              src_width_elems,
              ", dst=",
              dst_width_elems);

  const size_t width_bytes = static_cast<size_t>(src_width_elems * elem_size);
  const size_t src_pitch =
      static_cast<size_t>(src_cache.stride(row_dim) * elem_size);
  const size_t dst_pitch =
      static_cast<size_t>(dst_cache.stride(row_dim) * elem_size);
  const int64_t src_block_stride = src_cache.stride(block_dim);
  const int64_t dst_block_stride = dst_cache.stride(block_dim);
  TORCH_CHECK(src_cache.stride(row_dim) > 0 && dst_cache.stride(row_dim) > 0,
              "row stride must be positive for block migration, src=",
              src_cache.stride(row_dim),
              ", dst=",
              dst_cache.stride(row_dim));
  TORCH_CHECK(src_block_stride > 0 && dst_block_stride > 0,
              "block stride must be positive for block migration, src=",
              src_block_stride,
              ", dst=",
              dst_block_stride);
  TORCH_CHECK(width_bytes <= src_pitch,
              "invalid src pitch/width for block migration: width_bytes=",
              width_bytes,
              ", src_pitch=",
              src_pitch);
  TORCH_CHECK(width_bytes <= dst_pitch,
              "invalid dst pitch/width for block migration: width_bytes=",
              width_bytes,
              ", dst_pitch=",
              dst_pitch);

  const at::cuda::OptionalCUDAGuard device_guard(device_of(dst_cache));
  auto stream = c10::cuda::getCurrentCUDAStream().stream();

  char* src_base = static_cast<char*>(src_cache.data_ptr());
  char* dst_base = static_cast<char*>(dst_cache.data_ptr());

  for (int64_t i = 0; i < num_pairs; ++i) {
    const int64_t src_block = src_ids_ptr[i];
    const int64_t dst_block = dst_ids_ptr[i];
    TORCH_CHECK(src_block >= 0 && src_block < src_num_blocks,
                "src block id out of range: ",
                src_block,
                " / ",
                src_num_blocks);
    TORCH_CHECK(dst_block >= 0 && dst_block < dst_num_blocks,
                "dst block id out of range: ",
                dst_block,
                " / ",
                dst_num_blocks);

    char* src_ptr = src_base + src_block * src_block_stride * elem_size;
    char* dst_ptr = dst_base + dst_block * dst_block_stride * elem_size;

    AT_CUDA_CHECK(cudaMemcpy2DAsync(dst_ptr,
                                    dst_pitch,
                                    src_ptr,
                                    src_pitch,
                                    width_bytes,
                                    static_cast<size_t>(rows),
                                    cudaMemcpyDeviceToDevice,
                                    stream));
  }
}
