// SPDX-License-Identifier: Apache-2.0
// Fused multi-layer block-scatter CUDA op for the round-kv arena load.
//
// The arena load scatters a chunk's blocks into the paged KV cache.  The naive
// path issues nL (=32) `index_put`s PER CHUNK; in the busy serving process the
// many small CPU-dispatched launches starve the GPU (it idles waiting for the
// next launch under GIL contention) -> ~1.5 GB/s even though each kernel is
// fast.  This op does the WHOLE chunk's nL layers in ONE launch (and releases
// the GIL during execution), cutting per-chunk dispatches nL->1.
//
// Layout (matches round_kv_store block-major staging):
//   staging    : [nb, nL, 2, *rest] contiguous, 2-byte dtype (fp16/bf16)
//   idx        : [nb] int64  destination block ids
//   layer_ptrs : [nL] int64  each = a paged KV layer tensor's data_ptr()
//   FA  (dim==1, layer [2, NBLK, *rest]):  dst off = (kv*NBLK + blk)*P + r
//   MLA (dim==0, layer [NBLK, 2, *rest]):  dst off = (blk*2  + kv)*P + r
//   P = prod(rest); requires P % 8 == 0 (int4-vectorized).
//
// Each CUDA block copies one contiguous (j-block, li-layer, kv) run of P 16-bit
// elements (as int4 = 8 halves) into the layer's scattered block slot idx[j].
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime.h>

__global__ void licht_scatter_kernel(
    const uint16_t* __restrict__ staging,
    const int64_t*  __restrict__ idx,
    const int64_t*  __restrict__ layer_ptrs,
    int nb, int nL, int dim, long NBLK, long P)
{
    long total_runs = (long)nb * nL * 2;
    long P8 = P >> 3;                       // # int4 (8 halves) per run
    for (long run = blockIdx.x; run < total_runs; run += gridDim.x) {
        int  kv = (int)(run & 1L);
        long m  = run >> 1;
        int  li = (int)(m % nL);
        long j  = m / nL;
        long blk = idx[j];
        const uint16_t* src = staging + (((j * (long)nL + li) * 2 + kv) * P);
        uint16_t* dstbase = (uint16_t*)(layer_ptrs[li]);
        long dstoff = (dim == 1) ? ((kv * NBLK + blk) * P) : ((blk * 2 + kv) * P);
        const int4* s4 = (const int4*)src;
        int4* d4 = (int4*)(dstbase + dstoff);
        for (long r = threadIdx.x; r < P8; r += blockDim.x) d4[r] = s4[r];
    }
}

// ============================================================
// DIRECT arena -> paged scatter (NO GPU staging buffer)
// ============================================================
// Stage 5 / LRU direct load: the source is the cudaHostRegister'd shared
// arena (host pinned).  On UVA platforms (all 64-bit Linux, compute >= 2.0)
// a host pointer registered with cudaHostRegisterDefault is directly
// addressable from device code over PCIe.  The kernel reads each block's KV
// straight from host pinned memory and scatters it into the paged buffer in
// ONE launch — no intermediate GPU staging, no extra HBM round-trip.
//
//   arena_host : base host pointer of the registered arena (uint16 elems)
//   src_slots  : [nb] int64  physical arena slot id per block
//   dst_idx    : [nb] int64  destination paged block id per block
//   layer_ptrs : [nL] int64  each = a paged KV layer tensor's data_ptr()
//   arena slot layout: [num_slots, nL, 2, *rest], so block (slot,li,kv) is at
//     arena_host + ((slot*nL + li)*2 + kv)*P     (element offset, P=prod(rest))
//   dst layout identical to licht_scatter_kernel.
__global__ void licht_scatter_from_arena_kernel(
    const uint16_t* __restrict__ arena_host,
    const int64_t*  __restrict__ src_slots,
    const int64_t*  __restrict__ dst_idx,
    const int64_t*  __restrict__ layer_ptrs,
    int nb, int nL, int dim, long NBLK, long P)
{
    long total_runs = (long)nb * nL * 2;
    long P8 = P >> 3;                       // # int4 (8 halves) per run
    for (long run = blockIdx.x; run < total_runs; run += gridDim.x) {
        int  kv   = (int)(run & 1L);
        long m    = run >> 1;
        int  li   = (int)(m % nL);
        long j    = m / nL;
        long slot = src_slots[j];           // arena physical slot
        long blk  = dst_idx[j];             // dest paged block
        const uint16_t* src = arena_host
            + (((slot * (long)nL + li) * 2 + kv) * P);
        uint16_t* dstbase = (uint16_t*)(layer_ptrs[li]);
        long dstoff = (dim == 1) ? ((kv * NBLK + blk) * P)
                                 : ((blk * 2 + kv) * P);
        const int4* s4 = (const int4*)src;
        int4* d4 = (int4*)(dstbase + dstoff);
        for (long r = threadIdx.x; r < P8; r += blockDim.x) d4[r] = s4[r];
    }
}

void licht_scatter_from_arena(int64_t arena_host_ptr,
                              torch::Tensor src_slots, torch::Tensor dst_idx,
                              torch::Tensor layer_ptrs, int64_t nb, int64_t nL,
                              int64_t dim, int64_t NBLK, int64_t P) {
    TORCH_CHECK(arena_host_ptr != 0,
                "licht_scatter_from_arena: arena_host_ptr is null");
    TORCH_CHECK(src_slots.is_cuda() && dst_idx.is_cuda() && layer_ptrs.is_cuda(),
                "licht_scatter_from_arena: src_slots/dst_idx/layer_ptrs must be CUDA");
    TORCH_CHECK(src_slots.scalar_type() == at::kLong,
                "licht_scatter_from_arena: src_slots must be int64");
    TORCH_CHECK(dst_idx.scalar_type() == at::kLong,
                "licht_scatter_from_arena: dst_idx must be int64");
    TORCH_CHECK(layer_ptrs.scalar_type() == at::kLong,
                "licht_scatter_from_arena: layer_ptrs must be int64");
    TORCH_CHECK((P & 7) == 0,
                "licht_scatter_from_arena: P must be a multiple of 8");
    TORCH_CHECK(src_slots.numel() >= nb && dst_idx.numel() >= nb,
                "licht_scatter_from_arena: src_slots/dst_idx too small");
    TORCH_CHECK(layer_ptrs.numel() >= nL,
                "licht_scatter_from_arena: layer_ptrs too small");
    long total_runs = nb * nL * 2;
    if (total_runs < 1) return;
    int  threads = 256;
    long blocks  = total_runs < 65535 ? total_runs : 65535;
    const at::cuda::CUDAGuard device_guard(layer_ptrs.device());
    auto stream  = at::cuda::getCurrentCUDAStream();
    licht_scatter_from_arena_kernel<<<blocks, threads, 0, stream>>>(
        reinterpret_cast<const uint16_t*>(arena_host_ptr),
        src_slots.data_ptr<int64_t>(), dst_idx.data_ptr<int64_t>(),
        layer_ptrs.data_ptr<int64_t>(), (int)nb, (int)nL, (int)dim, NBLK, P);
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "licht_scatter_from_arena launch failed: ",
                cudaGetErrorString(err));
}

// ── Per-layer 变体 (流水线逐层加载) ──────────────────────────────────────
// 只 scatter 单层 layer_idx 的 block: arena[(slot*nL+layer_idx)*2+kv] -> 单层 paged.
// nL 仍传真实层数 (算 slot stride), layer_ptr 是该层 paged tensor 的 data_ptr.
// grid 循环 nb*2 (kv*nb), 不乘 nL. 与批量 kernel 第 layer_idx 层切片结果一致.
__global__ void licht_scatter_from_arena_layer_kernel(
    const uint16_t* __restrict__ arena_host,
    const int64_t*  __restrict__ src_slots,
    const int64_t*  __restrict__ dst_idx,
    int64_t layer_ptr,
    int nb, int nL, int layer_idx, int dim, long NBLK, long P)
{
    long total_runs = (long)nb * 2;
    long P8 = P >> 3;
    uint16_t* dstbase = (uint16_t*)layer_ptr;
    for (long run = blockIdx.x; run < total_runs; run += gridDim.x) {
        int  kv   = (int)(run & 1L);
        long j    = run >> 1;
        long slot = src_slots[j];
        long blk  = dst_idx[j];
        const uint16_t* src = arena_host
            + (((slot * (long)nL + layer_idx) * 2 + kv) * P);
        long dstoff = (dim == 1) ? ((kv * NBLK + blk) * P)
                                 : ((blk * 2 + kv) * P);
        const int4* s4 = (const int4*)src;
        int4* d4 = (int4*)(dstbase + dstoff);
        for (long r = threadIdx.x; r < P8; r += blockDim.x) d4[r] = s4[r];
    }
}

void licht_scatter_from_arena_layer(int64_t arena_host_ptr,
                                    torch::Tensor src_slots,
                                    torch::Tensor dst_idx, int64_t layer_ptr,
                                    int64_t nb, int64_t nL, int64_t layer_idx,
                                    int64_t dim, int64_t NBLK, int64_t P) {
    TORCH_CHECK(arena_host_ptr != 0,
                "licht_scatter_from_arena_layer: arena_host_ptr is null");
    TORCH_CHECK(layer_ptr != 0,
                "licht_scatter_from_arena_layer: layer_ptr is null");
    TORCH_CHECK(src_slots.is_cuda() && dst_idx.is_cuda(),
                "licht_scatter_from_arena_layer: src_slots/dst_idx must be CUDA");
    TORCH_CHECK(src_slots.scalar_type() == at::kLong,
                "licht_scatter_from_arena_layer: src_slots must be int64");
    TORCH_CHECK(dst_idx.scalar_type() == at::kLong,
                "licht_scatter_from_arena_layer: dst_idx must be int64");
    TORCH_CHECK((P & 7) == 0,
                "licht_scatter_from_arena_layer: P must be a multiple of 8");
    TORCH_CHECK(src_slots.numel() >= nb && dst_idx.numel() >= nb,
                "licht_scatter_from_arena_layer: src_slots/dst_idx too small");
    TORCH_CHECK(layer_idx >= 0 && layer_idx < nL,
                "licht_scatter_from_arena_layer: layer_idx out of range");
    long total_runs = nb * 2;
    if (total_runs < 1) return;
    int  threads = 256;
    long blocks  = total_runs < 65535 ? total_runs : 65535;
    const at::cuda::CUDAGuard device_guard(src_slots.device());
    auto stream  = at::cuda::getCurrentCUDAStream();
    licht_scatter_from_arena_layer_kernel<<<blocks, threads, 0, stream>>>(
        reinterpret_cast<const uint16_t*>(arena_host_ptr),
        src_slots.data_ptr<int64_t>(), dst_idx.data_ptr<int64_t>(),
        layer_ptr, (int)nb, (int)nL, (int)layer_idx, (int)dim, NBLK, P);
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess,
                "licht_scatter_from_arena_layer launch failed: ",
                cudaGetErrorString(err));
}

// ── Per-layer 直写 (store: GPU paged -> arena host) ──────────────────────
// licht_scatter_from_arena_layer 的镜像, 方向反过来: src=单层 paged 块 (src_idx),
// dst=arena host slot (dst_slots). GPU kernel 经 PCIe 直写 cudaHostRegister 的 arena,
// 省掉 D2H gather + CPU memcpy. 索引公式与读版完全一致 (arena/paged 布局不变).
__global__ void licht_gather_to_arena_layer_kernel(
    uint16_t* __restrict__ arena_host,        // dst (host pinned, 可写)
    const int64_t*  __restrict__ dst_slots,   // arena 物理 slot per block
    const int64_t*  __restrict__ src_idx,     // 源 paged block id per block
    int64_t layer_ptr,                        // 单层 paged tensor data_ptr (src)
    int nb, int nL, int layer_idx, int dim, long NBLK, long P)
{
    long total_runs = (long)nb * 2;
    long P8 = P >> 3;
    const uint16_t* srcbase = (const uint16_t*)layer_ptr;
    for (long run = blockIdx.x; run < total_runs; run += gridDim.x) {
        int  kv   = (int)(run & 1L);
        long j    = run >> 1;
        long slot = dst_slots[j];           // arena dst slot
        long blk  = src_idx[j];             // paged src block
        uint16_t* dst = arena_host
            + (((slot * (long)nL + layer_idx) * 2 + kv) * P);
        long srcoff = (dim == 1) ? ((kv * NBLK + blk) * P)
                                 : ((blk * 2 + kv) * P);
        const int4* s4 = (const int4*)(srcbase + srcoff);
        int4* d4 = (int4*)dst;
        for (long r = threadIdx.x; r < P8; r += blockDim.x) d4[r] = s4[r];
    }
}

void licht_gather_to_arena_layer(int64_t arena_host_ptr,
                                 torch::Tensor dst_slots,
                                 torch::Tensor src_idx, int64_t layer_ptr,
                                 int64_t nb, int64_t nL, int64_t layer_idx,
                                 int64_t dim, int64_t NBLK, int64_t P) {
    TORCH_CHECK(arena_host_ptr != 0,
                "licht_gather_to_arena_layer: arena_host_ptr is null");
    TORCH_CHECK(layer_ptr != 0,
                "licht_gather_to_arena_layer: layer_ptr is null");
    TORCH_CHECK(dst_slots.is_cuda() && src_idx.is_cuda(),
                "licht_gather_to_arena_layer: dst_slots/src_idx must be CUDA");
    TORCH_CHECK(dst_slots.scalar_type() == at::kLong,
                "licht_gather_to_arena_layer: dst_slots must be int64");
    TORCH_CHECK(src_idx.scalar_type() == at::kLong,
                "licht_gather_to_arena_layer: src_idx must be int64");
    TORCH_CHECK((P & 7) == 0,
                "licht_gather_to_arena_layer: P must be a multiple of 8");
    TORCH_CHECK(dst_slots.numel() >= nb && src_idx.numel() >= nb,
                "licht_gather_to_arena_layer: dst_slots/src_idx too small");
    TORCH_CHECK(layer_idx >= 0 && layer_idx < nL,
                "licht_gather_to_arena_layer: layer_idx out of range");
    long total_runs = nb * 2;
    if (total_runs < 1) return;
    int  threads = 256;
    long blocks  = total_runs < 65535 ? total_runs : 65535;
    const at::cuda::CUDAGuard device_guard(dst_slots.device());
    auto stream  = at::cuda::getCurrentCUDAStream();
    licht_gather_to_arena_layer_kernel<<<blocks, threads, 0, stream>>>(
        reinterpret_cast<uint16_t*>(arena_host_ptr),
        dst_slots.data_ptr<int64_t>(), src_idx.data_ptr<int64_t>(),
        layer_ptr, (int)nb, (int)nL, (int)layer_idx, (int)dim, NBLK, P);
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess,
                "licht_gather_to_arena_layer launch failed: ",
                cudaGetErrorString(err));
}

void licht_scatter(torch::Tensor staging, torch::Tensor idx,
                   torch::Tensor layer_ptrs, int64_t nb, int64_t nL,
                   int64_t dim, int64_t NBLK, int64_t P) {
    TORCH_CHECK(staging.is_cuda() && idx.is_cuda() && layer_ptrs.is_cuda(),
                "licht_scatter: all tensors must be CUDA");
    TORCH_CHECK(staging.is_contiguous(), "licht_scatter: staging must be contiguous");
    TORCH_CHECK(idx.scalar_type() == at::kLong, "licht_scatter: idx must be int64");
    TORCH_CHECK(layer_ptrs.scalar_type() == at::kLong,
                "licht_scatter: layer_ptrs must be int64");
    TORCH_CHECK((P & 7) == 0, "licht_scatter: P must be a multiple of 8");
    TORCH_CHECK(idx.numel() >= nb, "licht_scatter: idx too small");
    TORCH_CHECK(layer_ptrs.numel() >= nL, "licht_scatter: layer_ptrs too small");
    long total_runs = nb * nL * 2;
    if (total_runs < 1) return;
    int  threads = 256;
    long blocks  = total_runs < 65535 ? total_runs : 65535;
    // Launch on the tensors' own device/stream regardless of the caller's
    // current device (avoids cross-device illegal access).
    const at::cuda::CUDAGuard device_guard(staging.device());
    auto stream  = at::cuda::getCurrentCUDAStream();
    licht_scatter_kernel<<<blocks, threads, 0, stream>>>(
        (const uint16_t*)staging.data_ptr(), idx.data_ptr<int64_t>(),
        layer_ptrs.data_ptr<int64_t>(), (int)nb, (int)nL, (int)dim, NBLK, P);
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "licht_scatter launch failed: ",
                cudaGetErrorString(err));
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("licht_scatter", &licht_scatter,
          "Fused multi-layer block scatter (round-kv arena load)",
          py::arg("staging"), py::arg("idx"), py::arg("layer_ptrs"),
          py::arg("nb"), py::arg("nL"), py::arg("dim"),
          py::arg("NBLK"), py::arg("P"));
    m.def("licht_scatter_from_arena", &licht_scatter_from_arena,
          "Direct host-pinned-arena -> paged scatter (no GPU staging)",
          py::arg("arena_host_ptr"), py::arg("src_slots"), py::arg("dst_idx"),
          py::arg("layer_ptrs"), py::arg("nb"), py::arg("nL"), py::arg("dim"),
          py::arg("NBLK"), py::arg("P"));
    m.def("licht_scatter_from_arena_layer", &licht_scatter_from_arena_layer,
          "Per-layer host-pinned-arena -> single paged layer scatter (pipeline)",
          py::arg("arena_host_ptr"), py::arg("src_slots"), py::arg("dst_idx"),
          py::arg("layer_ptr"), py::arg("nb"), py::arg("nL"), py::arg("layer_idx"),
          py::arg("dim"), py::arg("NBLK"), py::arg("P"));
    m.def("licht_gather_to_arena_layer", &licht_gather_to_arena_layer,
          "Per-layer single paged layer -> host-pinned-arena direct write (store)",
          py::arg("arena_host_ptr"), py::arg("dst_slots"), py::arg("src_idx"),
          py::arg("layer_ptr"), py::arg("nb"), py::arg("nL"), py::arg("layer_idx"),
          py::arg("dim"), py::arg("NBLK"), py::arg("P"));
}
