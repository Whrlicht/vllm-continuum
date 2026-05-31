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
}
