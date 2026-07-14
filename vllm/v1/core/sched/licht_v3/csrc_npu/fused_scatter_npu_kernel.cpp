// SPDX-License-Identifier: Apache-2.0
// Ascend C kernel: scatter block-major staging into all paged KV layers.
#include <cstdint>

#include "kernel_operator.h"

extern "C" __global__ __aicore__ void licht_scatter_npu_kernel(
    GM_ADDR staging_gm, GM_ADDR idx_gm, GM_ADDR layer_ptrs_gm, int64_t nb,
    int64_t nL, int64_t dim, int64_t NBLK, int64_t P) {
    auto staging = reinterpret_cast<__gm__ uint16_t*>(staging_gm);
    auto idx = reinterpret_cast<__gm__ int64_t*>(idx_gm);
    auto layer_ptrs = reinterpret_cast<__gm__ uint64_t*>(layer_ptrs_gm);

    const int64_t total_runs = nb * nL * 2;
    const int64_t block_num = AscendC::GetBlockNum();
    for (int64_t run = AscendC::GetBlockIdx(); run < total_runs;
         run += block_num) {
        const int64_t kv = run & 1;
        const int64_t m = run >> 1;
        const int64_t layer_idx = m % nL;
        const int64_t j = m / nL;
        const int64_t dst_blk = idx[j];
        auto layer = reinterpret_cast<__gm__ uint16_t*>(
            layer_ptrs[layer_idx]);
        const int64_t src_off =
            (((j * nL + layer_idx) * 2 + kv) * P);
        const int64_t dst_off = (dim == 1)
            ? ((kv * NBLK + dst_blk) * P)
            : ((dst_blk * 2 + kv) * P);

        for (int64_t r = 0; r < P; ++r) {
            layer[dst_off + r] = staging[src_off + r];
        }
    }
}
