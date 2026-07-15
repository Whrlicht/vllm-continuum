// SPDX-License-Identifier: Apache-2.0
// Host launcher for NPU staged scatter:
//   staging [nb, nL, 2, *rest] -> layer[dst_block_ids]
//
// The Python caller passes the real paged-KV layer tensors.  This extension
// moves the nested Python loop into C++ and issues ACL device-to-device async
// copies on the current NPU stream.  For this KV path the unit of work is a
// whole physical block, so runtime DMA copies are the NPU fast path; a custom
// Ascend C scalar-copy kernel would add compiler/toolchain fragility without
// improving the data movement pattern.
#include <torch/extension.h>

#include <cstdint>
#include <vector>

#include <c10/core/DeviceType.h>

#include "acl/acl.h"

#if __has_include(<torch_npu/csrc/core/npu/NPUStream.h>)
#include <torch_npu/csrc/core/npu/NPUStream.h>
#define LICHT_HAS_TORCH_NPU_STREAM 1
#else
#define LICHT_HAS_TORCH_NPU_STREAM 0
#endif

namespace {

bool is_npu_tensor(const torch::Tensor& t) {
    return t.device().type() == c10::DeviceType::PrivateUse1;
}

void check_npu_tensor(const torch::Tensor& t, const char* name) {
    TORCH_CHECK(is_npu_tensor(t), name, " must be an NPU tensor");
}

aclrtStream current_npu_stream() {
#if LICHT_HAS_TORCH_NPU_STREAM
    return c10_npu::getCurrentNPUStream().stream();
#else
    // ACL APIs accept the default stream as nullptr on supported runtimes.
    // This keeps the extension buildable across torch-npu header variants.
    return nullptr;
#endif
}

}  // namespace

void licht_scatter(torch::Tensor staging, torch::Tensor idx,
                   std::vector<torch::Tensor> layers, int64_t nb, int64_t nL,
                   int64_t dim, int64_t NBLK, int64_t P) {
    check_npu_tensor(staging, "staging");
    check_npu_tensor(idx, "idx");
    TORCH_CHECK(staging.is_contiguous(),
                "licht_scatter_npu: staging must be contiguous");
    TORCH_CHECK(idx.scalar_type() == at::kLong,
                "licht_scatter_npu: idx must be int64");
    TORCH_CHECK(idx.numel() >= nb,
                "licht_scatter_npu: idx length is smaller than nb");
    TORCH_CHECK(static_cast<int64_t>(layers.size()) >= nL,
                "licht_scatter_npu: layers length is smaller than nL");
    TORCH_CHECK(staging.element_size() == 2,
                "licht_scatter_npu: staging must be fp16/bf16 sized");
    TORCH_CHECK(dim == 0 || dim == 1,
                "licht_scatter_npu: dim must be 0 or 1");
    TORCH_CHECK(nb >= 0 && nL > 0 && NBLK > 0 && P > 0,
                "licht_scatter_npu: invalid nb/nL/NBLK/P");
    if (nb == 0) {
        return;
    }

    aclrtStream stream = current_npu_stream();
    aclError ret = ACL_SUCCESS;

    auto idx_cpu = idx.to(torch::kCPU, /*non_blocking=*/false).contiguous();
    const auto* idx_ptr = idx_cpu.data_ptr<int64_t>();
    const auto* staging_base = static_cast<const char*>(staging.data_ptr());
    const size_t bytes = static_cast<size_t>(P) * staging.element_size();

    for (int64_t li = 0; li < nL; ++li) {
        torch::Tensor layer = layers[li];
        check_npu_tensor(layer, "layer");
        TORCH_CHECK(layer.element_size() == 2,
                    "licht_scatter_npu: layer must be fp16/bf16 sized");
        auto* layer_base = static_cast<char*>(layer.data_ptr());
        for (int64_t j = 0; j < nb; ++j) {
            const int64_t dst_blk = idx_ptr[j];
            TORCH_CHECK(dst_blk >= 0 && dst_blk < NBLK,
                        "licht_scatter_npu: destination block out of range");
            for (int64_t kv = 0; kv < 2; ++kv) {
                const int64_t src_off =
                    (((j * nL + li) * 2 + kv) * P) * staging.element_size();
                const int64_t dst_elem_off = (dim == 1)
                    ? ((kv * NBLK + dst_blk) * P)
                    : ((dst_blk * 2 + kv) * P);
                const int64_t dst_off = dst_elem_off * layer.element_size();
                ret = aclrtMemcpyAsync(layer_base + dst_off, bytes,
                                       staging_base + src_off, bytes,
                                       ACL_MEMCPY_DEVICE_TO_DEVICE, stream);
                TORCH_CHECK(ret == ACL_SUCCESS,
                            "licht_scatter_npu: aclrtMemcpyAsync failed, ret=",
                            ret, ", layer=", li, ", block=", j, ", kv=", kv);
            }
        }
    }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("licht_scatter", &licht_scatter,
          "Ascend C fused staged scatter for round-kv NPU load",
          py::arg("staging"), py::arg("idx"), py::arg("layers"),
          py::arg("nb"), py::arg("nL"), py::arg("dim"),
          py::arg("NBLK"), py::arg("P"));
}
