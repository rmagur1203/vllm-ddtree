// DDTree — GDN 트리 커널의 빌드 편입 래퍼 (libtorch stable ABI).
//
// JIT 경로는 csrc/ddtree/ddtree_gdn_tree.cu 를 쓴다. 커널 본체는 두 경로가
// csrc/ddtree/ddtree_gdn_tree.cuh 를 함께 포함해 공유한다.
#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/csrc/stable/tensor.h>
#include <torch/csrc/stable/library.h>
#include <torch/csrc/stable/c/shim.h>

#include "torch_utils.h"
#define DDT_FAIL_VHPK(v) STD_TORCH_CHECK(false, "DDTree: HV/H must be in {1,2,3,4,8}")
#include "../ddtree/ddtree_gdn_tree.cuh"

void ddtree_gdn_decode_tree_mtp(
    torch::stable::Tensor& mixed_qkv, torch::stable::Tensor& a, torch::stable::Tensor& b, torch::stable::Tensor& A_log,
    torch::stable::Tensor& dt_bias, torch::stable::Tensor& state_indices, torch::stable::Tensor& cu_seqlens,
    torch::stable::Tensor& num_accepted_tokens, torch::stable::Tensor& tree_parents, torch::stable::Tensor& state,
    torch::stable::Tensor& output_gate, torch::stable::Tensor& norm_weight, torch::stable::Tensor& out,
    double scale, double norm_eps) {
  STD_TORCH_CHECK(static_cast<int64_t>(mixed_qkv.sizes().size()) == 2, "mixed_qkv must be [L, 2*H*128 + HV*128]");
  STD_TORCH_CHECK(static_cast<int64_t>(state.sizes().size()) == 4 && state.sizes()[2] == kDimV && state.sizes()[3] == kDimK,
              "state must be [slots, HV, 128, 128]");
  STD_TORCH_CHECK(tree_parents.scalar_type() == torch::headeronly::ScalarType::Int && tree_parents.is_cuda(),
              "tree_parents must be a CUDA int32 tensor");
  STD_TORCH_CHECK(tree_parents.sizes().size() == state_indices.sizes().size() &&
                  tree_parents.sizes()[0] == state_indices.sizes()[0] &&
                  tree_parents.sizes()[1] == state_indices.sizes()[1],
              "tree_parents must have the same shape as state_indices");

  const int HV = static_cast<int>(state.sizes()[1]);
  const int64_t key_width = mixed_qkv.sizes()[1] - static_cast<int64_t>(HV) * kDimV;
  STD_TORCH_CHECK(key_width > 0 && key_width % (2 * kDimK) == 0,
              "mixed_qkv width inconsistent with state");
  const int H = static_cast<int>(key_width / (2 * kDimK));
  const int vhpk = HV / H;
  const int N = static_cast<int>(state_indices.sizes()[0]);
  const int siw = static_cast<int>(state_indices.sizes()[1]);
  // 🔴 원본은 siw <= 8 로 막혀 있다 (공유메모리 + 워프당 1토큰).
  //    여기서는 토큰을 타일링하고 상태를 공유메모리에 두지 않으므로 상한이 없다.

  const int dt_type = dt_bias.scalar_type() == torch::headeronly::ScalarType::Float  ? kDtBiasFloat32
                      : dt_bias.scalar_type() == torch::headeronly::ScalarType::BFloat16 ? kDtBiasBFloat16
                                                               : kDtBiasFloat16;
  const Strides strides{mixed_qkv.strides()[0], a.strides()[0], b.strides()[0],
                        output_gate.strides()[0], state.strides()[0]};
  const dim3 grid(N, HV);
  const cudaStream_t stream = get_current_cuda_stream();

  DDT_DISPATCH_VHPK(vhpk, {
    if (state.scalar_type() == torch::headeronly::ScalarType::Float) {
      gdn_tree_kernel<float, V><<<grid, kThreads, 0, stream>>>(
          static_cast<const __nv_bfloat16*>(mixed_qkv.mutable_data_ptr()),
          static_cast<const __nv_bfloat16*>(a.mutable_data_ptr()),
          static_cast<const __nv_bfloat16*>(b.mutable_data_ptr()),
          A_log.mutable_data_ptr<float>(), dt_bias.mutable_data_ptr(),
          state_indices.mutable_data_ptr<int32_t>(), cu_seqlens.mutable_data_ptr<int32_t>(),
          num_accepted_tokens.mutable_data_ptr<int32_t>(), tree_parents.mutable_data_ptr<int32_t>(),
          static_cast<float*>(state.mutable_data_ptr()),
          static_cast<const __nv_bfloat16*>(output_gate.mutable_data_ptr()),
          norm_weight.mutable_data_ptr(),
          static_cast<__nv_bfloat16*>(out.mutable_data_ptr()), H, HV, siw, dt_type,
          norm_weight.scalar_type() == torch::headeronly::ScalarType::BFloat16,
          static_cast<float>(scale), static_cast<float>(norm_eps), strides);
    } else {
      gdn_tree_kernel<__nv_bfloat16, V><<<grid, kThreads, 0, stream>>>(
          static_cast<const __nv_bfloat16*>(mixed_qkv.mutable_data_ptr()),
          static_cast<const __nv_bfloat16*>(a.mutable_data_ptr()),
          static_cast<const __nv_bfloat16*>(b.mutable_data_ptr()),
          A_log.mutable_data_ptr<float>(), dt_bias.mutable_data_ptr(),
          state_indices.mutable_data_ptr<int32_t>(), cu_seqlens.mutable_data_ptr<int32_t>(),
          num_accepted_tokens.mutable_data_ptr<int32_t>(), tree_parents.mutable_data_ptr<int32_t>(),
          static_cast<__nv_bfloat16*>(state.mutable_data_ptr()),
          static_cast<const __nv_bfloat16*>(output_gate.mutable_data_ptr()),
          norm_weight.mutable_data_ptr(),
          static_cast<__nv_bfloat16*>(out.mutable_data_ptr()), H, HV, siw, dt_type,
          norm_weight.scalar_type() == torch::headeronly::ScalarType::BFloat16,
          static_cast<float>(scale), static_cast<float>(norm_eps), strides);
    }
  });
  STD_TORCH_CHECK(cudaGetLastError() == cudaSuccess,
                  "ddtree gdn tree kernel launch failed");
}

