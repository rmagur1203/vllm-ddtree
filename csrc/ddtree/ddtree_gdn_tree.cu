// DDTree — GDN 융합 디코드 커널의 트리판.
//
// 원본: vllm/csrc/libtorch_stable/gdn/fused_gdn_decode_kernel.cu (a9a17e709)
//
// 원본의 두 가지 8토큰 제약을 없앤다:
//   1) 공유메모리가 kMaxMtpTokens 에 비례 (토큰당 1,544B, 정적 48KB 한도)
//   2) 프리로드가 '워프 하나당 토큰 하나' (8워프)
// → 토큰을 kTile(=8) 단위로 타일링하고, 상태는 공유메모리에 두지 않는다.
//
// 트리 지원: 각 토큰이 '직전 토큰' 이 아니라 '부모 노드' 의 상태에서 출발한다.
//   부모 슬롯 = state_indices[parent[t]] (parent < 0 이면 source_slot)
//   힙 순서라 parent(t) < t 가 보장되고, 워프 w 는 자기 행만 읽고 쓰므로
//   워프 간 동기화가 필요 없다 (같은 스레드가 쓴 주소를 그대로 읽는다).
//
// 곱셈-누적 순서는 원본과 동일 → 사슬 부모를 주면 원본과 같은 결과여야 한다.

// JIT 경로 (VLLM_DDTREE_JIT=1). 빌드 편입 경로는
// csrc/libtorch_stable/ddtree_gdn_tree.cu 를 쓴다.
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAException.h>
#define DDT_FAIL_VHPK(v) TORCH_CHECK(false, "DDTree: HV/H must be in {1,2,3,4,8}, got ", v)
#include "ddtree_gdn_tree.cuh"

void gdn_decode_tree_mtp(
    at::Tensor mixed_qkv, at::Tensor a, at::Tensor b, at::Tensor A_log,
    at::Tensor dt_bias, at::Tensor state_indices, at::Tensor cu_seqlens,
    at::Tensor num_accepted_tokens, at::Tensor tree_parents, at::Tensor state,
    at::Tensor output_gate, at::Tensor norm_weight, at::Tensor out,
    double scale, double norm_eps) {
  TORCH_CHECK(mixed_qkv.dim() == 2, "mixed_qkv must be [L, 2*H*128 + HV*128]");
  TORCH_CHECK(state.dim() == 4 && state.size(2) == kDimV && state.size(3) == kDimK,
              "state must be [slots, HV, 128, 128]");
  TORCH_CHECK(tree_parents.dtype() == at::kInt && tree_parents.is_cuda(),
              "tree_parents must be a CUDA int32 tensor");
  TORCH_CHECK(tree_parents.sizes() == state_indices.sizes(),
              "tree_parents must have the same shape as state_indices");

  const int HV = static_cast<int>(state.size(1));
  const int64_t key_width = mixed_qkv.size(1) - static_cast<int64_t>(HV) * kDimV;
  TORCH_CHECK(key_width > 0 && key_width % (2 * kDimK) == 0,
              "mixed_qkv width inconsistent with state");
  const int H = static_cast<int>(key_width / (2 * kDimK));
  const int vhpk = HV / H;
  const int N = static_cast<int>(state_indices.size(0));
  const int siw = static_cast<int>(state_indices.size(1));
  // 🔴 원본은 siw <= 8 로 막혀 있다 (공유메모리 + 워프당 1토큰).
  //    여기서는 토큰을 타일링하고 상태를 공유메모리에 두지 않으므로 상한이 없다.

  const int dt_type = dt_bias.scalar_type() == at::kFloat  ? kDtBiasFloat32
                      : dt_bias.scalar_type() == at::kBFloat16 ? kDtBiasBFloat16
                                                               : kDtBiasFloat16;
  const Strides strides{mixed_qkv.stride(0), a.stride(0), b.stride(0),
                        output_gate.stride(0), state.stride(0)};
  const dim3 grid(N, HV);
  auto stream = c10::cuda::getCurrentCUDAStream();

  DDT_DISPATCH_VHPK(vhpk, {
    if (state.scalar_type() == at::kFloat) {
      gdn_tree_kernel<float, V><<<grid, kThreads, 0, stream>>>(
          static_cast<const __nv_bfloat16*>(mixed_qkv.data_ptr()),
          static_cast<const __nv_bfloat16*>(a.data_ptr()),
          static_cast<const __nv_bfloat16*>(b.data_ptr()),
          A_log.data_ptr<float>(), dt_bias.data_ptr(),
          state_indices.data_ptr<int>(), cu_seqlens.data_ptr<int>(),
          num_accepted_tokens.data_ptr<int>(), tree_parents.data_ptr<int>(),
          static_cast<float*>(state.data_ptr()),
          static_cast<const __nv_bfloat16*>(output_gate.data_ptr()),
          norm_weight.data_ptr(),
          static_cast<__nv_bfloat16*>(out.data_ptr()), H, HV, siw, dt_type,
          norm_weight.scalar_type() == at::kBFloat16,
          static_cast<float>(scale), static_cast<float>(norm_eps), strides);
    } else {
      gdn_tree_kernel<__nv_bfloat16, V><<<grid, kThreads, 0, stream>>>(
          static_cast<const __nv_bfloat16*>(mixed_qkv.data_ptr()),
          static_cast<const __nv_bfloat16*>(a.data_ptr()),
          static_cast<const __nv_bfloat16*>(b.data_ptr()),
          A_log.data_ptr<float>(), dt_bias.data_ptr(),
          state_indices.data_ptr<int>(), cu_seqlens.data_ptr<int>(),
          num_accepted_tokens.data_ptr<int>(), tree_parents.data_ptr<int>(),
          static_cast<__nv_bfloat16*>(state.data_ptr()),
          static_cast<const __nv_bfloat16*>(output_gate.data_ptr()),
          norm_weight.data_ptr(),
          static_cast<__nv_bfloat16*>(out.data_ptr()), H, HV, siw, dt_type,
          norm_weight.scalar_type() == at::kBFloat16,
          static_cast<float>(scale), static_cast<float>(norm_eps), strides);
    }
  });
  C10_CUDA_CHECK(cudaGetLastError());
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("gdn_decode_tree_mtp", &gdn_decode_tree_mtp,
        "GDN decode MTP post-conv with tree parents (no 8-token cap)");
}
