"""
DDTree KV 컴팩션 — 수용된 트리 노드의 K/V 만 앞으로 당긴다.

체인 스펙 디코딩은 수용 토큰이 접두사라 num_computed_tokens 되감기로 끝나지만,
트리는 수용 경로가 트리 슬롯 범위에 흩어져 있어 gather 가 필요하다.

🔴 경쟁 조건 — 없다 (2026-09-02)
   예전에는 **행을 증가 순서로 순차 처리**해서 아직 안 읽은 소스를 안 덮게 했다.
   그 전제는 dst[i] <= src[i] 가 **슬롯 공간에서도** 성립해야 참인데, 트리가 KV
   블록 경계를 넘으면 다음 블록 id 가 더 작을 수 있어(블록은 free pool 에서 나온다)
   깨진다. 그래서 호출부가 매 스텝 `(dst > src).sum()` 을 **D2H 로 읽어** 안전한지
   확인했고, 그 읽기 하나가 파이프라인을 세웠다 (~0.95 ms).

   지금은 커널이 세그먼트의 행을 **전부 읽은 뒤에 전부 쓴다**. gather 의미가
   그대로라 순서에 아무 전제가 없고, 따라서 검사도 D2H 도 필요 없다.
   프로그램끼리는 원래 안 겹친다 — 세그먼트(요청)는 슬롯이 서로 다르고,
   fblk 는 행 안의 바이트 구간이 서로 다르고, layer 는 캐시가 다르다.
   대신 타일이 [MAXROWS, BLOCK] 이라 레지스터 압력이 곱으로 는다 (아래 BLOCK 조정).

레이아웃
   vLLM/FlashInfer 가 실제로 쓰는 배치에서 슬롯 하나의 K·V 는 메모리상 연속이다.
   예: shape (849, 8, 16, 256) stride (32768, 256, 2048, 1)
       → stride 내림차순 = dim0(블록) dim2(블록내오프셋) dim1 dim3
       → 슬롯당 연속 8*256 = 2048 원소, 행 시작 = slot * 2048
   이 성질을 strides 로 검사한다 (row_bytes 참고). 성립 안 하면 명시적으로 실패한다.
"""
from __future__ import annotations

import os

import torch
import triton
import triton.language as tl


def row_bytes(cache: torch.Tensor, block_size: int) -> int:
    """슬롯 하나가 차지하는 연속 바이트 수. 연속이 아니면 예외."""
    dims = sorted(range(cache.dim()), key=lambda i: -cache.stride(i))
    blk_dim = dims[0]
    off_dim = next((i for i in dims[1:] if cache.shape[i] == block_size), None)
    if off_dim is None:
        raise ValueError(f"block_size={block_size} 인 차원을 못 찾음: {tuple(cache.shape)}")

    row = 1
    for i in dims[dims.index(off_dim) + 1 :]:
        row *= cache.shape[i]
    if cache.stride(off_dim) != row or cache.stride(blk_dim) != block_size * row:
        raise ValueError(
            "슬롯이 메모리상 연속이 아닙니다 — 이 레이아웃은 아직 지원 안 함: "
            f"shape={tuple(cache.shape)} stride={cache.stride()} block_size={block_size}"
        )
    return row * cache.element_size()


@triton.jit
def _compact_kernel(
    cache_ptrs,          # int64[num_layers]
    src_ptr, dst_ptr,    # int32[total_rows] — 슬롯 id
    seg_start_ptr,       # int32[num_segments + 1]
    ROWB: tl.constexpr,     # 슬롯당 바이트
    BLOCK: tl.constexpr,
    MAXROWS: tl.constexpr,  # 세그먼트 최대 행 수 (2의 거듭제곱)
):
    layer = tl.program_id(0)
    seg = tl.program_id(1)
    fblk = tl.program_id(2)

    ptr = tl.load(cache_ptrs + layer).to(tl.pointer_type(tl.int8))
    lo = tl.load(seg_start_ptr + seg)
    hi = tl.load(seg_start_ptr + seg + 1)

    offs = fblk * BLOCK + tl.arange(0, BLOCK)
    cmask = offs < ROWB
    rows = lo + tl.arange(0, MAXROWS)
    rmask = rows < hi

    s = tl.load(src_ptr + rows, mask=rmask, other=0).to(tl.int64)
    d = tl.load(dst_ptr + rows, mask=rmask, other=0).to(tl.int64)

    # 🔴 읽기가 전부 끝난 뒤에 쓴다. v 에 대한 데이터 의존이라 컴파일러가 store 를
    #    load 앞으로 못 옮긴다 — 순서 전제 없이 gather 의미가 그대로 나온다.
    m = rmask[:, None] & cmask[None, :]
    v = tl.load(ptr + s[:, None] * ROWB + offs[None, :], mask=m)
    tl.store(ptr + d[:, None] * ROWB + offs[None, :], v, mask=m)


def attention_caches(caches, block_size: int):
    """슬롯 연속 레이아웃인 캐시만 고른다.

    🔴 하이브리드 모델(Qwen3.8-27B 등)의 kv_caches 에는 어텐션 KV 말고
       GDN/Mamba 재귀 상태 캐시도 섞여 있다. 후자는 토큰별 슬롯 구조가 아니라
       요청별 상태라 이 컴팩션 대상이 아니다 — 재귀 계층의 트리 처리는
       ddtree_gdn.py 와 패치된 fused_sigmoid_gating.py 가 담당한다.
       (실측: (921,1,1,3534848) block_size=1024 → row_bytes 가 거부)
    """
    out = []
    for c in caches:
        try:
            row_bytes(c, block_size)
        except ValueError:
            continue
        out.append(c)
    return out


# 한 프로그램이 레지스터에 드는 타일 크기(바이트). MAXROWS x BLOCK 이 여기에
# 맞도록 BLOCK 을 줄이고, 모자란 만큼은 fblk 축 병렬로 돌린다.
_TILE_BYTES = int(os.environ.get("VLLM_DDTREE_COMPACT_TILE", "4096"))

# 🔴 캐시 포인터는 KV 캐시를 할당한 뒤로 안 바뀌는데, 호출마다 파이썬 리스트를
#    만들어 H2D 로 올리고 있었다. 실측 (27B, 어텐션 16계층): 커널 호출 59.4us 중
#    31.7us — 절반이 넘는다. 스텝마다 캐시 그룹 수만큼 반복되므로 그대로 쌓인다.
#    키에 device 를 넣는 이유: 포인터 값은 디바이스마다 독립이라 겹칠 수 있다.
_PTR_CACHE: dict[tuple, torch.Tensor] = {}
_PTR_CACHE_MAX = 8      # 운영에서는 키가 (그룹당) 하나뿐이다 — 테스트용 상한
_NO_PTR_MEMO = os.environ.get("VLLM_DDTREE_NOPTRMEMO") == "1"


def _cache_ptrs(caches) -> torch.Tensor:
    """캐시 data_ptr 들을 담은 int64 텐서. 같은 캐시면 같은 텐서를 돌려준다."""
    dev = caches[0].device
    key = (dev.type, dev.index) + tuple(c.data_ptr() for c in caches)
    t = None if _NO_PTR_MEMO else _PTR_CACHE.get(key)
    if t is None:
        t = torch.tensor([c.data_ptr() for c in caches],
                         dtype=torch.int64, device=dev)
        if not _NO_PTR_MEMO:
            if len(_PTR_CACHE) >= _PTR_CACHE_MAX:
                _PTR_CACHE.clear()
            _PTR_CACHE[key] = t
    return t


def compact_kv_triton(caches, src_slots, dst_slots, seg_start, block_size: int,
                      max_seg_rows: int) -> None:
    """max_seg_rows: 세그먼트 하나의 최대 행 수. 호출부가 CPU 에 이미 갖고 있는
    값이라 공짜다 — 이걸 몰라서 GPU 에 물어보면 다시 동기화가 된다."""
    caches = attention_caches(caches, block_size)
    if not caches:
        return
    rowb = row_bytes(caches[0], block_size)
    maxrows = triton.next_power_of_2(max(1, int(max_seg_rows)))
    block = triton.next_power_of_2(min(rowb, max(64, _TILE_BYTES // maxrows)))
    grid = (len(caches), seg_start.numel() - 1, triton.cdiv(rowb, block))
    _compact_kernel[grid](
        _cache_ptrs(caches), src_slots.to(torch.int32), dst_slots.to(torch.int32),
        seg_start.to(torch.int32), ROWB=rowb, BLOCK=block, MAXROWS=maxrows,
    )


def compact_kv_torch(caches, src_slots, dst_slots, seg_start, block_size: int) -> None:
    """레퍼런스. gather 후 scatter 라 경쟁 조건이 없다."""
    src = src_slots.to(torch.long)
    dst = dst_slots.to(torch.long)
    for c in caches:
        rowb = row_bytes(c, block_size)
        row = rowb // c.element_size()
        dims = sorted(range(c.dim()), key=lambda i: -c.stride(i))
        p = c.permute(*dims)                        # stride 내림차순 → 연속
        assert p.is_contiguous()
        flat = p.reshape(-1, row)
        tmp = flat.index_select(0, src).clone()      # 먼저 전부 읽는다
        flat.index_copy_(0, dst, tmp)
