"""M2 / 단계 2 — KV 컴팩션 커널 검증.

세 가지를 본다:
  1. Triton 결과 == torch 레퍼런스 결과
  2. 둘 다 == 스냅샷으로 독립 계산한 기대값 (수용 행은 옮겨지고 나머지는 그대로)
  3. 경쟁 조건이 실제로 발생하는 배치(src 와 dst 가 겹치는 경우)를 포함
  4. 🔴 슬롯이 단조 증가가 아닌 배치 — 예전 커널은 행을 순차 처리하며
     dst<=src 를 전제해서 여기서 깨졌고, 호출부가 매 스텝 D2H 로 검사해
     피해 갔다. 지금 커널은 전부 읽고 나서 쓰므로 전제 자체가 없다.
"""
import sys, torch
from vllm.v1.spec_decode.ddtree.compact import compact_kv_triton, compact_kv_torch

torch.manual_seed(0)
DEV = "cuda"


def make_case(num_layers, num_blocks, bs, h, d, segments):
    """segments: [(base_slot, [수용된 트리 오프셋...]), ...]"""
    # 실제 vLLM/FlashInfer 배치를 흉내: (blocks, bs, h, 2d) 를 만들고
    # (blocks, h, bs, 2d) 로 permute — shape (849,8,16,256) stride (32768,256,2048,1) 과 동형
    caches = [torch.randn(num_blocks, bs, h, 2 * d, dtype=torch.bfloat16, device=DEV)
              .permute(0, 2, 1, 3)
              for _ in range(num_layers)]
    src, dst, seg = [], [], [0]
    for entry in segments:
        if entry[0] == "raw":       # 슬롯을 직접 지정 (단조성 반례용)
            src.extend(entry[1])
            dst.extend(entry[2])
        else:
            base, accepted = entry
            for i, a in enumerate(accepted):
                src.append(base + a)
                dst.append(base + i)
        seg.append(len(src))
    return (caches,
            torch.tensor(src, dtype=torch.int32, device=DEV),
            torch.tensor(dst, dtype=torch.int32, device=DEV),
            torch.tensor(seg, dtype=torch.int32, device=DEV))


def expected(caches0, src, dst, bs):
    """스냅샷에서 직접 계산한 기대 결과 — 슬롯 = 연속 행."""
    from vllm.v1.spec_decode.ddtree.compact import row_bytes
    out = []
    s = src.tolist(); t = dst.tolist()
    for c0 in caches0:
        row = row_bytes(c0, bs) // c0.element_size()
        dims = sorted(range(c0.dim()), key=lambda i: -c0.stride(i))
        f_in = c0.permute(*dims).reshape(-1, row)
        f_out = f_in.clone()
        for a, b in zip(s, t):
            f_out[b] = f_in[a]
        out.append(f_out)
    return out


CASES = [
    ("단순 (겹침 없음)",      1, 8, 16, 2, 8, [(0, [0, 3, 5])]),
    ("🔴 겹침 발생",          1, 8, 16, 2, 8, [(0, [0, 2, 3])]),      # dst[2]=2 == src[1]=2
    ("전부 수용 (no-op)",     2, 8, 16, 2, 8, [(16, [0, 1, 2, 3])]),
    ("루트만 수용",           2, 8, 16, 2, 8, [(32, [0])]),
    ("페이지 경계 걸침",       2, 8, 16, 2, 8, [(12, [0, 4, 7, 9])]),  # 12..21 → 페이지 2개
    ("배치 3요청",            4, 16, 16, 4, 16,
        [(0, [0, 2, 5]), (64, [0, 1, 3, 7]), (160, [0, 6])]),
    ("64레이어 큰 트리",       64, 32, 16, 8, 128,
        [(0, [0, 1, 4, 9, 15, 22, 31])]),
    # 🔴 블록 경계에서 다음 블록 id 가 더 작은 경우 (블록은 free pool 에서 나온다).
    #    트리 위치는 증가하는데 슬롯은 5,6,7 → 80,81 이 아니라 80,81 → 5,6 처럼
    #    거꾸로 간다. dst>src 인 행이 생겨 예전 전제가 깨진다.
    #    dst[0]=82 > src[0]=81 이라 예전 검사에 걸리고, 그 쓰기가 src[1]=82 를
    #    덮으므로 순차 커널은 **실제로 틀린 값을 쓴다**.
    ("🔴 슬롯 역전 (블록 경계)", 2, 8, 16, 2, 8,
        [("raw", [81, 82, 6], [82, 79, 5])]),
    ("🔴 역전 세그먼트 + 정상 세그먼트", 2, 8, 16, 2, 8,
        [("raw", [20, 21, 22], [20, 21, 22]),
         ("raw", [65, 66, 60], [66, 64, 59])]),   # dst[0]=66 == src[1]=66
]


def main():
    bad = 0
    for name, nl, nb, bs, h, d, segs in CASES:
        caches, src, dst, seg = make_case(nl, nb, bs, h, d, segs)
        snap = [c.clone() for c in caches]
        exp = expected(snap, src, dst, bs)

        def flat(cs):
            from vllm.v1.spec_decode.ddtree.compact import row_bytes
            o = []
            for c in cs:
                row = row_bytes(c, bs) // c.element_size()
                dims = sorted(range(c.dim()), key=lambda i: -c.stride(i))
                o.append(c.permute(*dims).reshape(-1, row))
            return o

        tri = [c.clone() for c in caches]
        _seg = seg.tolist()
        _maxrow = max(_seg[i + 1] - _seg[i] for i in range(len(_seg) - 1))
        compact_kv_triton(tri, src, dst, seg, bs, _maxrow)

        ref = [c.clone() for c in caches]
        compact_kv_torch(ref, src, dst, seg, bs)

        e_tri = max((a.float() - b.float()).abs().max().item() for a, b in zip(flat(tri), exp))
        e_ref = max((a.float() - b.float()).abs().max().item() for a, b in zip(flat(ref), exp))
        ok = e_tri == 0.0 and e_ref == 0.0
        bad += not ok
        print(f"  {'🟢' if ok else '🔴'} {name:<22} 레이어={nl:<3} 행={src.numel():<3} "
              f"triton오차={e_tri}  torch오차={e_ref}")

    print(f"\n{len(CASES)}건 중 실패 {bad}건 —",
          "🟢 컴팩션 정확" if bad == 0 else "🔴 불일치")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
