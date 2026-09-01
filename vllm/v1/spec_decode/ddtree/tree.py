"""
DDTree 트리 구성 — vLLM 통합용.

알고리즘은 liranringel/ddtree (MIT) 의 build_ddtree_tree 를 그대로 따릅니다.
달라진 점은 vLLM 이 쓸 수 있게 (a) 요청 배치를 다루고 (b) FlashInfer 가 받는
평탄화 bool 마스크를 바로 만들어 준다는 것뿐입니다.

원본: https://github.com/liranringel/ddtree  ddtree.py:84
"""
from __future__ import annotations

import heapq
import os
from dataclasses import dataclass, field

import numpy as np
import torch


@dataclass
class Tree:
    """요청 하나의 드래프트 트리. 인덱스 0 은 루트(= 직전 스텝의 확정 토큰)."""

    token_ids: np.ndarray          # [num_nodes-1] 루트 제외 노드의 토큰
    depths: np.ndarray             # [num_nodes-1] 루트 기준 깊이 (1부터)
    parents: list[int]             # [num_nodes] parents[0] = -1
    child_maps: list[dict[int, int]]   # child_maps[i][token] = 자식 인덱스
    num_nodes: int                 # 루트 포함

    # 노드별 '부모 조건부' log-prob 과 그 깊이에서의 rank.
    # 🔴 누적이 아니라 조건부여야 보정 측정이 성립한다 — 누적은 조상까지 섞여
    #    (깊이, rank) 칸마다 표본이 갈라진다.
    node_lp: np.ndarray | None = field(default=None, repr=False)
    node_rank: np.ndarray | None = field(default=None, repr=False)

    _vis: np.ndarray | None = field(default=None, repr=False)
    # 깊이별 상위 2개 log-prob — 드래프터 확률의 보정 상태를 사후 분석하려고 남긴다
    lp_top: list | None = field(default=None, repr=False)
    dyn_mode: str | None = field(default=None, repr=False)   # 동적 모양 선택 결과
    e_chain: float | None = field(default=None, repr=False)  # 기대 사슬 길이

    @property
    def is_chain(self) -> bool:
        """가시성이 causal 과 같은가 — 즉 노드가 0,1,2,... 로 한 줄인가.

        사슬 트리는 마스크를 줄 필요가 없다. causal 경로가 그대로 옳다.
        (마스크를 안 주면 accept 가 그 트리를 거부하므로, 런타임이 이런 트리를
         따로 표시해서 수용은 정상으로 돌게 해야 한다.)
        """
        return all(self.parents[i] == i - 1 for i in range(1, self.num_nodes))

    @property
    def visibility(self) -> np.ndarray:
        """[num_nodes, num_nodes] bool — vis[i][j] 는 노드 i 가 j 를 볼 수 있는가.
        조상과 자기 자신만 True (ancestor-only 마스크)."""
        if self._vis is None:
            n = self.num_nodes
            v = np.zeros((n, n), dtype=np.bool_)
            v[0, 0] = True
            for i in range(1, n):
                p = self.parents[i]
                v[i, :i] = v[p, :i]
                v[i, i] = True
            self._vis = v
        return self._vis


def build_tree(
    top_log_probs: np.ndarray,   # [depth_limit, topk] 내림차순
    top_token_ids: np.ndarray,   # [depth_limit, topk]
    budget: int,
    spine: bool = False,
    rank_bonus: float = 0.0,
) -> Tree:
    """best-first 힙으로 노드 예산만큼 트리를 넓힌다.

    rank_bonus(δ)는 '적응형 폭' 의 핵심이다. best-first 의 가중치는 노드가 수용
    경로에 놓일 확률의 **추정치**인데, 실측하면 그 추정이 깊이·rank 방향으로
    구조적으로 치우쳐 있다. 실제/추정 비를 C 라 하면 (8B+EAGLE3+예산5, 모양을
    강제해 선택 편향을 없앤 측정):

        깊이 1..5 의 rank0:  0.85  0.66  0.26  0.20  0.06
        깊이 1 의 rank0/1/2: 0.85  0.96  1.69

    즉 **깊은 척추는 크게 과대평가, 형제는 과소평가**된다. log C 가 깊이에 거의
    선형(기울기 -0.65)이고 rank 에도 선형(+0.3)이라, 노드 가중치에
    `depth_bonus*깊이 + rank_bonus*rank` 를 더하면 그대로 보정된다. depth_bonus 는
    lp 에 상수를 더해 이미 처리되므로(shape_and_build) 여기서는 rank 쪽만
    형제를 밀어 넣을 때 누적한다.

    폭이 고정 값으로 정해지지 않는 것이 핵심이다 — 보정된 가중치끼리 겨루므로
    드래프터가 확신하는 자리에서는 깊이가 이기고 헷갈리는 자리에서만 가지가
    열린다. 그게 '적응형' 의 의미다.

    가중치는 루트에서 그 노드까지의 누적 log-prob 이다. 매번 꺼낼 때마다
      - 형제(같은 깊이, rank+1)
      - 자식(깊이+1, rank 0)
    을 밀어 넣어, '타깃과 일치할 확률이 높은 순서'로 예산을 배분한다.

    spine=True 는 참조 구현에서 벗어난다. 누적 log-prob 은 깊이가 늘 때마다 단조
    감소하므로 best-first 는 **구조적으로 폭을 깊이보다 선호**한다. 그런데 실측에서
    수용 길이는 양극단이었다 — 즉시 실패하거나 지평 전체를 받아낸다 — 그리고 수용
    토큰의 80%가 길이 5 이상, 52%가 길이 8 이상에서 나왔다 (2026-08-28, 4B bf16).
    깊이를 못 내면 값어치의 대부분을 잘라먹는다. 그래서 지평까지 사슬(척추)을 먼저
    확보하고, 남는 예산만 best-first 로 분기에 쓴다.
    """
    depth_limit, topk = top_log_probs.shape
    if budget <= 0 or depth_limit == 0:
        return Tree(np.empty(0, np.int64), np.empty(0, np.int64), [-1], [dict()], 1,
                    node_lp=np.empty(0, np.float32), node_rank=np.empty(0, np.int32))

    # (정렬키, ranks, 부모인덱스, 깊이, rank, 누적logw) — ranks 는 동점 처리용
    heap: list[tuple[float, tuple[int, ...], int, int, int, float]] = []

    node_token_ids = np.empty(budget, dtype=np.int64)
    node_depths = np.empty(budget, dtype=np.int64)
    node_lp = np.empty(budget, dtype=np.float32)
    node_rank = np.empty(budget, dtype=np.int32)
    parents = np.empty(budget + 1, dtype=np.int32)
    parents[0] = -1
    child_maps: list[dict[int, int]] = [dict()]
    n = 0

    if spine:
        # 척추: 깊이마다 rank 0 을 골라 지평까지 사슬을 먼저 깐다.
        # 각 척추 노드의 형제(rank 1)를 힙에 심어, 남은 예산이 그 위에서 분기하게 한다.
        parent_index, logw = 0, 0.0
        for d in range(1, min(depth_limit, budget) + 1):
            lp = float(top_log_probs[d - 1, 0])
            logw += lp
            token_id = int(top_token_ids[d - 1, 0])
            cur = n + 1
            node_token_ids[n] = token_id
            node_depths[n] = d
            node_lp[n] = lp
            node_rank[n] = 0
            parents[cur] = parent_index
            child_maps.append(dict())
            child_maps[parent_index][token_id] = cur
            n += 1
            if topk > 1:
                w = logw - lp + float(top_log_probs[d - 1, 1]) + rank_bonus
                heapq.heappush(
                    heap, (-w, (0,) * (d - 1) + (1,), parent_index, d, 1, w))
            parent_index = cur
    else:
        first = float(top_log_probs[0, 0])
        heap.append((-first, (0,), 0, 1, 0, first))

    while heap and n < budget:
        _, ranks, parent_index, depth, rank, logw = heapq.heappop(heap)

        token_id = int(top_token_ids[depth - 1, rank])
        cur = n + 1
        node_token_ids[n] = token_id
        node_depths[n] = depth
        node_lp[n] = float(top_log_probs[depth - 1, rank])
        node_rank[n] = rank
        parents[cur] = parent_index
        child_maps.append(dict())
        child_maps[parent_index][token_id] = cur
        n += 1

        if rank + 1 < topk:                       # 형제
            w = logw - float(top_log_probs[depth - 1, rank]) + float(
                top_log_probs[depth - 1, rank + 1]
            ) + rank_bonus
            heapq.heappush(heap, (-w, ranks[:-1] + (rank + 1,), parent_index, depth, rank + 1, w))

        if depth < depth_limit:                   # 자식
            w = logw + float(top_log_probs[depth, 0])
            heapq.heappush(heap, (-w, ranks + (0,), cur, depth + 1, 0, w))

    return Tree(
        token_ids=node_token_ids[:n],
        depths=node_depths[:n],
        parents=parents[: n + 1].tolist(),
        child_maps=child_maps,
        num_nodes=n + 1,
        node_lp=node_lp[:n],
        node_rank=node_rank[:n],
    )


# float 임시본이 한 번에 잡을 수 있는 최대 바이트. 배치 전체를 한꺼번에
# float 로 올리면 [요청수 x 깊이 x vocab] 이라 배치가 크면 GiB 단위가 된다.
#
# 실측 (요청 64, vocab 151936, 깊이 15, A6000): 결과는 상한과 무관하게 동일하고
# 속도-메모리만 맞바꾼다.
#     16 MiB  15.67 ms  최대할당  310 MiB
#     64 MiB  12.34 ms  최대할당  406 MiB   ← 기본값
#    256 MiB  11.47 ms  최대할당  790 MiB
#   무제한     11.13 ms  최대할당 1391 MiB
# 배치가 큰 상황은 곧 메모리가 빠듯한 상황이라 안전 쪽을 기본으로 둔다.
_CAST_BUDGET = int(os.environ.get("VLLM_DDTREE_CAST_MIB", "64")) << 20


def _topk_rows(rows: torch.Tensor, k_all: int):
    """[행, vocab] → (정규화 log-prob 상위 k, 그 토큰 id). 둘 다 GPU 에 남는다.

    🔴 연산 **개수**가 값이다. 이 경로의 실제 비용은 GPU 커널(0.147 ms)도 D2H
       (0.056 ms)도 아니고 **CPU 디스패치 0.43 ms** 였다 — eager PyTorch 에서
       디스패치 하나가 30~40 us 인데 float() -> topk -> logsumexp(내부에서
       max/sub/exp/sum/log 5개) -> 뺄셈 으로 여덟 개 넘게 던지고 있었다.

       log_softmax 는 정의가 `logits - logsumexp(logits)` 그 자체라 그 전부와
       수학적으로 동일하다. dtype 인자로 캐스트까지 흡수해 **디스패치 2개**로
       줄인다. 출력은 바이트 동일이다.
    """
    lsm = torch.log_softmax(rows, dim=-1, dtype=torch.float32)
    return torch.topk(lsm, k=k_all, dim=-1)


def topk_from_logits(draft_logits: torch.Tensor, budget: int,
                     topk_cap: int = 1 << 30):
    """[..., depth, vocab] logits → (lp, ids, topk) numpy.

    선행 차원은 그대로 유지하므로 [depth, vocab] 도 [요청, depth, vocab] 도 받는다.

    🔴 **D2H 는 호출당 정확히 2회다.** 예전에는 요청마다 build_tree_from_logits
       를 불러 2n회 났는데, 각 복사가 드래프터 GPU 작업이 끝날 때까지 파이프라인을
       세우므로 배치 크기에 비례해 비용이 붙었다.
    """
    vocab = draft_logits.shape[-1]
    topk = max(1, min(budget, vocab, topk_cap))
    # 패딩용 여분 후보까지 한 번에 뽑는다 (topk_cap=1 이면 topk 가 1 이라 모자람)
    k_all = min(vocab, max(topk, budget + 1))

    flat = draft_logits.reshape(-1, vocab)
    step = max(1, _CAST_BUDGET // (vocab * 4))
    if step >= flat.shape[0]:
        lp_g, ids_g = _topk_rows(flat, k_all)
    else:
        parts = [_topk_rows(flat[i:i + step], k_all)
                 for i in range(0, flat.shape[0], step)]
        lp_g = torch.cat([a for a, _ in parts])
        ids_g = torch.cat([b for _, b in parts])

    shape = tuple(draft_logits.shape[:-1]) + (k_all,)
    # 🔴 D2H 를 한 번으로 합친다. .to("cpu") 는 매번 동기화라 두 번이면 두 번 선다.
    #    vocab < 2^24 면 토큰 id 를 float32 로 **정확히** 실을 수 있다 (float32 는
    #    2^24 까지 정수를 손실 없이 표현한다). 넘으면 예전처럼 따로 보낸다.
    if vocab < (1 << 24):
        both = torch.cat([lp_g, ids_g.to(torch.float32)], dim=-1).cpu().numpy()
        lp = both[:, :k_all].reshape(shape)
        ids = both[:, k_all:].astype(np.int64).reshape(shape)
    else:
        lp = lp_g.to(device="cpu", dtype=torch.float32).numpy().reshape(shape)
        ids = ids_g.to(device="cpu", dtype=torch.long).numpy().reshape(shape)
    return lp, ids, topk


def build_tree_from_logits(draft_logits: torch.Tensor, budget: int,
                          topk_cap: int = 1 << 30,
                          pad_to_budget: bool = True,
                          spine: bool = False,
                          depth_bonus: float = 0.0,
                          dynamic_tau: float | None = None,
                          allow_short: bool = False,
                          rank_bonus: float = 0.0) -> Tree:
    """드래프터 logits [depth_limit, vocab] → Tree. topk/정규화는 GPU 에서.

    topk_cap=1 이면 분기가 없는 순수 사슬이 된다 (이분법용).
    배치 경로는 topk_from_logits + shape_and_build 를 직접 쓴다 — 그래야 D2H 가
    요청마다가 아니라 스텝마다 한 번씩 난다."""
    lp, ids, topk = topk_from_logits(draft_logits, budget, topk_cap)
    return shape_and_build(lp, ids, budget, topk=topk, spine=spine,
                           depth_bonus=depth_bonus, dynamic_tau=dynamic_tau,
                           allow_short=allow_short, pad_to_budget=pad_to_budget,
                           rank_bonus=rank_bonus)


def shape_and_build(lp: np.ndarray, ids: np.ndarray, budget: int, *,
                    topk: int, spine: bool = False, depth_bonus: float = 0.0,
                    dynamic_tau: float | None = None, allow_short: bool = False,
                    pad_to_budget: bool = True, rank_bonus: float = 0.0) -> Tree:
    """모양 결정 + 트리 생성. DFlash 경로와 ngram 경로가 함께 쓴다.

    🔴 예전에는 이 로직이 build_tree_from_logits 안에만 있어서 ngram 경로
       (runtime.propose)가 build_tree 를 직접 불렀고, spine/depth_bonus/
       dynamic_tau 손잡이가 **전부 무시됐다**. 순수 어텐션 실험에서 동적 선택과
       진짜 사슬이 아무 효과가 없던 원인이다 (2026-08-28).
    """
    topk = max(1, min(topk, lp.shape[1]))
    _mode = None
    e_chain = None
    if dynamic_tau is not None:
        # 동적 모양 선택. 예산(=검증 forward 토큰 수)은 엔진 초기화 때 고정이라
        # 스텝마다 못 바꾼다. 그러나 같은 예산 안에서 모양은 바꿀 수 있다.
        # 판단 근거는 드래프터가 말하는 기대 사슬 길이 E = Σ_d ∏_{i<=d} p_i 다.
        # 실측(t21_calib)에서 E 는 실제 수용을 ~2배 과소평가하지만 스텝 간
        # **상대 비교**에는 쓸 수 있다 (피어슨 상관 0.748).
        e_chain = float(np.cumprod(np.exp(lp[:, 0])).sum())
        spine = e_chain >= dynamic_tau
        if spine and allow_short:
            # 진짜 사슬: 예산 전부를 깊이에 쏟고, 후보가 모자라면 짧은 드래프트를
            # 낸다. vLLM 은 스텝마다 폭을 다시 읽는다(prev_num_spec_tokens).
            topk = 1
            pad_to_budget = False
        _mode = "chain" if spine else "tree"

    if depth_bonus:
        # 🔴 log-prob 에 상수를 '곱하면' 무연산이다 — 단조 변환이라 heap 순서가
        #    그대로다. 깊이 d 노드가 얕은 형제 대비 depth_bonus*(d-1) 이득을
        #    보려면 '더해야' 한다. 같은 깊이 형제끼리는 상수가 상쇄된다.
        lp = lp + depth_bonus
    tree = build_tree(lp[:, :topk], ids[:, :topk], budget, spine=spine,
                      rank_bonus=rank_bonus)
    tree.lp_top = (lp[:, : min(2, lp.shape[1])] - depth_bonus).tolist()
    if depth_bonus and tree.node_lp is not None:
        # 보정 측정은 '드래프터가 말한 값' 을 봐야 하므로 보정분을 되돌린다.
        tree.node_lp = tree.node_lp - depth_bonus
    tree.dyn_mode = _mode
    tree.e_chain = e_chain
    if pad_to_budget:
        pad_tree_to_budget(tree, ids[0], budget)
    return tree


def pad_tree_to_budget(tree: Tree, depth0_ids: np.ndarray, budget: int) -> Tree:
    """루트 자식으로 무해한 노드를 붙여 노드 수를 예산에 맞춘다 (제자리 수정).

    드래프터의 horizon 이 예산보다 짧으면 (예: topk_cap=1 인 순수 사슬은
    깊이당 한 노드뿐이라 최대 horizon 개) 트리가 예산을 못 채운다. 그러면
    vLLM 이 잡아둔 드래프트 슬롯 수와 어긋나고, GDN 계층이 트리 정보를 못 받아
    8토큰 상한이 있는 원본 커널로 떨어져 죽는다 (state_indices [N,S] S<=8).

    붙이는 노드는 깊이 0 분포의 아직 안 쓴 상위 토큰이다. 타깃이 그 토큰을
    뽑으면 그대로 수용되고 — 그게 곧 greedy 정답이므로 무손실이다 — 아니면
    버려진다. child_maps 가 토큰을 키로 쓰므로 중복 토큰은 절대 넣지 않는다.
    """
    need = budget - (tree.num_nodes - 1)
    if need <= 0:
        return tree
    root_children = tree.child_maps[0]
    extra_tokens, extra_depths = [], []
    for tok in depth0_ids.tolist():
        if len(extra_tokens) >= need:
            break
        if tok in root_children:
            continue
        root_children[tok] = tree.num_nodes + len(extra_tokens)  # 루트 포함 인덱스
        extra_tokens.append(tok)
        extra_depths.append(1)
    if not extra_tokens:
        return tree
    tree.token_ids = np.concatenate(
        [tree.token_ids, np.asarray(extra_tokens, dtype=tree.token_ids.dtype)])
    tree.depths = np.concatenate(
        [tree.depths, np.asarray(extra_depths, dtype=tree.depths.dtype)])
    tree.parents.extend([0] * len(extra_tokens))
    tree.child_maps.extend({} for _ in extra_tokens)
    if tree.node_lp is not None:
        # 패딩 노드는 깊이 0 분포의 '안 쓴' 상위 토큰이라 rank 를 모른다.
        # 보정 측정에서 빼려고 rank=-1 로 표시한다.
        tree.node_lp = np.concatenate(
            [tree.node_lp, np.full(len(extra_tokens), np.nan, np.float32)])
        tree.node_rank = np.concatenate(
            [tree.node_rank, np.full(len(extra_tokens), -1, np.int32)])
    tree.num_nodes += len(extra_tokens)
    tree._vis = None                                   # 캐시 무효화
    return tree


def flat_tree_mask(
    trees: list[Tree],
    past_lens: list[int],
    device: torch.device | str = "cuda",
) -> torch.Tensor:
    """M1 패치의 마스크 공급자가 반환할 1D bool 텐서.

    요청 r 에 대해 (num_nodes x (past + num_nodes)) 를 만들고,
    과거 구간은 전부 True, 트리 구간은 조상만 True. 그걸 평탄화해 이어붙인다.
    """
    parts = []
    for tree, past in zip(trees, past_lens):
        n = tree.num_nodes
        m = np.ones((n, past + n), dtype=np.bool_)
        m[:, past:] = tree.visibility
        parts.append(torch.from_numpy(m.reshape(-1)))
    return torch.cat(parts).to(device)


def follow_tree(tree: Tree, sampled_token_ids: list[int]) -> tuple[list[int], int]:
    """타깃이 각 노드 위치에서 뽑은 토큰으로 트리를 내려간다.

    sampled_token_ids[i] 는 노드 i 의 로짓에서 나온 토큰.
    반환: (수용된 노드 인덱스 경로, 마지막 보너스 토큰)
    루트(인덱스 0)는 항상 수용된다 — 이미 확정된 토큰이므로.
    """
    accepted = [0]
    cur = 0
    nxt = int(sampled_token_ids[cur])
    while nxt in tree.child_maps[cur]:
        cur = tree.child_maps[cur][nxt]
        accepted.append(cur)
        nxt = int(sampled_token_ids[cur])
    return accepted, nxt
