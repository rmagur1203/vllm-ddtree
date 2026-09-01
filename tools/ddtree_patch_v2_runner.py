"""V2 러너에 DDTree 훅을 얹는다 (1단계: 훅 1·2·3·5).

🔴 컨테이너의 vLLM 리비전과 저장소 리비전이 다르므로, 반드시 **컨테이너에서
   뽑은 원본**에 적용해야 한다 (저장소본을 얹으면 ImportError 가 난다).

  사용: python3 patch_v2_runner.py <원본> <출력>
"""
import io, sys

src, dst = sys.argv[1], sys.argv[2]
s = io.open(src, encoding="utf-8").read()


def sub(old, new, what):
    global s
    if old not in s:
        raise SystemExit(f"🔴 앵커 없음: {what}")
    s = s.replace(old, new, 1)
    print(f"  적용: {what}")


# ── 훅 5 헬퍼 + import ────────────────────────────────────────────
i = s.index("\nimport ")
s = s[:i] + "\nimport contextlib\nimport os as _os_ddt" + s[i:]
sub("logger = init_logger(__name__)", '''logger = init_logger(__name__)


@contextlib.contextmanager
def _ddt_drafter(runner):
    """드래프터 forward 동안 DDTree 마스크 공급자를 잠근다.

    🔴 트리 마스크는 **타깃** 검증용이다. 드래프터 forward 에 걸면 드래프터가
       트리 모양 위에서 돌아 드래프트 자체가 오염된다.
    """
    dd = getattr(runner, "ddtree", None)
    if dd is None:
        yield
        return
    dd.in_drafter = True
    try:
        yield
    finally:
        dd.in_drafter = False


class _DDTSpecMD:
    """DDTree.accept 이 기대하는 V1 SpecDecodeMetadata 모양의 어댑터.

    V2 는 같은 정보를 InputBatch 에 다른 이름으로 들고 있다. numpy 로 만들어
    두면 .tolist() 가 GPU 동기화를 일으키지 않는다.
    """

    def __init__(self, input_batch):
        import numpy as _np

        nd = input_batch.num_draft_tokens_per_req
        if nd is None:
            nd = _np.zeros(input_batch.num_reqs, dtype=_np.int32)
        self.num_draft_tokens = [int(x) for x in nd]
        self.cu_num_draft_tokens = _np.cumsum(nd, dtype=_np.int64)
        # V1 은 선행 0 을 뺀 누적을 쓴다 (starts = [0] + cu[:-1])
        self.cu_num_sampled_tokens = _np.asarray(
            input_batch.cu_num_logits_np[1:], dtype=_np.int64)
''', "훅5 헬퍼 + accept 어댑터")

# ── 훅 1: 런타임 생성 + 공급자 등록 ───────────────────────────────
sub("""        # Speculative decoding.
        self.speculator = None""",
    """        # --- DDTree (VLLM_DDTREE_BUDGET 로 켠다) — V2 포팅 ---
        self.ddtree = None
        self._ddt_pos_saved = None
        import os as _os
        _ddt = int(_os.environ.get("VLLM_DDTREE_BUDGET", "0"))
        if _ddt > 0:
            from vllm.v1.spec_decode.ddtree import DDTreeRuntime
            from vllm.v1.attention.backends import flashinfer as _fi

            self.ddtree = DDTreeRuntime(
                budget=_ddt,
                depth_limit=int(_os.environ.get("VLLM_DDTREE_DEPTH", "8")),
                topk_cap=int(_os.environ.get("VLLM_DDTREE_TOPK", "64")),
                device=device,
            )
            self.ddtree.no_accept = _os.environ.get("VLLM_DDTREE_NOACCEPT") == "1"
            self.ddtree.no_mask = _os.environ.get("VLLM_DDTREE_NOMASK") == "1"
            self.ddtree.depth_cap = int(
                _os.environ.get("VLLM_DDTREE_DEPTHCAP", str(1 << 30)))
            # V2 는 모델 드래프터만 지원한다 (ngram 은 V1 폴백) — §28
            self.ddtree.use_ngram_drafter = False
            _fi.set_ddtree_mask_provider(self.ddtree.mask_provider)
            # 마스크 2분할용 q x q 공급자 (VLLM_DDTREE_SPLIT=1 일 때만 쓰인다)
            if hasattr(_fi, "set_ddtree_local_mask_provider"):
                _fi.set_ddtree_local_mask_provider(self.ddtree.local_mask_provider)
            try:
                from vllm.model_executor.layers.mamba.gdn import (
                    qwen_gdn_linear_attn as _gdn,
                )

                _gdn.set_ddtree_gdn_provider(self.ddtree.gdn_info)
            except Exception as _e:      # 순수 어텐션 타깃이면 해당 없음
                logger.info("DDTree(V2): GDN 훅 미등록 (%s)", _e)
            logger.info("DDTree(V2) 활성: budget=%d depth=%d",
                        _ddt, self.ddtree.depth_limit)

        # Speculative decoding.
        self.speculator = None""", "훅1 런타임 생성")

# ── 훅 2: begin_step + KV 그룹 ────────────────────────────────────
sub("            block_tables, slot_mappings = self.prepare_attn(input_batch)",
    """            block_tables, slot_mappings = self.prepare_attn(input_batch)
            # --- DDTree: 이번 스텝에 어느 요청이 트리를 갖는지 확정한다.
            #     어텐션 메타데이터(마스크)를 세우기 **전**이어야 한다. ---
            if self.ddtree is not None:
                self.ddtree.begin_step(
                    input_batch.req_ids,
                    input_batch.num_reqs,
                    scheduler_output.scheduled_spec_decode_tokens,
                    input_batch.num_computed_tokens_np,
                    input_batch.query_start_loc_np,
                )
                if self.ddtree.trace_n:
                    # 노드 문맥 사후 검증용. 접두사를 추정하지 않고 vLLM 이
                    # 어텐션에 실제로 쓰는 배열에서 그대로 읽는다.
                    self.ddtree.set_prefix_probe(
                        self.req_states.all_token_ids.gpu,
                        self.req_states.num_computed_tokens.gpu,
                        input_batch.idx_mapping_np,
                    )
                # 🔴 V2 의 슬롯 매핑은 [그룹, 토큰] 텐서다. 그룹별 블록 크기와
                #    짝지어 넘긴다 (V1 은 block_table 객체 목록이었다).
                self.ddtree.groups = [
                    (slot_mappings[_g], self.block_tables.kernel_block_sizes[_g])
                    for _g in range(len(self.block_tables.kernel_block_sizes))
                ]""", "훅2 begin_step")

# ── 훅 3: 깊이 RoPE (제자리 쓰기) ─────────────────────────────────
sub("""        model_inputs = {""",
    """        # --- DDTree: 형제 노드는 같은 RoPE 위치(트리 깊이)를 갖는다 ---
        # 🔴 §26: 캡처된 그래프는 정적 버퍼 주소를 읽는다. FULL 모드는 입력을
        #    아예 안 넘기고 캡처 버퍼를 재생하므로, 새 텐서로 바꿔치기하면
        #    조용히 무시된다. 제자리로 쓰고 forward 뒤에 되돌린다.
        if self.ddtree is not None and self.ddtree.active:
            _rp = self.ddtree.rope_positions(
                input_batch.positions, input_batch.positions.shape[0])
            if _rp is not None:
                self._ddt_pos_saved = input_batch.positions.clone()
                input_batch.positions.copy_(_rp)

        model_inputs = {""", "훅3 깊이 RoPE")

# ── 훅 3의 짝: forward 직후 원복 ──────────────────────────────────
sub("""        if self.is_last_pp_rank:
            if self.use_aux_hidden_state_outputs:""",
    """        # --- DDTree: 깊이 위치를 되돌린다. 드래프터가 같은 정적 버퍼를
        #     읽으므로, 안 되돌리면 드래프트가 트리 깊이 위에서 만들어진다. ---
        if self._ddt_pos_saved is not None:
            input_batch.positions.copy_(self._ddt_pos_saved)
            self._ddt_pos_saved = None

        if self.is_last_pp_rank:
            if self.use_aux_hidden_state_outputs:""", "훅3 원복")

# ── 훅 5: 드래프터 가드 (propose 를 감싼 with 에 얹는다) ──────────
n = s.count("""            with use_workspace_lane(self._draft_workspace_lane):
                self.speculator.propose(""")
n += s.count("""            with use_workspace_lane(self._draft_workspace_lane):
                draft_tokens = self.speculator.propose(""")
for tail in ("self.speculator.propose(", "draft_tokens = self.speculator.propose("):
    old = ("            with use_workspace_lane(self._draft_workspace_lane):\n"
           f"                {tail}")
    new = ("            with use_workspace_lane(self._draft_workspace_lane), \\\n"
           "                    _ddt_drafter(self):\n"
           f"                {tail}")
    while old in s:
        s = s.replace(old, new, 1)
print(f"  적용: 훅5 드래프터 가드 ({n}곳)")
if n == 0:
    raise SystemExit("🔴 propose 호출 지점을 못 찾았다")


# ── 훅 6: 드래프터의 깊이별 logits 로 트리를 만든다 ────────────────
sub("""            self.req_states.draft_tokens[input_batch.idx_mapping] = draft_tokens""",
    """            # --- DDTree: 드래프터가 낸 평면 사슬을 버리고, 같은 깊이별
            #     분포 위에 가지를 친 트리를 대신 스케줄한다 ---
            if self.ddtree is not None and not self.ddtree.use_ngram_drafter:
                _lg = getattr(self.speculator, "draft_logits", None)
                if _lg is None:
                    raise RuntimeError(
                        "DDTree(V2): speculator.draft_logits 가 없다. "
                        "draft_sample_method='probabilistic' 이어야 깊이별 "
                        "logits 가 캐시된다 (그리디면 argmax 만 하고 버린다)."
                    )
                _n = input_batch.num_reqs
                # 🔴 draft_logits 는 num_speculative_steps 열인데, 드래프터
                #    루프를 VLLM_DDTREE_DEPTH 로 끊으면 **앞쪽 그만큼만**
                #    채워진다. 나머지는 초기값(0.0) 이라 균일분포이고, 그 위에
                #    가지를 치면 전부 쓰레기 노드가 된다.
                #    (실측: 예산16/깊이5 에서 노드 16.1개에 수용 1.61)
                _dep = int(_os_ddt.environ.get("VLLM_DDTREE_DEPTH", "0"))
                _k = self.speculator.num_speculative_steps
                if _dep:
                    _k = min(_k, max(1, _dep))
                # 🔴 draft_logits 는 [max_num_reqs, ...] 로 **영속 슬롯**
                #    (req_state_idx) 기준이다. 배치 앞 n행을 그냥 집으면
                #    채워지지 않은 슬롯을 읽는다 — 전부 0인 배열이라 topk 가
                #    가장 낮은 토큰 ID(1,3,4,6…)를 돌려주고, 트리가 통째로
                #    쓰레기가 된다. idx_mapping 으로 모아야 한다.
                _rows = _lg[input_batch.idx_mapping[:_n]][:, :_k]
                if _os_ddt.environ.get("VLLM_DDTREE_LGDBG") == "1":
                    _nz = int((_rows != 0).sum())
                    logger.info(
                        "DDTree lgdbg: shape=%s dtype=%s nonzero=%d "
                        "min=%.3f max=%.3f col0_nonzero=%d",
                        tuple(_rows.shape), _rows.dtype, _nz,
                        float(_rows.min()), float(_rows.max()),
                        int((_rows[:, 0] != 0).sum()))
                _tree = self.ddtree.propose_from_drafter_logits(
                    input_batch.req_ids[:_n], _rows.reshape(_n * _k, -1), _k)
                if _tree is not None:
                    draft_tokens = _tree
            self.req_states.draft_tokens[input_batch.idx_mapping] = draft_tokens""",
    "훅6 트리 제안")


# ── 훅 4: 트리 워크로 수용하고 수용 경로 KV 를 앞으로 당긴다 ───────
sub("""        else:
            # Rejection sampling for spec decoding.
            assert self.rejection_sampler is not None
            assert self.speculator is not None""",
    """        elif self.ddtree is not None and self.ddtree.active:
            # --- DDTree: 사슬 기각 샘플러 대신 트리 워크로 검증한다 ---
            _tok, _paths = self.ddtree.accept(logits, _DDTSpecMD(input_batch))
            self.ddtree.compact(self.kv_caches, _paths)
            # 🔴 V2 의 SamplerOutput 은 num_sampled/num_rejected 를 요구한다
            #    (V1 은 안 그랬다). 방출 토큰 수에서 역산한다.
            #    d개 드래프트를 스케줄했으면 방출은 1..d+1 이고
            #    기각 = d - (방출 - 1) 이다.
            _ns = (_tok >= 0).sum(dim=1).to(torch.int32)
            _nd = torch.as_tensor(
                input_batch.num_draft_tokens_per_req
                if input_batch.num_draft_tokens_per_req is not None
                else [0] * input_batch.num_reqs,
                dtype=torch.int32, device=_ns.device)
            sampler_output = SamplerOutput(
                sampled_token_ids=_tok,
                logprobs_tensors=None,
                num_nans=None,
                num_sampled=_ns,
                num_rejected=(_nd - (_ns - 1)).clamp_(min=0),
            )
        else:
            # Rejection sampling for spec decoding.
            assert self.rejection_sampler is not None
            assert self.speculator is not None""",
    "훅4 트리 수용 + KV 압축")

io.open(dst, "w", encoding="utf-8").write(s)
compile(s, dst, "exec")
print(f"완료 → {dst}")
