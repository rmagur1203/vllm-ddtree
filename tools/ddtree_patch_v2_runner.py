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
s = s[:i] + "\nimport contextlib" + s[i:]
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
''', "훅5 헬퍼")

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

io.open(dst, "w", encoding="utf-8").write(s)
compile(s, dst, "exec")
print(f"완료 → {dst}")
