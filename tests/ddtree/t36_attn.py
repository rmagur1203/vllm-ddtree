"""T36 — §17 의 "순수 어텐션에서는 DDTree 가 이긴다" 를 cudagraph 에서 다시.

§17 은 Qwen3-0.6B + ngram 드래프터로 기준선 대비 22% 단축을 봤다. 그런데 그
측정은 enforce_eager 였고, §23 에서 eager 가 DDTree 의 GPU 추가 작업을
숨긴다는 게 드러났다 (하이브리드에서 +1.2% -> -31.6%).

같은 구성을 eager / cudagraph 양쪽에서 재고, 스텝 수를 직접 세서 '수용을
더 버는가' 와 '스텝이 더 비싼가' 를 가른다.

  DDT_MODE=base|ngram|ddtree
  DDT_EAGER=0|1     기본 0 (cudagraph)
  DDT_ARCH=attn|hybrid
  DDT_REPS=n        4프롬프트 한 바퀴를 n회
"""
import json, os, time

os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
os.environ.setdefault("VLLM_LOGGING_LEVEL", "WARNING")
# 🔴 러너를 고정하지 않는다. dflash/eagle3 는 V2 가 기본, ngram 은 V1 폴백 (§28).
if os.environ.get("DDT_RUNNER"):
    os.environ["VLLM_USE_V2_MODEL_RUNNER"] = os.environ["DDT_RUNNER"]

MODE = os.environ.get("DDT_MODE", "ddtree")
EAGER = os.environ.get("DDT_EAGER", "0") == "1"
ARCH = os.environ.get("DDT_ARCH", "attn")
# attn=Qwen3-0.6B(너무 작아 스펙 디코딩이 base 에 진다, §24-2)
# attn8=Qwen3-8B(스펙 디코딩이 cudagraph 에서도 값을 하는 크기)
_MODELS = {"attn": "Qwen/Qwen3-0.6B", "attn8": "Qwen/Qwen3-8B",
           "hybrid": "Qwen/Qwen3.5-4B"}
REPS = int(os.environ.get("DDT_REPS", "3"))
BUDGET = int(os.environ.get("DDT_BUDGET", "16"))
TAG = os.environ.get("DDT_TAG", f"{ARCH}_{MODE}_{'e' if EAGER else 'cg'}")

if MODE == "ddtree":
    os.environ["VLLM_DDTREE_BUDGET"] = str(BUDGET)
    os.environ.setdefault("VLLM_DDTREE_DEPTH", "8")
else:
    os.environ.pop("VLLM_DDTREE_BUDGET", None)

# 스텝 수를 센다 — 수용 이득과 스텝 단가를 가르려면 이게 있어야 한다 (§23-4)
# 🔴 V1/V2 러너 클래스가 다르다. 둘 다 계수해야 V2 에서 0 이 안 된다.
_STEPS = [0]


def _count_runner(mod_path, cls_name):
    try:
        import importlib
        cls = getattr(importlib.import_module(mod_path), cls_name)
    except Exception:
        return
    _o = cls.execute_model

    def _wrapped(self, *a, __o=_o, **k):
        _STEPS[0] += 1
        if not _RUNNER_SEEN:
            _RUNNER_SEEN.append(self)
        return __o(self, *a, **k)

    cls.execute_model = _wrapped


_RUNNER_SEEN = []
_count_runner("vllm.v1.worker.gpu_model_runner", "GPUModelRunner")
_count_runner("vllm.v1.worker.gpu.model_runner", "GPUModelRunner")  # V2 (모듈만 다르고 클래스명은 같다)

from vllm import LLM, SamplingParams

# 🔴 기본 4종은 **드래프터에게 최악의 표본**이다 (§35). 사실 조회, 짧은 목록,
#    문자 그대로의 반복, 피보나치 다음의 팩토리얼 — 전부 1위가 뻔하다.
#    "반복" 은 아예 반복열이다. 여기서는 형제가 놓일 자리가 없다.
EASY = [
    "The capital of France is",
    "List three prime numbers:",
    "Repeat after me: alpha beta gamma alpha beta gamma alpha beta",
    "def fibonacci(n):\n    if n < 2:\n        return n\n    return fibonacci(n-1)"
    " + fibonacci(n-2)\n\ndef factorial(n):",
]
EASY_NAMES = ["파리", "소수", "반복", "코드"]

# 드래프터 1위가 자주 틀리도록 고른 표본. 온도는 0 그대로다 — 타깃은
# 결정적이되 **드래프터가 못 맞히는** 자리를 만드는 게 목적이다.
HARD = [
    # 개방형 산문 — 정해진 다음 문장이 없다
    "Write a short story about a lighthouse keeper who finds a message"
    " in a bottle.\n\n",
    # 설명 — 어휘 선택의 자유도가 크다
    "Explain why distributed consensus is hard, in your own words:\n\n",
    # 정형화되지 않은 코드 — 보일러플레이트가 아니다
    "def merge_overlapping_intervals(intervals):\n"
    "    \"\"\"Merge overlapping [start, end) intervals and return them sorted.\"\"\"\n",
    # 고유명사·구체 사실이 계속 나온다
    "Three lesser-known facts about the history of the Portuguese language:\n1.",
    # 번역 — 대응이 하나로 안 정해진다
    "Translate to Korean, keeping the tone:\n"
    "\"The old bridge had seen better days, but it still held.\"\n\n",
    # 자유 나열 — 항목 내용이 열려 있다
    "Name six things you would pack for a week in the desert, and why:\n\n",
]
HARD_NAMES = ["소설", "설명", "코드2", "사실", "번역", "나열"]

# 🔴 문맥 길이를 실험 변수로 쓰려면 **프롬프트로** 늘려야 한다. 생성으로 늘리면
#    수용률이 다른 팔끼리 문맥이 달라져 비교가 오염된다 (수용이 무너지면 같은
#    스텝 수에서 문맥이 짧다).
_ctx = int(os.environ.get("DDT_CTX", "0"))
_FILL = ("The archive room held ledgers no one had opened in decades, each page "
         "recording a transaction whose parties were long gone. ")

_set = os.environ.get("DDT_SET", "easy")
if _set == "hard":
    PROMPTS, NAMES = HARD, HARD_NAMES
elif _set == "both":
    PROMPTS, NAMES = EASY + HARD, EASY_NAMES + HARD_NAMES
else:
    PROMPTS, NAMES = EASY, EASY_NAMES
if _ctx:
    _pad = _FILL * (_ctx // 20 + 1)      # 대략 문장당 20토큰
    PROMPTS = [_pad + p for p in PROMPTS]
_only = os.environ.get("DDT_ONLY")
if _only is not None:            # 노드 문맥 검증은 프롬프트 하나로 해야 정렬이 선다
    _i = [int(c) for c in _only]
    PROMPTS = [PROMPTS[i] for i in _i]
    NAMES = [NAMES[i] for i in _i]

kw = dict(
    model=_MODELS[ARCH],
    max_model_len=int(os.environ.get("DDT_MAXLEN", "1024")),
    attention_backend="FLASHINFER",
    gpu_memory_utilization=float(os.environ.get("DDT_UTIL", "0.20")),
    enforce_eager=EAGER,
    enable_prefix_caching=False,
    max_num_seqs=4,
    max_num_batched_tokens=int(os.environ.get("DDT_MAXLEN", "1024")),
)
if ARCH == "hybrid":
    kw["mamba_cache_mode"] = "align"
# DDT_DRAFTER=ngram | eagle3
#   ngram  : vLLM NgramProposer (V2 미지원 -> V1 폴백)
#   eagle3 : 모델 드래프터. 커버리지 100% 라 ngram 의 72% 불발이 없다 (§27-1).
DRAFTER = os.environ.get("DDT_DRAFTER", "ngram")
EAGLE3_MODEL = os.environ.get("DDT_EAGLE3", "AngelSlim/Qwen3-8B_eagle3")
if MODE in ("ddtree", "ngram"):
    if DRAFTER == "eagle3":
        kw["speculative_config"] = {
            "method": "eagle3", "model": EAGLE3_MODEL,
            "num_speculative_tokens": BUDGET,
            # 🔴 DDTree 는 깊이별 분포가 필요하다. 그리디면 speculator 가
            #    argmax 만 하고 logits 를 버려 draft_logits 가 None 이 된다.
            **({"draft_sample_method": "probabilistic"}
               if os.environ.get("DDT_DRAFT_PROB") == "1" else {}),
        }
    else:
        kw["speculative_config"] = {
            "method": "ngram", "num_speculative_tokens": BUDGET,
            "prompt_lookup_max": 4, "prompt_lookup_min": 2,
        }

if __name__ == "__main__":
    llm = LLM(**kw)
    # 🔴 러너 줄은 INFO 라 WARNING 로그에 안 남는다. 직접 기록한다 (§28).
    try:
        _rm = type(_RUNNER_SEEN[0]).__module__ if _RUNNER_SEEN else "?"
    except Exception:
        _rm = "?"

    sp = SamplingParams(temperature=0.0,
                        max_tokens=int(os.environ.get("DDT_MAXTOK", "96")))
    for p in PROMPTS:   # 워밍업 — 컴파일/캡처/오토튠이 첫 프롬프트에 섞이지 않게
        llm.generate([p], SamplingParams(temperature=0.0, max_tokens=8),
                     use_tqdm=False)

    # 🔴 워밍업이 4프롬프트를 먼저 돌린다. trace/통계를 여기서 비우지 않으면
    #    VLLM_DDTREE_TRACE 가 잡는 건 전부 워밍업 구간이다 (§31-8).
    try:
        from vllm.v1.spec_decode.ddtree import runtime as _rt0
        if _rt0.LAST is not None:
            _rt0.LAST.trace.clear()
    except Exception:
        pass

    rounds = []
    texts = None
    # 🔴 배치 1 에서는 '요청마다 도는 비용' 이 안 보인다. 운영은 max-num-seqs 64 다.
    BATCH = os.environ.get("DDT_BATCH") == "1"
    for r in range(REPS):
        per = []
        ts = []
        if BATCH:
            s0 = _STEPS[0]; t0 = time.perf_counter()
            outs = llm.generate(list(PROMPTS), sp, use_tqdm=False)
            dt = time.perf_counter() - t0
            n = sum(len(o.outputs[0].token_ids) for o in outs)
            per.append({"name": "batch", "wall": dt, "tokens": n,
                        "token_ids": [list(o.outputs[0].token_ids) for o in outs],
                        "tok_s": n / dt, "steps": _STEPS[0] - s0,
                        "ms_per_step": 1000 * dt / max(1, _STEPS[0] - s0)})
            ts = [o.outputs[0].text for o in outs]
            texts = ts
            tot = dt
            rounds.append({"per": per, "total": tot})
            print(f"  [{TAG}] {r+1}/{REPS} 배치 합계 {tot:.3f}s  {n} 토큰", flush=True)
            continue
        for name, p in zip(NAMES, PROMPTS):
            s0 = _STEPS[0]
            t0 = time.perf_counter()
            o = llm.generate([p], sp, use_tqdm=False)[0]
            dt = time.perf_counter() - t0
            n = len(o.outputs[0].token_ids)
            per.append({"name": name, "wall": dt, "tokens": n,
                        "token_ids": list(o.outputs[0].token_ids),
                        "tok_s": n / dt, "steps": _STEPS[0] - s0,
                        "ms_per_step": 1000 * dt / max(1, _STEPS[0] - s0)})
            ts.append(o.outputs[0].text)
        texts = ts
        tot = sum(x["wall"] for x in per)
        rounds.append({"per": per, "total": tot})
        print(f"  [{TAG}] {r+1}/{REPS} 합계 {tot:.3f}s  "
              + " ".join(f"{x['name']}={x['tok_s']:.0f}" for x in per), flush=True)

    tots = [x["total"] for x in rounds]
    res = {"tag": TAG, "mode": MODE, "arch": ARCH, "eager": EAGER,
           "drafter": DRAFTER, "runner_module": _rm,
           "budget": BUDGET, "rounds": rounds, "totals": tots,
           "total_best": min(tots), "total_mean": sum(tots) / len(tots),
           "texts": texts}
    try:
        from vllm.v1.spec_decode.ddtree import runtime as _rt
        if _rt.LAST is not None:
            res["stats"] = {k: v for k, v in _rt.LAST.stats.items()
                            if not isinstance(v, (dict, list))}
            if _rt.LAST.trace:      # VLLM_DDTREE_TRACE=N 일 때만 채워진다
                res["trace"] = _rt.LAST.trace
            # 구간 시간 — 오버헤드가 어디서 나오는지 귀속하려면 이게 있어야 한다
            res["t"] = dict(_rt.LAST.t)
            try:
                from vllm.v1.spec_decode.ddtree import tree as _tr
                _tr.tk_drain()
                res["tk"] = dict(_tr.TK_T)
            except Exception:
                pass
    except Exception as e:
        res["stats_error"] = repr(e)

    json.dump(res, open(f"/work/t36_{TAG}.json", "w"), indent=1)
    print(f"[{TAG}] eager={EAGER} 합계 최저 {min(tots):.3f}s 평균 "
          f"{sum(tots)/len(tots):.3f}s  {res.get('stats','')}", flush=True)
