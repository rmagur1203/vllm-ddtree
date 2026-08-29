"""다중 요청에서 요청별 드래프트 길이가 다를 수 있는지 검증한다.

한 스텝의 드래프트 텐서는 폭이 하나지만, 스케줄러 인터페이스(DraftTokenIds)는
list[list[int]] 이라 요청마다 길이가 달라도 된다. 런타임이 요청별 길이를
기록하고(draft_lens) 모델 러너가 CPU 복사 후 행을 잘라 넘긴다.

검사:
  1. 배치 실행이 죽지 않는다
  2. 실제로 길이가 갈린다 (안 갈리면 이 기능을 시험한 게 아니다)
  3. 트리가 버려지지 않는다 (tree_underfilled / dropped)
  4. 순수 어텐션에서 base greedy 와 토큰 단위 일치
"""
import json, os, sys, time

os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
os.environ.setdefault("VLLM_LOGGING_LEVEL", "WARNING")
from vllm import LLM, SamplingParams

MODE = sys.argv[1] if len(sys.argv) > 1 else "ddtree"
BUDGET = int(os.environ.get("VLLM_DDTREE_BUDGET", "31"))
PROMPTS = [
    "The capital of France is",
    "List three prime numbers:",
    "Repeat after me: alpha beta gamma alpha beta gamma alpha beta",
    "def fibonacci(n):\n    if n < 2:\n        return n\n",
]
# DDT_ARCH=attn(0.6B+ngram, 리스트 경로) | hybrid(4B+DFlash2, 텐서 경로)
ARCH = os.environ.get("DDT_ARCH", "attn")
kw = dict(model="Qwen/Qwen3.5-4B" if ARCH == "hybrid" else "Qwen/Qwen3-0.6B",
          max_model_len=1024, attention_backend="FLASHINFER",
          gpu_memory_utilization=float(os.environ.get("VLLM_DDTREE_UTIL", "0.30")),
          enforce_eager=True, enable_prefix_caching=False,
          max_num_seqs=8, max_num_batched_tokens=2048)
if ARCH == "hybrid":
    kw["mamba_cache_mode"] = os.environ.get("VLLM_DDTREE_MAMBA", "align")
if MODE == "ddtree":
    if ARCH == "hybrid":
        kw["speculative_config"] = {"method": "dflash", "model": "/hf/drafter-4b-full",
                                    "num_speculative_tokens": BUDGET,
                                    "attention_backend": "FLASHINFER"}
    else:
        kw["speculative_config"] = {"method": "ngram", "num_speculative_tokens": BUDGET,
                                    "prompt_lookup_max": 4, "prompt_lookup_min": 2}
llm = LLM(**kw)
sp = SamplingParams(temperature=0.0, max_tokens=64)

# 🔴 배치로 한꺼번에 보낸다 — 이게 이 테스트의 핵심이다
t0 = time.perf_counter()
outs = llm.generate(PROMPTS, sp, use_tqdm=False)
dt = time.perf_counter() - t0
res = {"mode": MODE, "tokens": [list(o.outputs[0].token_ids) for o in outs],
       "wall": dt}
if MODE == "ddtree":
    from vllm.v1.spec_decode.ddtree import runtime as rt
    if rt.LAST is not None:
        res["stats"] = dict(rt.LAST.stats)
        res["len_hist"] = dict(rt.LAST.stats.get("len_hist", {}))
json.dump(res, open(f"/work/out_multi_{ARCH}_{MODE}.json", "w"))
print(f"[{MODE}] 완료  {dt:.2f}s  " + (str(res.get("stats", {}))[:120] if MODE == "ddtree" else ""))
