# DDTree on vLLM — 구현 스코핑 (2026-08-27)

---

# 결론 (2026-09-01, §29~43 요약)

이 문서는 3900줄이 넘고 **철회된 중간 결론이 많다.** 먼저 여기를 읽을 것.

## 판정: 운영 모델에 넣을 근거가 없다

| 조합 | DDTree 대 사슬 | 근거 |
|---|---|---|
| **27B GDN + DFlash2 (운영)** | **0.62x** | §42, §43 |
| 4B GDN + DFlash | 0.68x | §23, §27 |
| 8B 순수 어텐션 + EAGLE3, 배치 1 | **1.00x (동률)** | §39, §44 |
| 8B 순수 어텐션 + EAGLE3, 배치 2 / 4 / 16 | 0.98x / 0.97x / 0.96x | §41 |

**이기는 조합은 하나도 없다.** 가장 좋은 경우(순수 어텐션 + 1위가 자주 틀리는
드래프터 + 배치 1)가 동률이고, 운영은 그 셋 다 반대다. TP=2 로 올려도 비율이
그대로다 (0.627x → 0.620x, §43).

## 왜 지는가 — 세 가지 독립된 이유

1. **GDN 하이브리드는 구조적으로 불가** (§27). 트리 GDN 커널이 계층수·은닉차원에
   비례해 스텝 단가를 +44~49% 올린다. 모델을 키우면 더 커진다.
2. **좋은 드래프터에서는 분기가 헛돈다** (§34, §42). DFlash2·EAGLE3 의 1위가 이미
   충분히 맞아서 2·3위 형제가 값을 못 한다. 드래프터 1위 적중률이 33% 인 표본을
   따로 만들어야 트리가 수용에서 이긴다 (§35).
3. **배치가 커지면 폭이 공짜가 아니다** (§41). "검증 토큰 토큰당 0.18 ms" 는 배치 1
   에서 GPU 가 놀기 때문이었다. 배치 16 이면 트리 쿼리가 272 토큰이라 연산
   바운드로 가고, 최적 예산이 16 → 4 로 줄어든다. 운영은 max-num-seqs 64 다.

## 살아남은 것 (재사용 가능)

  - **V2 러너 포팅**: 훅 6곳, 패치 스크립트 2개로 재현 가능
    (`tools/ddtree_patch_v2_runner.py`, `..._speculator.py`). TP=1/2 동작 확인.
  - **RoPE 기준 버그 수정** (`756c42ad82`) — `num_computed_tokens_np` 는 낙관적
    상한이다. V1 은 CPU 도 되감아 증상이 없었고 V2 에서만 터졌다.
  - **무손실 기준 재정의** (§33): "스펙 없는 그리디와 비트 동일" 은 **사슬도 못
    지킨다.** 기준은 '같은 조건의 사슬과 동일' 이고 DDTree 는 그걸 만족한다.
  - **성능 손잡이**: 압축 퇴화 빠른경로(§36-2), 사슬 마스크 생략(§38-2),
    topk 디스패치 절감(§39-4), accept·compact 요청당 비용 제거(§40).
  - **적응형 폭** (`VLLM_DDTREE_BETA` / `RANKB`, §34) — 기본값 0 이라 무연산.

## 🔴 측정 규칙 (비싸게 배웠다)

1. **계측이 3~5배 부풀린다** (§39-2). `synchronize` 도, 호출마다 만드는
   `torch.cuda.Event` 도. **끝에서 끝까지 A/B** 와 **격리 마이크로벤치**만 믿을 것.
2. **n=3 으로는 0.5% 를 못 가른다** (§40-2). 잡음 sd 가 0.6% 다. n=10 + 부호검정.
3. **조건(배치 크기)이 틀리면 n 을 키워도 못 본다** (§40-3).
4. **구간을 끄는 ablation 은 무효** (§37-1). 무손실 보호 때문에 마스크를 끄면
   수용도 RoPE 도 압축도 연쇄로 꺼진다.
5. **강제 모양에서 잰 한계 가치는 혼합 트리로 전이되지 않는다** (§34-5).
6. **최저값으로 팔을 비교하지 말 것** (§44). 분산이 다르면 최소값 비교는 편향
   추정량이다 — DDTree 는 사슬보다 sd 가 1.6배라 최저값으로만 이겼다.
   중앙값 + 부호검정을 쓸 것.

## 철회된 결론 (본문에 배너 있음)

§17 어텐션 22% 승 / §21-2 / §22 +1.2% 승 / §25 +7.3% / §31-15 루트 문맥 오염 /
§34-6 "기계값 0 이면 8.9% 승" / §36-4 "마스크가 문맥에 비례" / §29 "포화한 예산을
분기에 쓰면 된다" / **§39-5 "8B 배치1 에서 +1.0% 승" (§44 에서 철회 — 동률)**.

---

TROUBLESHOOTING.md §33 의 "적용 불가" 결론에 이은 후속.
**포크를 가져다 쓰는 게 아니라 직접 구현할 경우** 무엇이 필요한지 정리한 문서.

참조 구현: https://github.com/liranringel/ddtree (MIT, 3.1MB, Python, master@2026-04-16)
논문: Ringel & Romano, arXiv:2604.12989 (Technion)
업스트림 요청: vllm-project/vllm#40809 (Open, 담당자·PR 없음)

로컬 클론 위치(세션 스크래치패드, 휘발됨):
`.../scratchpad/ddtree` — 필요하면 다시 `git clone --depth 1`.

---

## 1. 알고리즘 요약 (읽은 결과)

`ddtree.py` 19.5KB 가 전부. 핵심 4함수:

| 함수 | 줄 | 역할 |
|---|---|---|
| `build_ddtree_tree` | 84 | 드래프터 logits → top-k → **CPU 힙(best-first)** 으로 노드 예산만큼 트리 확장. `visibility[i,j]` 조상 마스크 동시 생성 |
| `compile_ddtree_tree` | 169 | 트리 → 평면 `input_ids` + `position_ids`(= 노드 **깊이**) + additive 마스크 `[1,1,cur,past+cur]` |
| `follow_verified_tree` | 212 | 타깃이 뽑은 토큰이 `child_maps[cur]` 에 있으면 하강. 수용 경로 인덱스 반환 |
| `compact_dynamic_cache` | 245 | 수용된 노드의 KV 만 `index_select` 로 앞으로 당김 |

힙 확장 규칙 (`ddtree.py:129-147`): 노드를 꺼낼 때마다
- **형제** 푸시 — 같은 깊이 rank+1, 가중치는 `logw - logp[rank] + logp[rank+1]`
- **자식** 푸시 — 깊이+1 rank 0, 가중치는 `logw + logp[depth][0]`

가중치는 누적 log-prob. 즉 "타깃과 일치할 확률이 높은 순"으로 노드 예산을 배분.

### 🔴 참조 구현도 T>0 무손실이 아닙니다

`follow_verified_tree` 는 **단순 토큰 매칭 워크**이고 `model/utils.py:28 sample()` 은
`temperature < 1e-5` 면 argmax. `benchmark.py:27` 기본값도 `--temperature 0.0`.
**논문/참조 어디에도 트리 rejection sampling 이 없습니다.**
T=0 은 자명하게 무손실이지만, **T>0 무손실은 우리가 새로 설계해야 합니다**
(SpecInfer 계열 multi-round tree rejection sampling).
운영은 T>0 으로 서빙하므로 이건 선택이 아니라 필수입니다.

---

## 2. 🟢 결정적 발견 — 커널을 새로 쓸 필요가 없습니다

MLX 포트가 막힌 지점이 ancestor-only 마스크 커널인데,
**설치된 FlashInfer 0.6.17 이 이미 지원합니다.**

```bash
docker exec vllm-qwen36 python3 -c "
import flashinfer, inspect
print(inspect.signature(flashinfer.BatchPrefillWithPagedKVCacheWrapper.plan).parameters.keys())"
# → ... 'custom_mask', 'packed_custom_mask', 'causal', ...
```

vLLM 이 안 넘길 뿐입니다:

```bash
docker exec vllm-qwen36 grep -c 'custom_mask' \
  /usr/local/lib/python3.12/dist-packages/vllm/v1/attention/backends/flashinfer.py
# → 0
```

**페이지드 KV 위 임의 마스크는 라이브러리가 해결해 줍니다. 배선 문제입니다.**

---

## 3. 변경 지점 (nightly `0.26.1rc1.dev1177+ga9a17e709` 기준)

| # | 파일 | 작업 | 난이도 |
|---|---|---|---|
| 1 | `v1/spec_decode/ddtree.py` (신규) | `dflash.py`(332줄) 형제. **드래프터 forward 는 그대로 재사용**. 힙+컴파일 ~200줄 포팅 | 쉬움 |
| 2 | `v1/attention/backends/flashinfer.py` | `plan()` 에 `packed_custom_mask` 전달. 스펙 배치를 prefill wrapper 로. `plan()` 호출부는 316·334·1508줄 | 중간 |
| 3 | **KV 컴팩션 (신규 커널)** | 아래 §4 | 중간 |
| 4 | `v1/sample/rejection_sampler.py` (953줄) | `rejection_greedy_sample_kernel`(715) / `rejection_random_sample_kernel`(774) 둘 다 **체인 전용**. 트리판 필요 | **T>0 이 어려움** |
| 5 | 스케줄러/블록매니저 | `k+1` 대신 `tree_budget` 슬롯 예약. 부분수용 되감기는 `scheduler.py:1875 num_computed_tokens -= num_rejected` | 쉬움 |
| 6 | CUDA 그래프 | 우리는 이미 PIECEWISE (§32-3) 이므로 **추가 손해 없음**. 가변 트리 크기라 FULL 은 어차피 불가 | 해당없음 |

---

## 4. KV 컴팩션 — 유일한 진짜 신규 커널

체인 스펙 디코딩은 수용 토큰이 **접두사**라서 되감기가 `num_computed_tokens -= n` 으로 끝납니다.
트리는 수용 경로가 트리 슬롯 범위에 **흩어져** 있어 gather 가 필요합니다.

`vllm._custom_ops` 에 있는 것 / 없는 것:
- 있음: `gather_and_maybe_dequant_cache`, `cp_gather_cache`, `swap_blocks`, `swap_blocks_batch`, `reshape_and_cache(_flash)`
- **없음: 한 요청의 슬롯 범위 내부를 압축하는 연산**

`cp_gather_cache` 계열은 캐시 **밖으로** 모으는 용도(DCP/CP)라 그대로는 안 맞습니다.
→ `(layer, src_slot, dst_slot)` gather 를 하는 Triton 커널 신규 작성. 경계는 분명함.

---

## 5. 🔴 계산을 바꾸는 숫자 — DDTree 는 배치 1 기능입니다

`benchmark.py:23` 트리 예산 기본값: **`16,32,64,128,256,512,1024` 노드**.
우리 현재 구성은 **8 노드** (k=7 + 루트).

동시 32 × 256노드 = **스텝당 쿼리 토큰 8,192개** → 매 디코드 스텝이 프리필 크기.
이건 §32 에서 이미 실측한 k 트레이드오프가 한 자릿수 증폭된 형태입니다:

| | 동시 1 | 동시 32 |
|---|---|---|
| DFlash2 k=7 (8노드) | 97.0 tok/s | 571.3 tok/s |
| DFlash2 k=3 (4노드) | 79.5 tok/s | 768.4 tok/s |

**노드를 늘리면 저동시성은 오르고 고동시성은 무너집니다.**
따라서 DDTree 도 §32-3 의 동적 k 처럼 **배치별 동적 예산**이 필수이고,
배치 5 이상에서는 평면(트리 예산 0)으로 무너뜨려야 합니다.

**이 박스에서의 상한 추정**: 논문의 DFlash 대비 +10~15%(코드 태스크) 를 그대로 받으면
동시 1에서 97 → **약 110 tok/s**, 동시 8 이상은 **0**.
실트래픽(§32-9)은 동시성이 낮지 않으므로, **운영 이득은 크지 않습니다.**
가치는 운영 성능이 아니라 업스트림 기여·학습에 있습니다.

---

## 6. 개발 환경 제약

```
GPU 0: 44362 / 49140 MiB   GPU 1: 44362 / 49140 MiB   (운영 중)
```

- 참조 구현을 이 박스에서 돌려 상한을 먼저 재는 건 **불가** — BF16 27B 타깃 ~54GB 필요.
- **그러나 정확성 개발에는 27B 가 불필요합니다.** 남은 ~4.7GB 로 작은 Qwen3 +
  더미 드래프터(랜덤 logits)를 띄우고 **"트리 검증 출력 == 비스펙 greedy 출력"**
  토큰 단위 동일성만 보면 §3 의 1~4 를 전부 개발·검증할 수 있습니다.
  수용률은 정확성과 무관합니다.
- 성능 측정은 GPU 여유가 있는 곳(spark1 등)에서 별도로.
- 🔴 **운영 컨테이너 `vllm-qwen36` 은 건드리지 않습니다.** 개발은 별도 컨테이너/포트.

---

## 7. 제안하는 단계

1. **M1 — 트리 어텐션 배선.** flashinfer 백엔드에 `packed_custom_mask` 전달.
   작은 모델 + 손으로 만든 트리로 "마스크대로 attend 하는가" 단위 검증.
   *여기까지가 전체의 핵심 리스크. 여기가 되면 나머지는 노동입니다.*
2. **M2 — T=0 무손실.** 프로포저 + `follow_verified_tree` + KV 컴팩션 커널.
   합격 기준: 같은 프롬프트에서 스펙 끄고 돌린 greedy 출력과 **토큰 단위 완전 일치**.
3. **M3 — 동적 트리 예산.** §32-3 의 `num_speculative_tokens_per_batch_size` 와 같은 형태.
4. **M4 — T>0 무손실 트리 rejection sampling.** 논문에 없음. 설계부터.
5. **M5 — 성능 측정.** `upgrade-test/bench3c.py` 재사용(오염검출 포함).

M1 이 안 되면 나머지는 의미 없으므로, **M1 부터 짧게 치고 판단**하는 것을 권합니다.


---

# 8. M1 결과 — 🟢 통과 (2026-08-27)

**트리 어텐션 배선이 됩니다. DDTree 의 커널 리스크는 없습니다.**
작업물: `ddtree-dev/` (t1/t2/t3 스크립트 + `m1-flashinfer-treemask.patch`, 124줄)

## 8-1. FlashInfer 는 페이지드 KV 위에서 트리 마스크를 정확히 지킵니다

`ddtree-dev/t1_treemask.py` — vLLM 없이 단독. 요청 3개, 과거길이·트리크기 전부 다르게,
GQA(4 qo / 2 kv), page_size 16, sm_86, fp16.

```
🟢 통과  A: bool custom_mask        r0=0.0003 r1=0.0005 r2=0.0005   최대 0.00051
🔴 불일치 B: packed_custom_mask 직접  r0=0.0003 r1=1.3693 r2=3.1064   최대 3.10642
🔴 불일치 C: 마스크 없음 (대조군)      최대 1.23216
```

레퍼런스는 torch 로 계산한 마스크드 softmax 어텐션. **MLX 포트가 막힌 커널 문제는
CUDA 쪽엔 존재하지 않습니다.**

## 8-2. 🔴 `packed_custom_mask` 를 직접 넘기면 안 됩니다

원인은 정렬이 아니라 **단위 불일치**입니다 (`t2_align.py` 로 정렬 가설은 기각됨 —
q*kv 를 8의 배수로 맞춰도 여전히 실패).

```
plan() 의 _compute_page_mask_indptr  → 원소 누적합  [0, 441, 1081, 1137]
segment_packbits 가 반환하는 indptr   → 바이트 오프셋 [0,  56,  136,  143]   ← 커널이 쓰는 값
```

`plan()` 은 `packed_custom_mask` 를 받으면 `segment_packbits` 단계를 건너뛰므로
원소 누적합이 그대로 커널에 들어갑니다. 오프셋 0인 **요청 0만 우연히 맞습니다.**
→ **bool `custom_mask` 를 넘기고 FlashInfer 가 `segment_packbits` 하게 둡니다.**

## 8-3. 🔴 마스크를 쓰면 indptr 을 CUDA 로 넘겨야 합니다

`segment_packbits` 는 CUDA indptr 을 요구합니다 (CPU 면
`RuntimeError: input_indptr must be a CUDA tensor`).
vLLM 은 `qo_indptr_prefill_cpu` 등 CPU 텐서를 넘기므로, 마스크가 있을 때만
`.to(device)` 로 바꿔 넘깁니다. `plan()` 이 내부에서 다시 CPU 로 내리므로 동작에는 지장 없음.

## 8-4. 🟢 스펙 검증 토큰은 prefill 경로를 탑니다 — DDTree 에 결정적

이게 안 되면 DDTree 는 불가능했습니다. decode wrapper 는 custom_mask 를 지원하지 않으니까요.

`flashinfer.py:896` 의 `supports_spec_as_decode` 는 trtllm-gen 디코드 커널이나
dedicated XQA 일 때만 True 입니다. **sm_86 에서는 둘 다 아니므로 False** →
`reorder_batch_threshold` 가 1로 남고 → q_len>1 인 스펙 쿼리가 prefill wrapper 로 갑니다.

실측 (`t3_vllm_wire.py`, ngram k=3): **관측된 q_len `[4, 5]`** — 예상대로 k+1=4.

## 8-5. 배선 검증 (엔드투엔드)

패치한 백엔드로 Qwen3-0.6B + ngram 스펙 디코딩을 돌리고:

| 검증 | 방법 | 결과 |
|---|---|---|
| A | causal 과 **동등한** 마스크 주입 → 출력이 마스크 없을 때와 완전히 같아야 함 | 🟢 두 프롬프트 모두 문자 단위 일치 |
| B | 첫 키를 가림 → 출력이 달라져야 함 | 🟢 변함 |

A 가 통과했다는 건 인덱싱·요청별 오프셋·페이지 레이아웃·GQA 가 전부 맞다는 뜻입니다.

## 8-6. 패치 내용 (`m1-flashinfer-treemask.patch`, 5개 hunk)

1. `set_ddtree_mask_provider(fn)` 훅 + `_ddtree_build_mask()` — 프로포저가 스텝마다 마스크 공급
2. `FlashInferMetadata.ddtree_mask_active: bool = False` 필드 추가
3. prefill `plan()` 에 `custom_mask=` 전달, 마스크 있으면 indptr 을 CUDA 로 + `causal=False`
4. plan 직후 `ddtree_mask_active` 세팅
5. `forward()` 의 `assert prefill_wrapper._causal == attn_metadata.causal` 를
   `or attn_metadata.ddtree_mask_active` 로 완화

## 8-7. 개발 환경 메모

- 🔴 `VLLM_ATTENTION_BACKEND` 환경변수는 **nightly 에서 제거됐습니다**
  (`Unknown vLLM environment variable detected`). `LLM(attention_backend="FLASHINFER")`
  또는 `--attention-backend` 를 쓰세요.
- 🔴 훅이 모듈 전역이라 `VLLM_ENABLE_V1_MULTIPROCESSING=0` 이 필요합니다.
  기본값에서는 엔진이 spawn 된 별도 프로세스라 부모의 전역이 안 보입니다.
  (실제 통합에서는 프로포저가 워커 안에 있으므로 해당 없음)
- 개발 컨테이너: `gpu_memory_utilization=0.065` (운영이 GPU0 를 44.4/49.1 GiB 쓰는 중),
  `--gpus '"device=0"'` (GPU1 은 §29 열 문제).
- 재현:
  ```
  docker run --rm --gpus '"device=0"' -v $PWD/ddtree-dev:/work -w /work \
    -v $PWD/hf-cache/huggingface:/hf -e HF_HOME=/hf \
    -v $PWD/ddtree-dev/patched/flashinfer.py:/usr/local/lib/python3.12/dist-packages/vllm/v1/attention/backends/flashinfer.py:ro \
    --entrypoint python3 vllm/vllm-openai:nightly t3_vllm_wire.py
  ```

## 8-8. 다음 (M2)

남은 것은 §3 의 1·3·4 입니다. 커널 리스크가 없으므로 전부 파이썬 + Triton:
- `v1/spec_decode/ddtree.py` — 힙 트리 빌드 + 마스크 공급자 등록
- KV 컴팩션 Triton 커널 (수용 경로만 앞으로 당기기)
- T=0 트리 검증 → 합격 기준: 스펙 끈 greedy 출력과 토큰 단위 완전 일치


---

# 9. M2 결과 — 구성요소 3/3 검증 완료, 러너 통합은 미완 (2026-08-27)

작업물: `ddtree-dev/ddtree_tree.py`, `ddtree-dev/ddtree_compact.py`,
테스트 `t4`/`t5`/`t6`, 참조 대조용 `ddtree-dev/ref/` (MIT, vendored).

## 9-1. 🟢 트리 빌드 — 참조 구현과 완전 동일 (`t4_tree_build.py`)

같은 logits 에서 `token_ids` / `depths` / `parents` / `child_maps` / `visibility` 가
전부 일치해야 통과. **12/12 일치.**

경계값 포함: 실제 Qwen3 어휘 151,936 / budget=1 / depth_limit=1 / 깊은 트리(depth 15, budget 512).

참조 구현은 `datasets`·`flash-attn` 까지 끌고 오므로, `build_ddtree_tree` 함수만
AST 로 떼어내 실행합니다 (`t4` 상단). 트리 빌드 자체는 heapq/numpy/torch 만 씁니다.

## 9-2. 🟢 KV 컴팩션 — 7/7 오차 0 (`t5_compact.py`, `ddtree_compact.py`)

**경쟁 조건이 진짜 있습니다.** `dst[i] = base+i`, `src[i] = base+a[i]` (a 증가, `a[i] >= i`)
이므로 `dst[i] <= src[i]` 인데, `a = [0,2,3]` 같은 경우 `dst[2] == src[1]` 이라
순진한 병렬 커널은 아직 안 읽은 소스를 덮어씁니다.

해결: `i' > i` 이면 `src[i'] > dst[i]` 이므로 **행을 증가 순서로 순차 처리**하면 안전합니다.
프로그램 하나가 (레이어, 요청, 피처블록) 을 맡아 행을 루프하고, 병렬은 그 세 축으로만 냅니다.
**임시 버퍼가 필요 없습니다.**

검증 케이스: 겹침 없음 / 🔴겹침 발생 / 전부 수용(no-op) / 루트만 수용 /
페이지 경계 걸침 / 배치 3요청 / 64레이어 큰 트리 — 전부 Triton == torch 레퍼런스 == 독립 기대값.

메모: Triton 의 int64 → `tl.pointer_type` 캐스팅이 동작하므로 **레이어 전체를 한 번에
런치**할 수 있습니다 (레이어당 런치 불필요). 지금은 fp16 만 — 운영은 fp8 KV 라
M5 전에 dtype 분기 필요.

## 9-3. 🟢 트리 검증이 T=0 에서 무손실 (`t6_lossless.py`)

타깃을 결정적 함수로 두고 **드래프터를 완전 난수**로 넣어도
평범한 greedy 결과와 토큰 단위로 같아야 한다 — **12/12 일치.**

드래프터 품질은 수용률에만 영향, 출력에는 영향 없음이 확인됩니다
(예산 4→128 에서 스텝당 수용 0.00→0.25).

## 9-4. 🔴 남은 것 — 러너 통합. 진짜 비용은 여기 있습니다

커널도, 컴팩션도, 검증 로직도 아니었습니다. **vLLM 의 입력 준비가
"위치당 토큰 하나"를 전제한다는 것**이 실제 장벽입니다.

`gpu_model_runner.py:2244`

```python
self.positions[:total] = num_computed_tokens[req_indices] + query_pos
```

이 `self.positions` 가 **두 군데에 동시에** 쓰입니다:

| 쓰임 | 위치 | 트리에 필요한 값 |
|---|---|---|
| `compute_slot_mapping` (KV 를 어디 쓸지) | `:2254` | **노드 인덱스** (0..N-1, 연속) |
| 모델 forward 의 RoPE | `:3724` | **노드 깊이** (형제끼리 같은 값) |

트리는 형제 노드가 같은 깊이를 가지므로 이 둘이 갈라집니다.

### 해법 — 두 값을 분리하면 나머지는 그대로 굴러갑니다

`self.positions` 는 **연속(체인)으로 두고** slot_mapping·블록할당·KV 기록·
수용 후 회계가 전부 기존 경로를 타게 합니다. RoPE 용으로만 깊이 텐서를 따로 만들어
`:3724` 에서 갈아끼웁니다. **KV 컴팩션(§9-2)이 이 트릭을 성립시킵니다** — 수용 경로를
앞으로 당겨 놓으면 "노드 N개를 쓰고 j개만 남겼다"가 체인과 구분되지 않기 때문입니다.

### 통합 작업 목록 (= M3)

1. `self.ddtree_rope_positions` 추가 + `:3724` 에서 치환 — **작음**
2. 트리 노드 토큰을 input_ids 에 주입 (`_prepare_input_ids`, 기존 드래프트 주입 경로 재사용) — **작음**
3. 트리 인식 수용 (`rejection_sampler` 호출부 `:3777` 분기, T=0 greedy 부터) — **중간**
4. 컴팩션 훅 + `num_computed_tokens += j` — **중간**
5. `v1/spec_decode/ddtree.py` 프로포저 (드래프터 logits → 트리 → 마스크 공급자 등록) — **중간**

`SpecDecodeMetadata` 는 체인 모양(`draft_token_ids` 평탄 + `num_draft_tokens`)이라
`parents` 를 실을 필드가 없습니다. 확장 필요.

## 9-5. 현재 상태 요약

| 조각 | 상태 | 근거 |
|---|---|---|
| 트리 어텐션 커널 | 🟢 완료 | M1, FlashInfer custom_mask |
| 백엔드 배선 | 🟢 완료 | M1 패치 124줄, 엔드투엔드 검증 |
| 스펙 토큰이 prefill 경로 | 🟢 확인 | sm_86 에서 q_len [4,5] 관측 |
| 트리 빌드 | 🟢 완료 | 참조와 12/12 동일 |
| KV 컴팩션 | 🟢 완료 | 7/7 오차 0 |
| T=0 무손실 로직 | 🟢 완료 | 12/12 greedy 일치 |
| **러너 통합** | 🔴 **미완** | §9-4 |
| T>0 무손실 | 🔴 미착수 | 논문에도 없음 (§1) |

**"실모델에서 greedy 와 토큰 단위 일치"라는 M2 합격 기준은 아직 충족되지 않았습니다.**
알고리즘 수준에서는 증명됐고, 실모델 검증은 러너 통합(M3) 이후에 가능합니다.


---

# 10. M3 — 러너 통합 (진행 중, 2026-08-27)

작업물: `ddtree-dev/ddtree_runtime.py` (상태 기계), `ddtree-dev/m3-runner-ddtree.patch`,
테스트 `t7_ddtree_e2e.py`.

## 10-1. 통합 설계 — 훅 5개, 러너 diff 89줄

로직은 전부 `ddtree_runtime.DDTreeRuntime` 에 모으고 러너에는 훅만 넣었습니다.

| 훅 | 위치 | 내용 |
|---|---|---|
| A | `__init__` (`self.kv_caches` 뒤) | `VLLM_DDTREE_BUDGET` 으로 켜고 마스크 공급자 등록 |
| B | `_prepare_inputs`, `compute_slot_mapping` 뒤 | `begin_step()` — 이번 스텝 트리 확정, slot_mapping·block_size 캡처 |
| C | forward 직전 `positions = self.positions[...]` | 트리 구간만 **깊이**로 치환 |
| D | `_sample`, `rejection_sampler` 앞 | 트리 워크 수용 + KV 컴팩션 |
| E | `propose_draft_token_ids` 진입부 | 체인 드래프터 대신 트리 생성 |

**§9-4 의 트릭이 성립했습니다**: `self.positions` 를 연속으로 두니
slot_mapping·블록할당·KV 기록·`num_computed_tokens` 회계가 전부 기존 경로를 그대로 탑니다.
훅 2번(토큰 주입)은 **불필요**했습니다 — `_calc_spec_decode_metadata:2983` 이
드래프트를 `self.input_ids.gpu[logits_indices]` 에서 뽑기 때문에, 트리 노드를
힙 순서 평탄 리스트로 내보내면 기존 경로가 그대로 처리합니다.

드래프터는 테스트용 **ngram 트리**입니다: 마지막 n-gram 이 과거에 나온 모든 위치를 찾아
깊이별 후속 토큰 **분포**를 만듭니다. 일치가 여러 개면 그게 드래프터의 불확실성이고
DDTree 는 그 위에 가지를 칩니다. 일치가 하나면 체인(= 평범한 ngram)이 됩니다.

## 10-2. 동작은 합니다

Qwen3-0.6B, budget=16, 프롬프트 4개:
`{'steps': 94, 'tree_steps': 51, 'accepted': 177, 'nodes': 992}`
— 트리가 만들어지고, 마스크가 붙고, 검증이 돌고, 컴팩션이 실행됩니다.

## 10-3. 🔴 미해결 — 배치에서 출력이 갈립니다

| 조건 | 결과 |
|---|---|
| 프롬프트 0 **단독** | 🟢 96토큰 완전 일치 |
| 프롬프트 4개 **동시** | 🔴 0번·3번 불일치 (1·2번은 일치) |
| 4개 동시 + 분기 끔(topk=1, 순수 사슬) | 🔴 동일하게 0번·3번 불일치 |
| 4개 동시 + **수용 완전 차단**(루트 토큰만) | 🔴 동일하게 0번·3번 불일치 |

**수용을 꺼도 깨진다**는 게 결정적입니다. 루트 노드는 과거만 보므로
그 argmax 는 항상 옳아야 합니다. 즉 버그는 수용 로직이 아니라
**과거 KV 를 오염시키는 배선**에 있고, 단일 요청에서는 안 나타납니다.

### 확인된 원인 하나 (수정함)

세그먼트 → 요청 매핑을 "디코드가 앞"이라는 배치 순서 가정으로 했는데,
**트리 크기가 모두 같으면 어긋나도 알아챌 수 없습니다** (budget 이 같으면 q_len 도 같음).
`(q_len, kv_len)` 서명으로 직접 찾고, 중복 서명은 모호하므로 causal 로 물러서게 바꿨습니다.

### 함께 고친 무손실 구멍 두 개

1. **트리가 드롭된 요청에 체인 수용을 적용하면 안 됩니다.** 드래프트가 힙 순서
   트리 노드라 사슬이 아니어서, 형제를 후속으로 오인합니다. 루트 토큰만 내보내도록 수정.
2. **트리 마스크를 실제로 받은 요청만 트리 워크를 해야 합니다.** causal 마스크로
   계산된 logits 로 트리를 내려가면 greedy 가 아닌 토큰을 수용할 수 있습니다.
   `masked_trees` 로 마스크와 수용을 결합했습니다.

## 10-4. 개발 환경 — 메모리가 빡빡합니다

운영이 GPU0 를 45.5/49.1 GiB 쓰고 있어 개발 인스턴스에 남는 게 **2.7 GiB** 뿐입니다.
`max_model_len=1024`, `max_num_batched_tokens=1024`, `max_num_seqs=4`,
`gpu_memory_utilization=0.048` 로 겨우 들어갑니다. 운영 트래픽에 따라 변동합니다.

🔴 **툴 타임아웃으로 잘린 `docker run` 이 컨테이너를 남깁니다** — GPU 를 1.9 GiB 씩
붙들고 있다가 다음 실행을 OOM 시킵니다. 긴 실행은 반드시 백그라운드로 돌리고,
실패하면 `docker ps` 로 고아 컨테이너를 먼저 확인하세요.

🔴 §29 의 GPU1 열 문제가 진행 중입니다 (실측 92°C). 개발은 GPU0 에서만 하고,
불필요한 실행을 줄이세요.


## 10-5. 🔴 합격 기준이 틀렸습니다 — 배치에서는 vLLM 자체도 갈립니다

배치에서만 출력이 갈리는 원인을 찾느라 가설 셋을 세웠고 **셋 다 틀렸습니다.**
기록해 둡니다. 같은 길을 다시 가지 않기 위해서입니다.

| 가설 | 어떻게 반증됐나 |
|---|---|
| 세그먼트→요청 매핑이 배치 순서 가정 때문에 어긋난다 | `(q,k)` 서명으로 바꾸니 4개 중 통과가 2→3개로 올랐지만 **원인은 아니었음** |
| 깊이 RoPE 와 트리 마스크가 다른 집합에 적용된다 | 마스크 받은 트리로 제한 → `rope_skipped=0`, **통계·출력 완전 동일** |
| cascade wrapper 경로에 훅이 없어 마스크가 빠진다 | `rope_skipped=0` 이 곧 "모든 트리가 마스크를 받았다"는 뜻 → 해당 없음 |

### 결정적 실험 — NOMASK

마스크 자체를 끄면 (`VLLM_DDTREE_NOMASK=1`) 마스크·깊이 RoPE·트리 수용·컴팩션이
**전부 비활성**됩니다 (`masked=0, accepted=0, rope_skipped=48`).
DDTree 의 유일한 효과는 "힙 순서 드래프트 16개를 스케줄하고 1개만 수용"이고,
이건 vLLM 체인 스펙 디코딩과 동등합니다. **그런데도 깨졌습니다.**

즉 제가 의심한 곳은 전부 원인이 아니었습니다.

### 진짜 원인 — 부동소수점

```
기준선 2회 반복                      : 🟢 🟢   (비결정성 아님)
vLLM '자체' ngram 체인 스펙 디코딩   : 🟢 🔴   ← 기준선과 갈림
```

**vLLM 자체 스펙 디코딩도 비스펙 greedy 와 갈립니다.**
배치 크기 2에서 기준선은 스텝당 쿼리 2개, 스펙은 18~34개입니다. 행렬 곱의 축약
형상이 달라지고 bf16 에서 상위 두 토큰이 근접하면 argmax 가 뒤집힙니다.
프롬프트 1·2 는 통과하고 0·3 만 깨진 것도 "근접 케이스가 있는 프롬프트만 갈린다"와
일치합니다.

**"비스펙 greedy 와 토큰 단위 일치"는 배치 스펙 디코딩의 합격 기준이 될 수 없습니다.**
제 구현에서도, vLLM 자체에서도 성립하지 않습니다.

### 올바른 합격 기준

**요청 단독 실행에서 비트 단위 일치.** 배칭 잡음이 없어 결정적입니다.
배치 동작은 "vLLM 자체 스펙 디코딩과 같은 정도로만 갈린다"로 확인합니다.

## 10-6. 남은 취약점 — 매핑을 서명에 의존하는 것

`(q_len, kv_len)` 서명은 휴리스틱입니다. 트리가 아닌 요청이 우연히 같은 서명을 가지면
그쪽에 트리 마스크가 갑니다. 실측에서 `ambiguous` 는 배치 2요청이면 0 이지만,
4요청 + 매 스텝 트리인 조건에서는 30까지 올라갑니다 (중복은 causal 로 물러섬).

**제대로 된 해법**: 모듈 전역 공급자 대신 **`CommonAttentionMetadata` 에 트리 정보를
실어 보내는 것**입니다. 세그먼트↔요청 매핑이 vLLM 자신의 것과 같아져 추측이 사라집니다.
업스트림 PR 이라면 이 형태여야 합니다.


## 10-7. 🟢 M3 결과 — 요청 단독 실행에서 비트 단위 무손실 (4/4)

Qwen3-0.6B, budget=16, depth_limit=8, 96토큰, 비스펙 greedy 대비:

| 프롬프트 | 결과 | 트리 스텝 | 수용 |
|---|---|---|---|
| "The capital of France is" | 🟢 96토큰 비트 단위 일치 | 10 | 61/160 (38.1%) |
| "List three prime numbers:" | 🟢 일치 | 31 | 2/496 (0.4%) |
| "Repeat after me: alpha beta gamma…" | 🟢 일치 | 11 | 86/176 (48.9%) |
| 코드 (fibonacci/factorial) | 🟢 일치 | 9 | 16/144 (11.1%) |

수용률 편차가 큰 건 **테스트용 ngram 트리 드래프터** 탓입니다 (DFlash2 가 아님).
반복적인 프롬프트에서 48.9%, 소수 나열에서 0.4%. 무손실 검증에는 무관합니다 —
드래프터 품질은 수용률에만 영향을 주고 출력에는 영향을 주지 않는다는 것이
§9-3 에서 이미 확인됐습니다.

**배치 실행은 판정에 쓰지 않습니다.** vLLM 자체 스펙 디코딩도 같은 정도로
갈리기 때문입니다 (§10-5).

## 10-8. M3 최종 상태

| 조각 | 상태 | 근거 |
|---|---|---|
| 트리 어텐션 커널 | 🟢 | M1 — FlashInfer custom_mask, 오차 0.0005 |
| 백엔드 배선 | 🟢 | M1 패치 124줄, 엔드투엔드 |
| 스펙 토큰이 prefill 경로 | 🟢 | sm_86 에서 q_len [4,5] 관측 |
| 트리 빌드 | 🟢 | 참조 구현과 12/12 동일 |
| KV 컴팩션 | 🟢 | 7/7 오차 0 (경쟁 조건 케이스 포함) |
| T=0 무손실 로직 | 🟢 | 시뮬레이션 12/12 |
| **러너 통합** | 🟢 | 러너 diff 93줄, 훅 5개 |
| **실모델 무손실** | 🟢 | **단독 4/4 비트 단위 일치** |
| T>0 무손실 | 🔴 미착수 | 논문에도 없음 (§1) |
| 성능 측정 | 🔴 미착수 | M5 |

### 산출물

```
ddtree-dev/
  ddtree_tree.py            트리 빌드 (참조와 동일 검증)
  ddtree_compact.py         KV 컴팩션 Triton 커널 (dtype 무관, 바이트 단위)
  ddtree_runtime.py         상태 기계 — 러너 훅이 부르는 전부
  m1-flashinfer-treemask.patch   124줄
  m3-runner-ddtree.patch          93줄
  t1..t8                    단위·통합 테스트
  ref/                      liranringel/ddtree (MIT) 대조용
```

### 다음 (M4/M5)

1. **T>0 무손실 트리 rejection sampling** — 논문에 없음. 설계부터.
2. **매핑을 `CommonAttentionMetadata` 로** (§10-6) — 업스트림 품질에 필수.
3. **DFlash2 드래프터 연결** — 지금은 테스트용 ngram 트리. 실제 수용률을 보려면 필요.
4. **동적 트리 예산** — §5 의 배치 1 편향 때문에 필수.
5. **성능 측정** — `upgrade-test/bench3c.py` 재사용.


---

# 11. M4 — DFlash2 드래프터 연결 (코드 완료, 하드웨어 미검증, 2026-08-27)

작업물: `ddtree-dev/m4-dflash2-decouple.patch` (118줄, 3파일).

## 11-1. 🔴 구조적 차단 — 트리 예산이 7노드로 묶여 있었습니다

`model_executor/models/qwen3_dflash2.py:131`

```python
block_size=1 + speculative_config.num_speculative_tokens,   # 드래프터 conv 의 블록 폭
```

그런데 vLLM 은 요청당 정확히 `num_speculative_tokens` 개의 드래프트를 스케줄합니다.
DFlash2 체크포인트는 `dflash_config.block_size = 8` 로 학습됐으므로,
이 결합 하에서는 **`num_speculative_tokens = 7` → 트리 예산도 7노드**입니다.
논문의 예산은 16~1024 입니다 (§5).

### 🔴 게다가 검사가 없습니다

`config/speculative.py:174` 의 block_size 일치 검사는 **DSpark 전용**입니다
(`_get_qwen3_dspark_value`). DFlash 는 검사하지 않습니다.

→ 예산을 16으로 키우면 **크래시 없이 돕니다.** 다만 conv 가 17폭으로 동작해
학습된 8과 어긋나고, **드래프트 품질만 조용히 나빠집니다.**

이것이 AEON-7 의 *"DDTree mode is not faster yet"* (§33-4) 를 설명할 가능성이 있습니다.
결합을 안 풀면 트리가 7노드로 묶이거나, 어긋난 블록으로 드래프트를 뽑게 됩니다.

### 분리 패치

| 개념 | 값 | 출처 |
|---|---|---|
| `drafter_k` — 드래프터가 한 번에 내놓는 위치 수 | 7 | 체크포인트 `dflash_config.block_size - 1` |
| `num_speculative_tokens` — 요청당 스케줄되는 드래프트 수 = **트리 예산** | 16~1024 | 사용자 설정 |

- `qwen3_dflash2.py`: conv block_size 를 체크포인트에서 읽음
- `dflash.py`: `self.drafter_k` 도입, 쿼리·마스크 토큰 수를 그걸로 계산 (4곳)

`propose()` 는 호출마다 `self.num_speculative_tokens` 를 덮어쓰지만
(`llm_base_proposer.py:541`), `drafter_k` 는 `__init__` 에서 체크포인트로부터
한 번만 계산되므로 영향받지 않습니다.

## 11-2. T=0 에서는 드래프터 분포가 안 나옵니다

`take_last_draft_probs()` 는 `_enable_probabilistic_draft_probs`
(= `rejection_sample_method == "standard"` **and** `draft_sample_method == "probabilistic"`)
이고 `all_greedy` 가 아닐 때만 채워집니다. T=0 이면 항상 `None` 입니다.

DDTree 는 T=0 에서도 깊이별 분포가 필요하므로 `_greedy_sample` 에서 **logits 를
가로채도록** 패치하고 `take_last_draft_logits()` 를 추가했습니다.
반환 형상은 `[batch * drafter_k, vocab]`, 요청 우선(요청당 k행 연속)입니다 —
`llm_base_proposer.py:455` 주석이 이 순서를 명시합니다.

## 11-3. 소비 쪽

`DDTreeRuntime.propose_from_drafter_logits(req_ids, logits, drafter_k)`.
드래프터가 **정상적으로 돈 뒤** 러너의 `propose_draft_token_ids` 끝에서 가로챕니다
(propose() 인자가 복잡해 재구성하지 않는 편이 안전합니다).
드래프터가 돌려준 평면 체인 드래프트는 버리고, 같은 분포 위에 가지를 친 트리를
대신 스케줄합니다.

`self.ddtree.use_ngram_drafter` 로 두 경로를 가릅니다 —
`method == "dflash"` 면 실제 드래프터, 아니면 테스트용 ngram 트리.

🔴 요청마다 topk 를 CPU 로 내리므로 배치가 크면 동기화 비용이 큽니다.
성능 측정(M5) 전에 배치 topk + 단일 D2H 로 바꿔야 합니다.

## 11-4. 🔴 하드웨어 검증 불가 — 이 박스에서는 못 돌립니다

| | 크기 |
|---|---|
| 가장 작은 DFlash 쌍: `Qwen/Qwen3.5-4B` + `z-lab/Qwen3.5-4B-DFlash` | 9.32 + 1.27 = **10.6 GB** |
| GPU0 여유 (운영이 45.5/49.1 GiB 사용 중) | **3.6 GB** |

INT4 로 낮춰도 안 들어갑니다.
`load_format="dummy"` 는 전역이라 타깃까지 랜덤이 되는데, 랜덤 타깃은 로짓이 거의
균일해 근접 동점이 폭증하므로 무손실 판정에 쓸 수 없습니다.

**§11 의 패치들은 하드웨어에서 검증되지 않았습니다.** 확인한 것은:
- 세 파일 모두 구문 검사 통과
- 5개 파일 패치 스택으로 **기존 ngram 경로 회귀 없음** (프롬프트 0·2 비트 단위 일치)

실행하려면: **spark1**(27B + DFlash2 보유, GPU 여유 있음) / 이 박스 점검 창 / 보류.

## 11-5. 패치 목록

```
m1-flashinfer-treemask.patch   124줄  flashinfer 백엔드 — 트리 마스크 배선
m3-runner-ddtree.patch         117줄  gpu_model_runner — 훅 6개
m4-dflash2-decouple.patch      118줄  dflash / llm_base_proposer / qwen3_dflash2
```


---

# 12. M4/M5 — DFlash2 연결과 하이브리드 대응 (2026-08-27, 미완)

운영을 내리고 실제 Qwen3.8-27B + DFlash2 로 진행. **무손실 미달성.**
아래는 재조사를 반복하지 않기 위한 기록.

## 12-1. 구조적 발견 넷 — 오늘의 본체

| 발견 | 근거 | 영향 |
|---|---|---|
| **DFlash2 는 Model Runner V2 경로** | `config/vllm.py:694` 폴백 로그. `VLLM_USE_V2_MODEL_RUNNER=0` 로 V1 강제 가능 | M3 훅이 전부 V1 에 있어 무효였음. ngram 은 V2 미지원이라 V1 폴백 → 검증 경로 ≠ 실제 경로 |
| **트리 예산이 drafter_k 에 묶임** | `qwen3_dflash2.py:131` `block_size = 1 + num_speculative_tokens`, 체크포인트는 8 | 손 안 대면 예산 ≤ 7. 논문은 16~1024. **DFlash 에는 검증도 없음** (`speculative.py:174` 는 DSpark 전용) → 예산을 키우면 conv 가 조용히 어긋난 폭으로 동작 |
| **Qwen3.8-27B 는 하이브리드** | `qwen3_5.py`: `IsHybrid` + `QwenGatedDeltaNetAttention` | 재귀 계층은 어텐션 마스크로 형제를 못 막음. DDTree 의 전제가 깨짐 |
| **GDN 은 융합 CUDA 커널을 씀** | `VLLM_GDN_DECODE_KERNEL` 기본 `cuda` → `ops.fused_gdn_decode_post_conv_mtp` | `_forward_core` 의 Triton 경로는 **죽은 코드**. 파이썬 패치로는 못 건드림 → vLLM 재빌드 필요 |

## 12-2. 함정 둘

- **비동기 스케줄링의 `num_computed_tokens_cpu` 는 낙관값**이다. 스케줄한 만큼 먼저
  올리고 거부분은 나중에 되감는다 (`gpu_model_runner.py:1603`). 그래서 kv_len 을
  매칭 키로 쓰면 안 된다 (실측: 기대 (17,45) vs 실제 (17,31)).
  ngram 은 동기 스케줄링이라 안 드러난다.
- **드래프트 반환 계약이 경로마다 다르다.** ngram 은 `list[list[int]]`,
  DFlash 는 **GPU 텐서** `[batch, num_spec]` (`gpu_model_runner.py:1979` 단언).

## 12-3. 분기에서만 터지는 버그들 — 전부 같은 뿌리

**"수용 경로가 접두사가 아니다"** 에서 나온다. 사슬에서는 우연히 일치해서 안 드러난다.

| 버그 | 위치 |
|---|---|
| SSM 초기 상태를 `num_accepted-1` 에서 적재 | `fused_sigmoid_gating.py` 커널 |
| conv 히스토리를 0으로 지움 (굴리지 않음) | 내 `fixup_history` |
| 드래프터가 트리 마스크를 가져감 (폭이 같아 구분 불가) | 내 마스크 공급자 |
| Mamba precopy `src_off = num_accepted-1` | `worker/mamba_utils.py:457` |

**올바른 해법은 개별 대응이 아니라 "수용 후 접두사로 만들기"** — 어텐션 KV 컴팩션과
같은 원칙을 GDN 상태에도 적용하는 것 (`ddtree_gdn.compact_gdn`).

## 12-4. 실측

| 설정 | 무손실 | tok/s | 비고 |
|---|---|---|---|
| base (스펙 없음) | 기준 | 10.4 / 12.7 | eager, TP=1 |
| **flat — 네이티브 DFlash2 k=7** | 🟢 | **17.6 / 63.3** | 파이프라인 정상 |
| tree7chain (CUDA 커널, GDN 패치 비활성) | 🟢 | 16.4 / 59.7 | 수용 51% |
| tree7 (CUDA 커널, 분기) | 🔴 | 18.6 / 53.6 | |
| tree7chain (**Triton 커널, GDN 패치 활성**) | 🔴 | 14.0 / 33.4 | **사슬인데도 깨짐** |

마지막 줄이 결론이다: **내 GDN 트리 구현은 사슬로도 환원되지 않는다.**
그 앞의 `tree7chain` 🟢 는 GDN 패치가 죽은 코드여서 나온 결과였다.

유력한 원인: SSM 커널에서 같은 프로그램 내 `tl.store` → `tl.load` 로 부모 상태를
읽는 부분. t10 은 K=V=16 으로 통과했지만 실모델은 head_dim 128 이라 `b_h` 가
[32,128] 로 여러 워프에 걸친다. `tl.debug_barrier()` 는 전역 메모리 가시성을
보장하지 않는다.

## 12-5. 방법론 실수 (반복하지 말 것)

1. **폴백 경로에 카운터를 안 붙였다.** GDN 훅이 한 번도 호출되지 않았는데
   (`gdn_calls: 0`) 세 라운드를 추측으로 고쳤다. 계측을 붙인 첫 실행에서 즉시 드러났다.
2. **커널을 바꾸고 기준선은 안 바꿨다.** tree 만 Triton 으로 돌리고 CUDA 기준선과
   비교해 무효한 판정을 냈다.
3. **실패한 실행이 낡은 JSON 으로 위장**해 `+519%` 같은 허수를 보고했다.
   실행 직전 해당 모드 JSON 을 지우도록 고쳤다.

## 12-6. 남은 일

1. GDN SSM 커널의 store→load 가시성 확보 (또는 다른 방식으로 부모 상태 전달)
2. 사슬 환원 검증 → 분기 검증
3. **융합 CUDA GDN 커널 대응** — vLLM 재빌드 필요
4. 훅을 V2 러너로 이식
5. `CommonAttentionMetadata` 로 매핑 정식화


## 12-7. 🔴 결론 — 파이썬 커널 대체로는 비트 단위 일치가 불가능하다

`tree7chain`(분기 없는 트리 = 사슬)이 끝내 base 와 일치하지 않았다.
개별 버그를 다섯 개 잡은 뒤에도 남았고, 원인은 버그가 아니라 **접근의 한계**다.

`t12_conv_equiv.py` 로 내 트리 conv 와 vLLM 의 `causal_conv1d_update` 를
직접 대조했을 때 **상대오차 0.22%** 였다. 나는 이걸 "bf16 잡음"으로 넘겼지만,
**argmax 를 뒤집기에 충분하다.** GDN 레이어가 48개이고 매 스텝 누적되면
토큰이 갈린다.

계산 순서가 다르면 정밀도가 다르고, 정밀도가 다르면 스펙 디코딩의 수용 판정이
달라진다. 파이썬/Triton 으로 융합 CUDA 커널을 '대체'하는 방식으로는
비트 단위 일치에 도달할 수 없다.

### 그래서 올바른 경로

1. **커널 자체를 트리 인식으로 고친다** — 대체가 아니라 수정.
   `fused_gdn_decode_post_conv_mtp` (CUDA) 와 `causal_conv1d_update` (Triton)
   양쪽에 부모/조상 인덱스를 받는 경로를 추가. **vLLM 재빌드 필요.**
2. 또는 **순수 어텐션 타깃에서만 DDTree 를 지원한다.** 논문의 평가 모델
   (Qwen3-Coder-30B-A3B 등)이 그렇다. 하이브리드는 범위 밖.

### 단위 테스트의 결함 (재발 방지)

t10/t11/t12 는 전부 **한 스텝**만 본다. 상태를 다음 스텝으로 넘기는 부분은
아무도 검증하지 않았고, 오늘 실패의 다수가 거기서 나왔다
(conv_state 갱신 누락, SSM 초기 인덱스, Mamba precopy).
**커널을 대체·수정할 때는 반드시 (a) 원본과의 동등성, (b) 여러 스텝에 걸친
상태 이월을 함께 검증해야 한다.** 그리고 동등성 기준은 "bf16 잡음 수준"이
아니라 **비트 단위**여야 한다 — 스펙 디코딩에서는 미세 오차가 수용 판정을 바꾼다.

## 12-8. 오늘 확정된 것 / 아닌 것

| | |
|---|---|
| 🟢 flat 네이티브 DFlash2 는 무손실, base 대비 +71% / +406% | 파이프라인 정상 |
| 🟢 트리 어텐션 마스크·KV 컴팩션·러너 훅은 실모델에서 정상 | `tree7chain`(CUDA 커널 조건) 무손실로 확인 |
| 🟢 분리 패치 동작 (`drafter_k=7` + 예산 16/32 별도) | 실측 |
| 🟢 SSM 트리 커널 정확 (실모델 크기, 오차 0) | t10 |
| 🔴 GDN 재귀 계층 트리 대응 | 커널 수정 필요, 재빌드 없이는 불가 |
| 🔴 융합 CUDA GDN 커널 | 파이썬으로 접근 불가 |
| 🔴 V2 러너 이식 | 미착수 |


---

# 13. M6 — 커널 '수정' 접근 (진행 중, 2026-08-27)

§12-7 의 결론(파이썬 대체로는 비트 단위 일치 불가)에 따라 **커널 자체를 수정**하는
경로로 전환. `causal_conv1d_update` 는 Triton 이라 **재빌드 없이 가능**하고,
융합 CUDA 경로는 `VLLM_GDN_DECODE_KERNEL=triton` 으로 우회한다.

## 13-1. 접근

곱셈-누적 순서는 원본 그대로 두고, 윈도 값의 **출처만** 조상으로 바꾼다.
사슬(parent = i-1)을 주면 같은 비트가 들어가므로 **비트 단위로 동일해야 한다.**
이것이 대체 방식과의 결정적 차이다.

패치 지점: `_causal_conv1d_update_kernel` 에 `tree_cols_ptr` + `IS_TREE` 추가,
토큰 루프 진입부에서 `col0..col{W-2}` 를 조상에서 재적재.

## 13-2. 현재 상태

| 시도 | 결과 |
|---|---|
| 레지스터 선택 (`h0/h1/h2` 스냅샷 + `tl.where`) | T=1 🟢 **비트 단위 동일**, T≥2 🔴 (차 0.586) |
| 타깃 블록에서 절대열 적재 | T=2 🔴 (차 0.401) |

**T=1 이 비트 단위로 통과한 것이 중요하다** — 배선(인자 전달, `IS_TREE` 분기,
인덱스 계산)은 맞다. 남은 건 토큰 1 이후의 윈도 출처다.

### 🔴 두 번째 시도가 틀린 이유 (확인됨)

`new_conv_state` 는 소스를 **한 칸 시프트**해 저장한다
(`conv_state_ptrs_source` 가 `idx_tokens + 1`). 따라서 타깃 블록에는
STEP1 의 가장 오래된 히스토리 `h0` 가 **이미 밀려나 없다.**
절대열 공식 하나로는 히스토리와 노드값을 모두 읽을 수 없다.

**올바른 형태**: 출처를 둘로 나눈다.
- 절대열 c < W-1 (히스토리) → **소스 블록**, 열 `conv_state_token_offset + c` (STEP1 과 동일)
- 절대열 c >= W-1 (노드 i)  → **x** 의 토큰 `c - (W-1)`

## 13-3. 남은 디버깅은 박스를 점유하지 않는다

`t13_conv_kernel_tree.py` 는 DIM=512, T≤8 의 **마이크로 테스트**로 20초면 끝난다.
27B 를 띄울 필요가 없다. 이 단계의 반복은 운영과 병행 가능하다.

## 13-4. 남은 일

1. conv 커널 윈도 출처를 소스/x 두 갈래로 분리 → T≥2 비트 단위 일치
2. 분기 트리 검증 (경로별 독립 실행 대조)
3. **여러 스텝 상태 이월** 검증 — §12-7 에서 이게 빠져 실패했다
4. 27B 엔드투엔드 (여기서만 박스 필요)
5. 융합 CUDA GDN 커널 (재빌드) / V2 러너 이식


## 13-5. 커널 '수정' 접근 — 사슬 비트 단위 일치 달성 🟢

`causal_conv1d_update` (Triton) 를 **대체가 아니라 수정**한 결과:

```
T=1 / T=2 / T=8   A 사슬 == 원본   🟢 비트 단위 동일 (최대차 0.000000)
```

§12-7 의 벽(파이썬 대체 → 0.22% 오차 → argmax 뒤집힘)을 넘었다.
**접근이 옳았음이 실증됐다.**

### 성공한 형태 (중요)

세 번의 실패를 거쳐 확정된 제약:

1. **히스토리는 레지스터 스냅샷에서** 가져와야 한다.
   STEP 2 가 새 상태를 소스와 **같은 블록에 덮어쓸 수 있어**, 루프 안에서
   `conv_state` 를 다시 읽으면 이미 밀려난 값을 읽는다.
   (진단: STEP1 과 동일한 상수 오프셋으로 재적재해도 T=1 부터 원본과 달라짐)
2. **루프 반송 레지스터 `col0..col2` 에 대입하면 안 된다.**
   루프 끝의 시프트와 충돌해 토큰 1 이후가 깨진다 (토큰 0 만 정상).
   → 트리일 때는 `acc` 를 직접 누적하고, 원본 블록은 `if not IS_TREE:` 로 감싼다.
3. 곱셈-누적 순서(w0, w1, ...)를 원본과 동일하게 유지해야 비트가 보존된다.

## 13-6. 🔴 미해결 — 분기 트리

부모가 `i-1` 이 아닌 노드(인덱스 ≥ 2)가 전부 틀린다.

```
평평한 트리 parents=[-1,0,0,0], 절대열 [[0,1,2],[1,2,3],[1,2,3],[1,2,3]]
  노드 0 🟢   노드 1 🟢   노드 2 🔴 0.345   노드 3 🔴 0.290
```

**노드 1·2·3 의 `tc` 가 `[1,2,3]` 으로 완전히 동일한데 1만 통과한다.**
차이는 마지막 슬롯 `x[idx_token]` 뿐이고, 그건 사슬 테스트에서 비트 단위로 검증됐다.

배제한 것:
- `tl.where` 스칼라 조건 → 산술 선택(`(tc==k).to(float32)` 가중합)으로 바꿔도 **오차 동일**
- 히스토리 레지스터 선택 → t14 의 FILL=0/1/2 전부 통과
- x 적재 자체 → t14 의 T=1 FILL=3 통과

남은 모순: 커널이 읽는 `tc` 를 출력에 실어 덤프하면 **전부 0** 으로 나온다.
그런데 사슬이 비트 단위로 통과하므로 `tc` 는 올바르게 읽히고 있어야 한다.
→ **덤프 코드 자체가 신뢰할 수 없다.** 다음에는 덤프를 별도 출력 버퍼에 쓰는
   방식으로 다시 확인할 것 (acc 를 재사용하지 말 것).

## 13-7. 재현 방법

```
# 20초, 27B 불필요
docker run --rm --gpus '"device=0"' -v $PWD/ddtree-dev:/work -w /work \
  -v $PWD/ddtree-dev/patched/causal_conv1d.py:/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/mamba/ops/causal_conv1d.py:ro \
  --entrypoint python3 vllm/vllm-openai:nightly t13_conv_kernel_tree.py   # 사슬/분기
  # t15_flat_tree.py : 평평한 트리를 torch 기대값과 직접 대조 (노드별 오차)
  # t14_probe_cols.py: 전열을 한 값으로 고정해 소스별 분리 검증 (T14_FILL)
```


---

# 14. 🟢 DDTree 동작 — 하이브리드 27B 에서 무손실 (2026-08-28)

§12-7 의 "파이썬 대체로는 비트 단위 일치 불가" 는 맞았다. 그러나 거기서
"이 모델에서는 재빌드 없이 불가능" 으로 확대 해석한 것은 **틀렸다.**
`causal_conv1d` 도 `fused_sigmoid_gating` 도 Triton 이라 JIT 이고,
융합 CUDA 경로는 `VLLM_GDN_DECODE_KERNEL=triton` 으로 우회하면 된다.
**커널을 수정하는 경로(사용자 선택 1번)로 무손실을 달성했다.**

## 14-1. 결과

| 설정 | 무손실 | tok/s (산문/코드) | 컴팩션 실적 |
|---|---|---|---|
| base (스펙 없음) | 기준 | 10.3 / 12.1 | — |
| flat — 네이티브 DFlash2 k=7 | 🟢 | **15.1 / 52.4** | — |
| tree7 (예산 7) | 🟢 | 14.0 / 42.5 | ssm 672, conv 912 |
| tree16 | 🟢 | 14.0 / 43.4 | KV행 150, ssm 816, conv 1440 |
| tree32 | 🟢 | 13.0 / 40.4 | KV행 230, ssm 1296, conv 2208 |
| flat7 (깊이1 분기) | 🟢 | 9.7 / 14.4 | KV행 5, ssm 288, conv 288 |

전부 `enforce_eager=True`, TP=1, `VLLM_GDN_DECODE_KERNEL=triton`, `VLLM_USE_V2_MODEL_RUNNER=0`.
컴팩션이 실제로 대량 이동하는 조건에서 출력이 base 와 토큰 단위로 일치한다.

## 14-2. 🔴 아직 flat 보다 느리다

DDTree(14.0/43.4) < flat(15.1/52.4). 산문 −7%, 코드 −17%.
**AEON-7 의 "DDTree mode is not faster yet" 과 같은 상태다.**

원인 후보 (성능 작업 미착수):
1. 타깃 forward 폭이 17 vs 8 — 트리가 넓다 (§5 의 배치1 편향과 같은 구조)
2. **레이어별 파이썬 루프** — `register_state` 48회/스텝,
   `compact_gdn` 이 스텝당 ssm 17회 + conv 30회의 개별 `.copy_()`
3. 트리 빌드가 요청마다 CPU 힙 + topk D2H 동기화
4. `tree_cols` 를 스텝마다 numpy→GPU 로 새로 만든다

## 14-3. 커널 수정에서 확정된 제약 (재현에 필수)

1. **히스토리는 레지스터 스냅샷에서.** STEP 2 가 새 상태를 소스와 같은 블록에
   덮어쓸 수 있어, 루프 안에서 conv_state 를 다시 읽으면 밀려난 값을 읽는다.
2. **노드 값은 conv_state 타깃 블록에서.** `causal_conv1d_update` 는
   `out is None` 이면 `out = x` 로 제자리 출력한다(:1237). 조상 토큰의 x 는
   이미 출력으로 덮여 있다. (증상: x 인덱스 < idx_token 이면 전부 오답)
3. **루프 반송 레지스터 `col0..col2` 에 대입 금지.** 루프 끝 시프트와 충돌해
   토큰 1 이후가 깨진다. 트리일 때는 `acc` 를 직접 누적하고 원본 블록을
   `if not IS_TREE:` 로 감싼다.
4. **곱셈-누적 순서를 원본과 동일하게.** 그래야 사슬이 비트 단위로 보존된다.
5. **컴팩션은 KV 캐시 그룹별로.** 하이브리드는 어텐션/Mamba 그룹이 따로이고,
   27B 에서 `block_table[0]` 은 **Mamba**(block_size=1024)다. 그것만 쓰면
   어텐션 캐시가 필터에 전부 걸러져 컴팩션이 아예 실행되지 않는다.

## 14-4. 오늘 잡은 원인 전체 — 전부 "사슬에서는 무해"

| # | 원인 | 왜 안 드러났나 |
|---|---|---|
| 1 | 훅이 V1 러너에만, DFlash2 는 **V2** 경로 | ngram 이 V2 미지원 → V1 폴백 |
| 2 | 트리 예산이 `drafter_k` 에 묶임 (≤7) | — |
| 3 | GDN 이 **융합 CUDA 커널** 사용 | Triton 패치가 죽은 코드 |
| 4 | 패치를 **prefill 커널**에 잘못 적용 (`s.index()` 첫 일치) | update 커널이 원본이라 당연히 일치 |
| 5 | `out = x` **제자리 출력** | 자기 토큰만 읽으면 무해 |
| 6 | 루프 반송 레지스터 충돌 | — |
| 7 | STEP 2 가 소스 블록을 덮어씀 | — |
| 8 | 컴팩션이 **Mamba 그룹** slot_mapping 사용 | 컴팩션이 항등 |
| 9 | 비동기 스케줄링의 `num_computed` 낙관값 | 동기 스케줄링(ngram)엔 없음 |
| 10 | 드래프트 반환 계약이 경로마다 다름 (리스트 vs 텐서) | — |

## 14-5. 방법론 — 효과가 있었던 것 / 없었던 것

**효과 있었음**
- `no_accept` 이분법: 마스크·RoPE·`tree_cols`·SSM 부모가 전부 정확함을 한 번에 확정
- 폴백 경로 카운터: `gdn_calls: 0` 이 세 라운드의 헛수고를 즉시 끝냄
- 원본 커널과의 **비트 단위** 동등성 테스트 (t13/t16)
- 실패한 실행이 낡은 JSON 으로 위장하지 못하게 실행 전 삭제

**효과 없었음 (반복하지 말 것)**
- 계측 없이 가설로 고치기 — 오늘 실패의 대부분
- 단위 테스트를 **한 스텝**만 돌리기 — 상태 이월 버그를 전부 놓침
- 동등성 기준을 "bf16 잡음 수준"으로 잡기 — 스펙 디코딩에선 비트 단위여야 함
- 내 구현을 **내가 쓴 레퍼런스**와 비교 (t11) — vLLM 과 규약이 어긋나도 통과

## 14-6. 산출물

```
ddtree-dev/
  ddtree_tree.py, ddtree_compact.py, ddtree_gdn.py, ddtree_runtime.py
  patched/  8개 파일 (causal_conv1d.py, fused_sigmoid_gating.py 포함)
  t1..t16   단위·통합 테스트 (t13/t16 이 비트 단위 동등성)
```

## 14-7. 남은 일

1. **성능** — §14-2 의 네 가지. 특히 레이어별 파이썬 루프 제거
2. **동적 트리 예산** — §5 의 배치1 편향 때문에 필수
3. **융합 CUDA GDN 커널** 대응 (재빌드) — 운영은 이걸 쓴다
4. **V2 러너 이식** — 업스트림 기본 경로
5. `CommonAttentionMetadata` 로 매핑 정식화
6. T>0 무손실 트리 rejection sampling (논문에도 없음)

---

# 15. 비양자화 모델(Qwen3.5-4B bf16) 실험 — 트리가 사슬에 진다

양자화가 스펙 디코딩 이득을 깎는다는 가설을 확인하려다, 비양자화 모델에서
DDTree 를 돌릴 수 있게 되면서 그동안 27B INT4 에서 못 갈랐던 것들이 갈렸다.

## 15-1. 어떻게 열었나 — SWA→full 등가 변환

`z-lab/Qwen3.5-4B-DFlash` 드래프터는 6층 중 5층이 sliding(window 4096) +
1층 full 이다. 혼합이면 KV 캐시 그룹이 2개 필요하고, 그건 V2 러너 전용이라
V1 훅으로 만든 DDTree 가 들어갈 수 없다
(`qwen3_dflash.py:147` NotImplementedError).

우리 벤치는 `max_model_len=1024` 라 시퀀스가 절대 4096 을 넘지 않는다.
**윈도우보다 짧은 시퀀스에서 sliding attention 은 full attention 과 동일하다.**
그래서 `layer_types` 를 전부 `full_attention` 으로 바꾼 사본을 만들었다:

    /home/user/vllm/hf-cache/huggingface/drafter-4b-full/
      config.json          — layer_types 전부 full, sliding_window 제거
      model.safetensors    — 원본 블롭으로의 상대 심볼릭 링크 (용량 0)

🔴 상대 링크여야 한다. 절대 링크는 컨테이너(`/hf`) 안에서 깨지고,
   블롭이 root 소유라 하드링크도 안 된다.
🔴 시퀀스가 4096 을 넘으면 등가가 깨진다. config 에 주석으로 박아뒀다.

타깃 `Qwen/Qwen3.5-4B` 는 linear_attention 24 + full_attention 8 하이브리드로
27B 와 같은 계열이고, GDN 차원도 kDimK/kDimV=128 로 동일하다. 다른 건
value head 수뿐이고(48 vs 32 → vhpk 3 vs 2) 둘 다 커널에 인스턴스화돼 있다.

## 15-2. 결과 — 트리는 모든 폭에서 같은 폭 사슬에 진다

drafter horizon 15 (`dflash_config.block_size=16`). 27B 는 7 이었다.

    설정            산문     코드   수용/스텝  상태복제  최초불일치
    base           20.0    26.0
    flat 원본 폭16   23.9   106.7      -        -      114
    사슬 폭16       23.8    99.7    3.91        0      114
    사슬 폭32       25.1    91.5    4.14      216      일치
    사슬 폭64       25.0    88.0    4.27      216      일치
    트리 폭16       24.4    56.5    2.83      840      114
    트리 폭32       24.4    72.9    3.29      720      일치
    트리 폭64       23.1    58.1    2.90     1080      57 / 78

**(1) 우리 구현 오버헤드는 없다.** 같은 폭 사슬(`VLLM_DDTREE_TOPK=1`)이 99.7 로
DFlash 사슬(101.8~106.7) 대비 2~6% 이내다. §14-2 에서 의심하던 "구현 비용"은
실체가 없었다 — 그때 수치는 폭이 다른 것끼리 비교한 것이었다.

**(2) 지는 건 분기 그 자체다.** 같은 예산에서 트리는 스텝당 2.83 토큰,
사슬은 3.91 을 받는다. 노드 N개 트리의 깊이는 N보다 훨씬 작으므로, top-1 이
거의 맞는 텍스트(코드)에서는 예산을 분기에 쓰는 게 순손해다. 게다가 사슬은
상태 복제가 0인데 트리는 ssm 840 + conv 1200 건을 낸다. 산문에서는 거의
동률(24.4 vs 23.8)인데, 이건 "불확실한 구간에서만 값을 한다"는 논문 전제와 맞다.

**(3) 폭을 넓히면 수용은 오르고 처리량은 떨어진다.** 사슬 3.91→4.14→4.27 인데
99.7→91.5→88.0. 27B 에서 "폭을 넓혀도 수용이 안 오른다"고 본 건 horizon 7 이
천장이었기 때문이고, horizon 15 에서는 수용이 오르지만 forward 비용이 더 빨리 큰다.

## 15-3. 무손실 판정 기준이 틀렸다

DFlash 사슬이 base greedy 와 토큰 114 에서 갈린다. 우리 폭16 도 **같은 114**.
다중토큰 forward 의 리덕션 순서가 달라 argmax 가 뒤집히는 것이고,
이 모델에서 "base greedy 와 비트 일치"는 성립하지 않는 기준이다.
🔴 앞으로 무손실 판정은 **DFlash 사슬을 기준선**으로 삼아야 한다 (base 가 아니라).

## 15-4. 고친 버그 — 예산 미달 트리가 원본 커널을 죽인다

`propose_from_drafter_logits` 는 트리가 예산을 못 채우면 등록을 포기했다.
그러면 GDN 계층이 `gdn_info() → None` 을 받고 **원본 커널로 물러서는데,
원본은 8토큰 상한이 있다**:

    RuntimeError: fused_gdn_decode_post_conv_mtp ...
    state_indices must have shape [N, S] with 1 <= S <= 8

topk_cap=1 사슬은 깊이당 한 노드뿐이라 최대 horizon(15) 개다. 예산 31 이면
**항상** 미달이라 매 스텝 죽는다. 게다가 배치에 트리 요청과 비트리 요청이
섞이면 어텐션은 트리 마스크, GDN 은 사슬이 되어 출력이 깨지는 정합성 구멍도 있었다.

수정: `pad_tree_to_budget` — 루트 자식으로 깊이0 분포의 미사용 상위 토큰을 붙여
노드 수를 예산에 정확히 맞춘다. 이 노드는 타깃이 그 토큰을 뽑으면 수용되는데
그게 곧 greedy 정답이므로 무손실이다. `child_maps` 가 토큰을 키로 쓰므로
**중복 토큰은 절대 넣지 않는다**.

검증: `t17_pad.py` (예산 5/15/31/63 × topk 1/2/64 에서 인덱스 정합·형제 토큰
중복 없음·가시성·추적). 예산이 이미 차면 패딩은 무연산이라 t4 참조 동일성 12/12 유지.
수정 후 죽던 사슬 폭32/폭64 가 돌고 **둘 다 완전 무손실**이다.

## 15-5. 🔴 정정 — "커널을 T=33 까지 검증했다"는 틀렸다

`cuda/t_tile.py` 는 검증이 성립하지 않는 테스트였다. 커널 규약은

    source_slot = state_indices[accepted-1],  슬롯 0 은 무효(null)
    → source_slot <= 0 이면 출력을 0으로 채우고 반환 (ddtree_gdn_tree.cu:115)

인데 테스트가 초기 상태를 **슬롯 0** 에 넣고 `accepted=0` 을 넘겼다. 두 번째
호출의 source 인덱스(`accepted-1`)도 의도한 슬롯을 가리키지 않았다.
규약을 맞춰 다시 돌리면 T=16 부터 최대차 11.25 로 **실패**한다.

`cuda/t18_branch.py` 로 대체했다 (t_tile.py 는 삭제). 올바른 규약
(초기 상태를 `row[0]`, `accepted=1`)으로:

    (A) 타일 경계  T = 9/16/17/24/25/32/33/48/64/65/96   전부 오차 0
    (B) 분기       T = 16/32/33/64/65/96, 최대 깊이 11    전부 오차 0

(B)는 참조 구현이 필요 없다 — 노드 결과는 조상 경로에만 의존하므로
**트리 전체 실행의 노드 t 출력 == 그 노드의 조상 경로만 사슬로 돌린 마지막 출력**
이어야 한다. 커널만으로 성립하는 검사다.

`t13_conv_kernel_tree.py` 도 부모 배열이 8개로 하드코딩돼 있어 넓은 폭을 못 봤다.
임의 트리로 일반화해 T=8/16/32/64/65 전부 오차 0 확인.

**결론: CUDA 커널(SSM·conv 모두)은 결백하다.** 폭64 이탈은 커널 밖이다.

## 15-6. 🔴 정정 — bf16 flat 가속비를 폭 8 로 쟀다

§15 이전에 보고한 bf16 flat 가속비(2.21×/3.16×)는 `num_speculative_tokens=7`,
즉 **폭 8** 이었다. 이 드래프터 horizon 은 15 다. 제 폭(16)으로는 코드 4.10×.
`t9_dflash2_e2e.py` 의 `FLAT_K` 를 모델별 체크포인트 값으로 분리했다.

이 때문에 양자화 비교에서 bf16 쪽이 불리하게 측정되고 있었다.

## 15-7. 양자화 가설은 이 박스에서 못 가른다

    Qwen3.5-4B  bf16      base 20.6/27.0  flat 45.6/86.2   (폭8, 과소측정)
    Qwen3.8-27B AWQ-INT4  base 10.4/12.6  flat 29.3/57.0   2.81x / 4.53x

INT4 쪽 가속비가 더 컸지만 **모델 크기가 교란 변수**다. 같은 모델 두 정밀도로
재려 했으나 전부 막혔다:

  - fp8 로 온더플라이 양자화 → A6000(sm_86) 미지원
    (`cutlass_scaled_mm_sm80_epilogue`, fp8 텐서코어는 Hopper 부터)
  - 27B bf16 원본 → 55.6GB > 여유 디스크 39GB
  - 유일한 큰 항목이 운영이 쓰는 AWQ 가중치 20GB 라 지울 수 없음

말할 수 있는 건 하나: 운영 모델(27B AWQ-INT4)에서 스펙 디코딩은 2.8×/4.5× 를
내고, 양자화가 이득을 죽인다는 징후는 이 박스에서 안 보인다.

## 15-8. 남은 것

  - 27B 예산32 실패는 별개다 — 그 실행은 `gdn_T_mismatch: 0` 이라 §15-4 의
    패딩 버그가 아니다.

## 15-9. 이탈 판정법 — 로짓 격차로 '수치'와 '버그'를 가른다

`t19_tiegap.py` 가 base 의 위치별 top-1/top-2 로짓 격차를 전부 저장한다
(`out_tiegap.json`). 어떤 구성이 base 와 갈린 위치의 격차를 조회해 분류한다.

    전체 216개 위치: 중앙값 7.875, 최소 0.000, 하위5% 0.625

    설정   프롬프트  위치   격차     판정
    flat      0    114   0.000   동점 → 수치
    폭16       0    114   0.000   동점 → 수치
    폭40       0     57   0.250   근사동점 → 수치
    폭40       1     72   7.875   🔴 버그
    폭64       1     78   8.625   🔴 버그

🔴 "출력이 자연스러우니 수치겠지" 는 근거가 안 된다. 폭64 의 프롬프트1 은
   배열 리터럴 한 토큰(`5`→`3`) 차이라 멀쩡해 보였지만 격차 8.625 였다.
   무손실 스펙 디코딩에서 타깃 argmax 가 아닌 토큰이 나오면 정의상 위반이다.

## 15-10. 미해결 버그 — 기각한 가설 네 개

폭 40 이상의 **깊게 분기한** 트리에서만 나온다. 폭 32 이하와 척추 구성은 무손실.
재현성 있음 (폭64 를 두 번 돌려 57/78 동일).

기각한 가설:

  1. **KV 압축 경합** — 커널은 세그먼트 안에서 순차로 돌고 세그먼트끼리 슬롯이
     겹치지 않는다. 전제(`dst[i] <= src[i]`) 위반을 직접 세는 계측을 넣었고
     (`compact_unsafe`), 폭40·폭64 에서 **0건**이었다. 계측과 안전 대체 경로
     (`compact_kv_torch`, gather 후 scatter)는 남겨뒀다.
  2. **슬롯 0 가드** — 커널이 슬롯 0 을 무효로 본다. 그러나 vLLM 은 블록 0 을
     null block 으로 예약한다 (`v1/core/block_pool.py:190`). 관례가 맞다.
  3. **CUDA GDN 커널** — `VLLM_DDTREE_GDN_CHECK=1` 로 같은 입력에 CUDA 와 Triton 을
     모두 돌려 비교했다. SSM 상태는 1416회 호출 **전부 상대오차 0.00000**,
     출력만 최대 0.3% (게이팅+RMSNorm 의 bf16 반올림). 격차 8.6 을 뒤집을 수 없다.
     🔴 Triton 으로 바꾸면 증상이 사라지지만 이는 그 0.3% 가 중간에 수용 판정을
        하나 바꿔 트리 모양이 달라지고 **유발 조건을 우연히 비껴간 것**이다.
        "Triton 이 고친다" 를 "CUDA 가 원인" 으로 읽으면 안 된다.
  4. **커널 수학** — `cuda/t18_branch.py` 로 임의(비연속·비단조) 슬롯,
     `accepted` 1~33, T=96, 분기 깊이 11 까지 전부 오차 0.

남은 후보: 어텐션 마스크, RoPE 위치, accept 로직.

## 15-11. 수용 길이는 양극단이다 — best-first 가 값어치를 잘라먹는다

`acc_hist`(수용 길이) 와 `depth_hist`(트리 최대깊이) 를 stats 에 넣고 측정했다.

    폭16 사슬:  깊이 15 고정(48/48)
      수용길이  0:12 1:7 2:9 3:1 4:2 5:2 6:2 7:4 9:1 12:1 14:2 15:3
      → 길이>=5 가 수용토큰의 80%, >=8 이 52%          코드 102.1

    폭16 트리:  깊이 1~15, 대부분 2~7
      수용길이  0:8 1:17 2:9 3:7 4:5 5:2 6:4 7:2 8:1 10:1 11:1 15:1
      → 길이>=5 가 55%, >=8 이 26%                     코드  58.5

수용은 **즉시 실패하거나 지평 전체를 받아낸다**. 코드 생성의 성질과 맞는다 —
뻔한 구간은 통째로 통과하고 결정 지점에서 한 번에 막힌다. 그리고 값어치의
대부분이 긴 수용에 몰려 있다.

🔴 누적 log-prob 은 깊이가 늘 때마다 단조 감소하므로, best-first 는 **구조적으로
   폭을 깊이보다 선호**한다. 예산 15 를 줘도 깊이가 2~7 에 머물러 길이 8~15
   구간에 도달하지 못한다. 평균 수용(3.91 vs 2.90)만 봐서는 이 구조가 안 보인다.

## 15-12. 척추 우선 배분 — 결함은 고쳤으나 분기는 값을 못 한다

`build_tree(..., spine=True)` / `VLLM_DDTREE_SPINE=1`. 지평까지 사슬을 먼저 깔고
각 척추 노드의 형제(rank 1)를 힙에 심어, 남는 예산만 그 위에서 분기시킨다.
참조 구현에서 벗어나므로 **기본값은 off** (t4 참조 동일성 12/12 유지).

    설정          산문     코드   수용/스텝  깊이중앙  >=5비중
    base         20.0    26.0
    flat 원본 폭16 23.8   101.8      -        -       -
    사슬 폭16      24.7   102.1    3.91       15     80%
    트리 폭16      24.6    58.5    2.90        4     55%
    척추 폭16      24.7   104.2    3.91       15     80%
    척추 폭24      25.3    77.8    4.02       15     65%
    척추 폭32      24.6    77.4    3.54       15     70%
    척추 폭64      25.5    82.5    4.50       15     68%

깊이 중앙값이 4 → 15 로 회복됐다. 척추 폭16(104.2)은 측정 최고치이고 ngram 사슬
flat(101.8)을 넘는다 — 다만 예산 15 = 지평이라 이 구성은 사실상 사슬이다.

**분기의 경제성이 성립하지 않는다.** 척추 위에 분기를 얹으면 수용은
3.91 → 4.50 (+15%) 오르는데 처리량은 104.2 → 82.5 (-21%) 떨어진다.
역산하면 스텝 시간이 45% 늘었다. 토큰 48개를 더 넣는 비용(넓어진 forward +
상태 복제 + 마스크 + 압축)이 수용 증가분을 압도한다.

**단 산문은 반대다.** 산문 최고는 척추 폭64(25.5) > 사슬 폭16(24.7).
코드 최고는 폭16, 산문 최고는 폭64 — "불확실한 텍스트에서만 분기가 값을 한다"는
예측과 맞는다.

## 15-13. TP=2, CC=1 은 이 박스에서 답이 아니다

"CC=1 이면 GPU 가 놀고 있으니 폭 추가가 싸지 않겠나" 를 실측했다.

    설정        산문    코드    무손실   TP1 코드   비율
    base       15.3   22.3     기준      26.0
    flat 폭16   28.4   75.7     🔴      106.7    0.71x
    사슬 폭16    27.6   76.1     🟢       99.7    0.76x
    트리 폭16    28.2   52.7     🔴       56.5    0.93x
    사슬 폭32    30.1   67.4     🟢       91.5    0.74x
    트리 폭32    28.6   48.5     🔴       72.9    0.67x

    같은 폭 트리/사슬 비율:  폭16  TP=2 0.693 vs TP=1 0.567
                            폭32  TP=2 0.720 vs TP=1 0.797

기전은 실재한다 — 폭16 에서 트리가 사슬을 따라잡는 폭이 커졌다(0.57→0.69).
그러나 폭32 는 반대로 벌어졌고, 무엇보다 **TP=2 자체가 전 구간에서 느리다**
(0.67~0.93배). 이 박스는 GPU P2P 가 안 돼 `disable_custom_all_reduce` 로
호스트 메모리를 거치는데, 배치 1 에서는 그 통신비가 연산 분할 이득보다 크다.

TP=2 실행 시 주의:
  - `VLLM_ENABLE_V1_MULTIPROCESSING=0` 를 켜면 안 된다 (워커가 별도 프로세스다)
  - spawn 된 자식이 모듈을 다시 import 하므로 실행부가 `__main__` 가드 안에
    있어야 한다. 없으면 `RuntimeError: An attempt has been made to start a new
    process before the current process has finished its bootstrapping phase`
  - 🔴 트리가 **랭크마다 따로** 만들어진다. 랭크 간 드래프터 logits 가 동일해야
    같은 트리가 나온다는 가정에 기댄다 — 미검증이다
  - `ddtree_runtime.LAST` 는 워커 프로세스에 있어 부모에서 stats 를 못 읽는다
    (TP=2 결과의 `drafter_k=None` 이 그 증상)

## 15-14. 남은 일

  - §15-10 의 미해결 버그 — 어텐션 마스크 / RoPE / accept 로 좁혀졌다
  - **동적 폭**: 코드 최고가 폭16, 산문 최고가 폭64 라는 두 극단이 나왔으므로
    드래프터 확신도로 폭을 고르는 근거가 생겼다. 남은 유일한 승산이다
  - 27B 예산32 실패 (별개 원인)
  - V2 러너 이식 — 4B DFlash 드래프터를 등가 변환 없이 쓰려면 필요하다


---

# 16. "우리가 잘못 구현한 건 아닌가" — 아니다. 원인은 드래프터 확률의 오보정이다.

## 16-1. 참조 구현과의 대조

`ddtree-dev/ref/ddtree.py` 의 `ddtree_generate` 를 읽고 우리 것과 맞췄다. 구조 동일:
드래프터 **한 번** 실행 → `[horizon, vocab]` 로짓 → 모든 분기가 그 위치별 주변확률을
공유 → 타깃 샘플로 트리 하강 → 수용 경로로 캐시 압축.

🔴 참조의 기본값은 `tree_budget = draft_horizon` 이다 (block_size-1). 즉 논문 기본
   설정은 우리 "폭16" 과 같고, flat DFlash 와 **검증 forward 토큰 수가 동일**하다.
   비용이 같은데 수용률이 높다는 게 논문의 주장인데, 우리는 그 설정에서 수용이
   떨어졌다 (2.90 vs 3.91). 비용으로는 설명이 안 되는 정면 모순이었다.

## 16-2. 구현 검증 다섯 층 — 전부 통과

| 층 | 방법 | 결과 |
|---|---|---|
| 트리 빌더 | 참조와 직접 대조 (`t4_tree_build.py`) | 12/12 동일 |
| 커널 수학 | 임의 슬롯·accepted 1~33·T=96·깊이 11 (`cuda/t18_branch.py`) | 오차 0 |
| conv 트리 | T=8~65 (`t13_conv_kernel_tree.py`) | 오차 0 |
| GDN 통합 | 실행 중 CUDA vs Triton 상태 비교 (`VLLM_DDTREE_GDN_CHECK=1`) | 1416회 오차 0 |
| **분기 문맥** | **노드 로짓 vs 경로 단독실행 (`t20_node_context.py`)** | **48/48 일치** |

`t20` 이 결정적이다. 지금까지의 모든 검증은 **최종 출력**만 봤는데, 트리 마스크나
RoPE 가 분기 노드에서 틀리면 그 노드의 드래프트가 절대 안 맞으면서도 출력은
척추만으로 정상이라 **무손실 판정을 통과한다** — 조용히 수용률만 갉아먹는다.
t20 은 처음으로 그 층을 연다. 불변식은:

    노드 i 의 타깃 샘플 == 그 노드의 조상 경로를 평범한 시퀀스로 돌린 결과의 argmax

🔴 t20 첫 실행은 "48개 중 27개 불일치" 였는데 **테스트가 틀린 것**이었다. prefill 이
   낸 첫 토큰은 accept() 훅을 안 거쳐 `emitted_before` 가 1 적었고, prefix 가 한 칸
   밀려 있었다. 루트 노드(트리가 개입하지 않는다)까지 틀린 게 단서였다.
   `gen[emitted_before + off] == sampled[0]` 로 정렬을 자체 검증하고, 못 맞추면
   숫자를 내놓지 않고 중단하도록 고쳤다. 보정 후 off=1, 불일치 0.

## 16-3. 진짜 원인 — 드래프터가 자기 정확도를 과소평가한다

`t21_calib.py`: 순수 사슬로 돌려 깊이별 **예측 확률**과 **실제 조건부 수용률** 대조.

    깊이   예측    실제    도달   비율
     1    0.708  0.739    46    1.04
     2    0.658  0.794    34    1.21
     4    0.629  0.944    18    1.50
     7    0.452  0.846    13    1.87
     9    0.368  1.000     7    2.72
    12    0.248  1.000     6    4.04
    14    0.233  1.000     5    4.29
    평균  0.433  0.844

오차가 **깊이에 따라 커지고**(1.04 → 4.29), 조건부 수용률이 깊은 곳에서도 0.85~1.0 을
유지한다 — 위치 간 강한 양의 상관이다. best-first 의 가중치는 확률의 **곱**이라
독립을 전제하므로 이 상관을 전혀 못 본다. 오차가 곱으로 누적된다:

    깊이 10 도달 확률   예측(독립 곱) 0.0012   실제(조건부 누적) 0.130   → 108배

그래서 "깊이 10 노드(0.0012) vs 깊이 1 형제(~0.1) → 형제가 80배 낫다" 고 오판하고
예산이 전부 얕은 형제로 샌다. 트리 깊이가 4~5 에 머무는 이유, 수용이 사슬보다 낮은
이유, 척추 배분이 들은 이유, 논문 설정에서 결과가 반대인 이유가 한 번에 설명된다.

⚠️ 깊은 구간 표본이 작다 (깊이 12~15 는 5~6회). 개별 1.000 은 노이즈지만 단조 증가
   추세는 견고하다.

## 16-4. 깊이 보정 — 방향은 맞지만 결론을 못 바꾼다

`build_tree(..., depth_bonus=β)` / `VLLM_DDTREE_BETA`. 가중치를 `Σ(log pᵢ + β)` 로
둔다. 깊이 d 노드가 얕은 형제 대비 `β(d-1)` 이득을 본다.

🔴 log-prob 에 상수를 **곱하는** 건 무연산이다. 단조 변환이라 heap 순서가 그대로다
   (`α·S_d > α·S_1 ⟺ S_d > S_1`). 처음에 alpha 로 시도했다가 α=1.0~0.15 에서
   수용 3.29, 깊이 5 가 **완전히 동일**하게 나와서 알았다. 곱이 아니라 덧셈이어야 한다.

    설정              산문    코드   수용/스텝  깊이중앙
    사슬 폭16         24.7   102.1    3.91      15
    β=0    폭32(참조) 24.3    72.9    3.29       5
    β=0.35 폭32      23.1    77.0    3.25       6
    β=0.7  폭32      25.0    75.7    3.56       7
    β=1.0  폭32      24.4    69.2    3.57       8
    β=0.7  폭64      23.1    63.5    3.02       6

깊이 5→8, 수용 3.29→3.57 로 개선되지만 사슬을 못 넘는다. β→∞ 는 척추 모드로
수렴한다 (β=0.7 폭32 의 75.7/3.56 ≈ 척추 폭32 의 77.4/3.54).

⚠️ 합성 로짓으로 β 범위를 잡을 때 rank-1 이하를 거의 균등하게 깔아 형제 확률을
   과소하게 잡았다. 그래서 β=0.7 에서 깊이 15 를 예측했지만 실제는 7 이다.
   실제 드래프터는 2순위에도 상당한 질량이 있다.

## 16-5. 왜 분기가 이길 수 없나 — 비용 분해

    구성        수용   tok/s  스텝시간
    사슬 폭16   3.91   102.1   38.3ms
    사슬 폭64   4.27    88.6   48.2ms
    척추 폭64   4.50    82.5   54.5ms

16→64 토큰에서 **+26% 는 넓어진 forward 자체**, **+13% 만 분기 부대비용**
(상태 복제·마스크·압축). 우리 오버헤드를 0 으로 만들어도 forward 만으로 +26% 인데
수용은 최대 +15% 다. **병목은 구현이 아니라 산술이다.**

## 16-6. 결론

구현은 참조와 동등하다. 이 조합에서 DDTree 가 안 되는 이유는 둘이다:

  1. **드래프터가 논문이 상정한 것보다 정확하다.** DFlash2 는 지평 15 를 통째로
     맞히는 일이 흔하다 (48스텝 중 5회). 예산을 분기보다 깊이에 쓰는 게 항상 낫다.
  2. **넓힌 forward 비용이 수용 증가분을 넘는다.** 보정으로 트리가 옳은 모양으로
     자라게 해도 이 산술은 안 바뀐다.

산문에서는 일관되게 트리가 근소 우위다 (척추 폭64 25.5 vs 사슬 폭16 24.7).
**동적 폭** — 드래프터 확신도로 사슬/분기를 고르는 것 — 이 유일하게 남은 승산이고,
`t21_calib` 의 깊이별 조건부 수용률이 그 판단 기준을 준다.


---

# 17. 순수 어텐션 + 불확실 드래프터 — 여기서는 DDTree 가 이긴다

> 🔴 **이 결론은 재현되지 않는다. §24 를 먼저 읽어라.**
> 여기 근거는 "파리 프롬프트에서 ngram 사슬이 42.3 tok/s 로 base 보다 느리다"
> 인데, 지금 다시 재면 ngram 사슬이 **177.8 tok/s** 다. 비교 대상이 그때보다
> 훨씬 강해졌고, DDTree 는 eager/cudagraph 어느 쪽에서도 ngram 사슬에 진다.


## 17-1. 구성

`Qwen/Qwen3-0.6B` (`Qwen3ForCausalLM`, 28층, `layer_types: None` → GDN·SWA 없는
순수 풀 어텐션) + **ngram 드래프터** (`t7_ddtree_e2e.py`).

ngram 은 과거 일치가 여럿일 때 진짜 다중 후보 분포를 만든다 — DDTree 가 상정한
"불확실한 드래프터" 그 자체다. DFlash2 는 반대로 너무 정확했다 (§16).

## 17-2. 결과 — ngram 사슬 대비 22% 빠르다

    설정              파리    소수     반복    코드   합계초   무손실
    base             55.8   55.0    57.5   55.8    6.86
    ngram 사슬 폭16   42.3   68.5   641.3   71.4    5.17    🟢
    ngram 사슬 폭32   42.3   69.8   644.8   71.5    5.13    🟢
    DDTree 트리 폭16 117.8   53.4   391.2   64.2    4.35    🟢
    DDTree 트리 폭32 117.1   56.4   414.2   66.5    4.20    🟢
    DDTree 트리 폭64 146.0   57.6   406.4   67.0    3.99    🔴
    동적 τ=1.5 폭32  116.6   56.9   419.0   66.4    4.19    🟢

**합계 5.13 → 4.20초 (22% 단축), base 대비 1.63배.** 무손실이다.

프롬프트별로는 갈린다:
  - 반복 텍스트: ngram 사슬이 크게 이긴다 (645 vs 414). 단일 사슬이 거의 항상 맞으니
    깊이가 전부다.
  - 비반복 텍스트("파리"): **ngram 사슬이 base 보다 느리다** (42.3 < 55.8).
    드래프트가 계속 틀려 forward 만 낭비한다. 트리는 여러 후보를 동시에 걸어
    이걸 2.1배 이득으로 뒤집는다 (117.8).

합계를 뒤집는 건 이 마지막 항목이다. **논문의 주장이 재현된다.**

## 17-3. 법칙 — 드래프터 확신도가 전부다

    환경                                   드래프터    이기는 쪽
    27B/4B 하이브리드 + DFlash2             항상 확신    사슬
    0.6B 순수 어텐션 + ngram, 반복 텍스트     확신        사슬 (641 vs 406)
    0.6B 순수 어텐션 + ngram, 비반복 텍스트   불확실      트리 (146 vs 42)

§16 의 부정적 결과는 구현 결함이 아니라 **DFlash2 가 너무 정확해서**였다는 것이
양방향으로 확인된다.

## 17-4. 동적 선택 — 작동하지만 고를 모양이 부족하다

τ 를 1.5/3.0/6.0 어떻게 잡아도 4.19~4.25초로 고정 트리(4.20)와 같다.

선택 규칙 자체는 유효하다. 기대 사슬 길이 `E = Σ_d ∏_{i<=d} p_i` 와 실제 수용의
피어슨 상관이 **0.748** 이고, 사분위별 실제 수용이 0.64 → 2.25 → 3.36 → 9.08 로
단조 증가한다 (§16-3 의 데이터). 모드별 분할도 깨끗하다 (사슬쪽 5.3~7.8,
트리쪽 1.6~2.0).

🔴 문제는 **고를 수 있는 모양의 집합**이다. 우리 "사슬" 선택은 척추 모드인데,
   예산이 지평보다 크면 남는 예산이 분기로 가서 결국 트리와 비슷해진다. 반복
   프롬프트에서 ngram 사슬의 641 을 회복 못 하고 405~419 에 머무는 게 그 증거다.
   **확신 시 예산 전부를 깊이에 쏟는 진짜 사슬 모드가 없다.**

다음 단계: 확신 시 순수 사슬(topk=1)을 쓰되, ngram 처럼 후보가 부족해 예산을
못 채우는 경우 **남는 예산을 비워두는**(드래프트를 짧게 내는) 처리가 필요하다.
지금은 `pad_tree_to_budget` 이 무조건 예산을 채우고, 채우지 못하면 트리 등록을
포기해 스펙 디코딩이 아예 안 돈다.

🔴 실제로 순수 어텐션 첫 실행에서 `topk=1` 구성이 `tree_steps=0` 이었다 —
   스펙 디코딩이 아예 안 돌아 base 와 같은 수치(55.9 vs 55.8)가 나왔다.
   이걸 "사슬" 로 표에 넣고 "트리가 사슬을 이긴다" 고 읽으면 완전히 틀린다.
   대조군은 반드시 **vLLM 의 ngram 사슬** 로 잡아야 한다.

## 17-5. 미해결 버그의 범위가 좁혀졌다

폭64 의 정확성 버그가 **GDN 없는 순수 어텐션에서도 재현**된다 (`pa_t64` 🔴).
GDN 커널·conv·상태 복제를 전부 용의선상에서 제외할 수 있다.
남은 후보: **어텐션 마스크 / RoPE 위치 / accept 로직**.
0.6B 에서 재현되므로 디버깅 비용이 27B·4B 대비 훨씬 싸다.

## 17-6. 측정 방법 정정

🔴 스텝 시간을 `(수용/스텝 + 1) / tok_s` 로 계산하면 안 된다. `stats` 는 전체
   프롬프트 합산인데 `tok_s` 는 프롬프트별이라 둘을 섞으면 무의미하다.
   §15~16 에서 이 방식으로 낸 스텝 시간 표와 "+26% forward / +13% 분기" 분해,
   그 위에 세운 손익분기 계산은 **전부 무효**다.

🔴 `self.t` 의 단계별 타이머(propose/accept/kv_compact/gdn_compact)는 **CPU 측
   호출 시간만** 잰다. GPU 는 비동기라 실제 비용이 다음 동기화 지점으로 밀린다.
   이 타이머로 "사슬과 트리의 오버헤드가 같다" 고 결론내도 안 된다.

올바른 방법: `t9`/`t7` 의 `per_prompt` — 프롬프트마다 stats 스냅샷 차분과
wall-time 을 함께 기록한다. 통계 혼합도 없고 비동기 누락도 없다.


---

# 18. 폭 40 버그 추적 — 커널은 결백했고, 틀린 것은 판정 기준이었다

§15-10 의 미해결 버그를 끝까지 추적한 기록. 결론부터: **고칠 코드가 없었다.**

## 18-1. 증상을 정확히 좁히기까지

    이탈 위치 78   base=20 → tree=18   격차 8.625 (중앙값 7.5 초과 = 진짜 이탈)
      └ 스텝19 노드56, 깊이 8, 수용 경로의 마지막, 격차 8.500
          └ 24스텝 중 수용 경로 오류는 이 하나. 그 스텝만 수용이 8까지 갔다
              └ 최대깊이 8 인 스텝이 둘인데, 수용이 4에서 끊긴 쪽은 무사

🔴 "폭 40 이상" 은 증상이지 원인이 아니다. 실제 조건은 **수용 경로가 깊이 8까지
   내려가는 것**이고, 폭은 그런 깊은 노드가 트리에 들어올 확률을 높일 뿐이다.

## 18-2. fp64 참조가 답을 냈다

실제 실행에서 덤프한 입력(부모·슬롯·초기상태, **깊이 8 스텝 포함**)으로 커널
수식을 fp64 로 다시 구현했다 (`t23_fp64ref.py`, `t26_tri_vs_fp64.py`).

    CUDA   상태 vs fp64            0.000000
    Triton 상태 vs fp64            0.000000
    CUDA   출력 vs fp64            0.003560
    Triton 출력(fp32 유지) vs fp64  0.000803
    Triton→bf16 저장 vs fp64       0.003560   ← CUDA 와 동일
    fp64 를 bf16 으로 저장만        0.003560   ← 저장 자체의 하한

**출력 오차가 bf16 출력 텐서의 이론적 하한과 정확히 같다.** 커널은 더 정확해질 수
없다. 두 구현은 1 ULP 만큼 다르고, GDN 은 재정규화가 없는 재귀 계층이라 24개
계층 × 수십 스텝에 걸쳐 그 차이가 쌓여 깊은 트리에서 결정을 뒤집는다.

## 18-3. 🔴 실패한 수정 세 건 — 전부 상대 비교로 판단했다

| 시도 | 근거 | 결과 |
|---|---|---|
| shared_out 을 fp32 로 | 참조가 fp32 를 쓴다고 읽음 | CUDA-Triton 오차 0.2%→0.78% **악화** |
| 게이트를 정규화 전에 | `layernorm_guard.py` 비그룹 경로를 읽음 | 오차 **358% 폭증** |
| conv 압축 off-by-one | 단위 테스트 + 사슬 통제군 통과 | 실제 실행 **더 악화** |

공통 원인: **절대 기준 없이 두 구현을 서로 비교**하거나, **프로덕션 호출을
재현하지 못한 합성 하네스**로 판단했다.

  - 1번: Triton 을 기준으로 삼았는데 Triton 은 기준이 아니었다. fp64 로 재면
    bf16 반올림은 애초에 원인이 아니다 (RMSNorm 이 분자와 rstd 를 함께 스케일해
    균일한 상대 오차를 상쇄한다 — Triton 출력을 bf16 으로 깎아도 0.000803 그대로).
  - 2번: `layernorm_guard.py` 에는 비그룹/그룹 두 갈래가 있고 GDN 은 **그룹**이다.
    그룹 경로는 정규화가 먼저다 — 원래 커널이 맞았다.
  - 3번: 사슬 통제군이 증명한 건 **하네스의 자기일관성**뿐이다. `num_accepted_tokens`
    의 스텝별 변화 등 실제 인자를 재현하지 못했다.

## 18-4. 무엇이 실제로 원인을 좁혔나

  - **`VLLM_DDTREE_GDN_CHECK=2`** — CUDA 경로 그대로 두고 **출력·상태만 Triton
    것으로 주입**하는 A/B. 무손실이 회복되면서 conv 와 상태 등록을 한 번에 배제했다.
  - **fp64 참조** — 유일하게 신뢰할 수 있었던 도구. 실제 덤프 입력을 썼기 때문이다.
  - **깊은 스텝 선택 덤프** — 얕은 스텝만 검증하고 일반화하던 구멍을 막았다.
  - **로짓 격차 분류기** — 두 방향 모두로 오판을 잡았다 (자연스러운 출력이 격차
    8.6 이었던 적도, 🔴 가 격차 0.06 인 동점이었던 적도 있다).

## 18-5. 결론 — 기준을 모델 종류별로 나눈다

  - **순수 어텐션**: base greedy 와 비트 일치가 성립한다 (KV 가 토큰별 독립).
  - **하이브리드(GDN)**: 성립하지 않는다. DFlash 사슬을 기준선으로
    삼거나 허용오차 비교를 쓴다. DFlash 사슬조차 base 와 갈린다 (토큰 114, 격차 0.000).
  - 비트 일치가 꼭 필요하면 `VLLM_GDN_DECODE_KERNEL=triton` (무손실, 대신 느리다).

## 18-6. 테스트 설계에서 얻은 것

이 세션에서 진단 도구가 **다섯 번** 틀렸고 전부 그럴듯한 숫자를 냈다
(t_tile 의 슬롯 0, t20/t22 의 접두사 off-by-one, 로짓 인덱스 오독, t13 의 부모
하드코딩, t24 의 프로덕션 미재현). 매번 살려준 것은 **틀릴 수 없는 통제군**이었다.

규칙으로 남긴다:

  1. 검사 대상 안에 **트리가 개입하지 않는 항목**(루트 노드)을 넣는다. 그게 틀리면
     하네스가 틀린 것이다.
  2. 정렬·전제가 검증되지 않으면 **숫자를 출력하지 않고 중단**한다.
  3. 자기일관성 검사(A vs A')는 잘못된 규약을 못 잡는다. **절대 기준**을 둔다.
  4. 오프라인 하네스는 **실제 덤프 입력**을 쓴다. 합성 호출은 프로덕션을 재현하지 못한다.
  5. 이탈은 **로짓 격차를 재기 전에 판정하지 않는다** — 어느 방향으로도.


---

# 19. 계층별 CUDA 이벤트 계측 — 어텐션은 무죄, 비용은 GDN 과 파이썬에 있다 (2026-08-31)

질문: **"트리 어텐션을 넣으면 GDN 에서 이길 수도 있는 것 아닌가."**
그동안 추론만 했고 계층별 시간을 잰 적이 없어서, 직접 쟀다.

도구: `vllm/v1/spec_decode/ddtree/layerprof.py` (`VLLM_DDTREE_LAYERPROF=1`).
디코더 계층과 그 하위 모듈(`linear_attn`/`self_attn`, `mlp`)에 forward 훅을 걸어
CUDA 이벤트로 GPU 구간을 재고, 모델 전체 CPU 벽시계도 같이 남긴다.
`kernel_profile()` 은 `torch.profiler` 로 커널별 순수 GPU 시간을 따로 뽑는다.

측정: Qwen3.5-4B (하이브리드 bf16) + `/hf/drafter-4b-full`, TP=1, `enforce_eager`,
FlashInfer, 단일 요청, 산문 프롬프트 128토큰. 재현은 `ddtree-dev/run_prof.sh` +
`t28_layerprof.py`, 표는 `t28_report.py` / `t28_kern.py`.

## 19-1. 검증 forward 1회당 (CUDA 이벤트)

| 실행 | 폭 | tok/s | 스텝 GPU ms | GDN 혼합 | 전체어텐션 | MLP | 수용/스텝 |
|---|---|---|---|---|---|---|---|
| base (스펙 없음) | 1 | 23.4 | 39.84 | 14.35 | 4.40 | 8.76 | — |
| flat15 (DFlash 사슬) | 16 | **62.5** | 46.87 | 20.85 | 4.54 | 8.84 | — |
| tree16 (Triton 트리 커널) | 17 | 53.8 | 51.04 | 24.94 | 4.61 | 9.60 | 3.32 |
| **cuda16 (CUDA 트리 커널)** | 17 | **61.6** | **43.31** | **17.23** | 4.73 | 9.56 | 3.44 |
| tree32 | 33 | 49.3 | 52.47 | 26.14 | 4.62 | 8.96 | 2.95 |
| tree64 (Triton) | 65 | 45.8 | 55.60 | 31.55 | 4.85 | 9.67 | 2.93 |
| **cuda64 (CUDA)** | 65 | **55.7** | **45.90** | **26.33** | 4.69 | 8.33 | 3.00 |

폭 1 → 65 로 넓힐 때 늘어난 15.76 ms 의 내역: GDN +17.19, **전체어텐션 +0.44**,
MLP +0.92. 어텐션 비중은 11.1% → 8.7% 로 **떨어진다**.

## 19-2. 🔴 eager 훅만 믿으면 안 된다 — 런치 바운드

GPU 이벤트 시간과 CPU 벽시계가 모든 실행에서 소수 둘째 자리까지 일치한다
(예: 55.60 대 55.58). 완전한 **CPU 런치 바운드**다. 그 상태에서는 GPU 일이
늘어도 런치 공백을 넘기 전까지 Δ 가 0 으로 보인다. 실제로 훅은 MLP 증가를
+0.92 ms 로 봤지만 커널 프로파일은 GEMM 이 +3.64 ms/스텝 늘었다고 말한다.
**훅 수치는 작은 커널 경로에서 실제 GPU 일을 과소평가한다.**

그래서 `torch.profiler` 로 교차검증했다. 스텝당 순수 커널 시간:

| | 폭 1 | 폭 17 | 폭 65 |
|---|---|---|---|
| GEMM | 13.16 ms | 15.80 | 16.80 |
| GDN | 0.22 | 1.63 | **7.30** |
| 어텐션 | 0.080 | 0.163 | **0.196** |
| 합계 | 15.63 | 21.21 | 28.45 |

증가분 12.82 ms 중 GDN 55%, GEMM 28%, **어텐션 0.9%**.

커널 이름 단위 (폭 65):

    306.28 ms  864회  354.5 us  fused_sigmoid_gating_delta_rule_update_kernel
      4.92 ms  288회   17.1 us  flashinfer::BatchPrefillWithPagedKVCacheKernel

계층당 GDN 354.5 us 대 어텐션 17.1 us — **20.7배**이고 GDN 계층이 3배 많다.
어텐션 커널의 회당 시간은 폭 1/17/65 에서 **9.2 / 9.2 / 9.1 us 로 불변**이다.

## 19-3. `custom_mask` 를 직접 껐다 켠 프로브

같은 폭 65, GDN 경로 동일, FlashInfer 마스크만 차이:

| | 어텐션 ms/forward | 스텝 GPU ms |
|---|---|---|
| `VLLM_DDTREE_NOMASK=1` (causal) | 4.70 | 54.23 |
| tree64 (`custom_mask`) | 4.85 | 55.60 |

**트리 마스크의 대가는 0.15 ms — 스텝의 0.27%.** 전용 트리 어텐션 커널이
이 비용을 0 으로 만들어도 0.27% 다.

⚠ NOMASK 는 트리를 전부 버리므로(`dropped=134, masked=0`) GDN 경로가 완전히
같지는 않다 (GDN 30.94 대 31.55). 어텐션 쪽 비교만 유효하다.

## 19-4. 🟢 결론 — 트리 어텐션은 답이 아니고, CUDA 커널이 답이었다

세 계측이 같은 방향을 가리킨다. **트리 어텐션(#42121 에서 제거된 백엔드)을
되살려도 GDN 하이브리드의 결과는 안 바뀐다.** 어텐션은 폭 65 에서 순수 커널
시간의 0.7%, 폭에 따라 사실상 증가하지 않는다.

대신 **CUDA 트리 커널이 진짜 지렛대**였다. Triton 폴백 대비:

  - 폭 17: GDN 24.94 → **17.23 ms** (−31%), 53.8 → **61.6 tok/s** (+14.5%)
  - 폭 65: GDN 31.55 → **26.33 ms** (−17%), 45.8 → **55.7 tok/s** (+21.6%)

수용률은 그대로다 (3.32 → 3.44, 2.93 → 3.00) — 순수하게 커널 비용만 줄었다.

🔴 **§15 의 "트리가 사슬에 진다" 는 Triton 폴백에서 측정한 것이다.**
CUDA 커널을 쓰면 `cuda16` 의 forward 가 DFlash 사슬 `flat15` **보다 싸다**
(43.31 대 46.87 ms, GDN 17.23 대 20.85). 그런데 tok/s 는 61.6 대 62.5 로
아직 진다. 이유는 다음 항이다.

## 19-5. 다음 병목 — 모델 forward 밖의 파이썬 비용

`cuda16` 기준, 스텝당:

| 구간 | ms/스텝 |
|---|---|
| propose (트리 빌드, CPU topk + 힙) | 3.19 |
| accept | 1.83 |
| kv_compact | 1.33 |
| gdn_compact | 0.50 |
| mask (numpy → GPU) | 0.20 |
| **합계** | **7.04** |

forward 43.31 ms 대비 **16%** 다. 사슬 경로에는 없는 비용이고, 이게 `cuda16`
의 수용률 우위(3.44 대 ~3.0)를 상쇄한다. 마스크는 0.20 ms 로 무시할 수준이다.

🔴 **이 표에서 `propose` 를 "파이썬 오버헤드" 로 읽으면 틀린다.** §20 에서
   쪼개 보니 3.19 ms 중 1.88 ms 는 D2H 지점에서 **드래프터 forward 가 끝나기를
   기다리는 정지**였다. 그 GPU 일은 어차피 해야 하는 것이므로 우리 비용이
   아니다. 실제 추가 비용은 7.04 가 아니라 **약 5.2 ms/스텝** 이고, 그중 가장
   큰 항목은 `propose` 가 아니라 `accept`(1.84) 와 `kv_compact`(1.30) 다.

## 19-6. 계측 자체의 한계

- `enforce_eager` 전용이다. torch.compile 이 켜지면 nn.Module 훅이 트레이스에
  흡수되어 안 불린다. 프로덕션(컴파일+CUDA 그래프) 비율은 다를 수 있다.
- 훅 구간에는 CPU 런치 공백이 포함된다 (§19-2). 절대값이 아니라 **폭에 따른
  증가분**과 **커널 프로파일**로 읽어야 한다.
- 단일 요청·단일 프롬프트다. 배치가 커지면 GEMM 이 커지고 GDN/어텐션 비중이
  둘 다 내려간다 — 그래도 둘의 **상대 비**는 안 바뀔 것으로 본다 (미측정).


---

# 20. propose 배치 topk — 배치에서 1.64배, 단일 요청에서는 0 (2026-08-31)

§19-5 에서 `propose` 3.19 ms/스텝을 다음 병목으로 지목했다. 고치면서
**그 진단의 절반이 틀렸다는 것**을 알았다.

## 20-1. 무엇을 고쳤나

`propose_from_drafter_logits` 가 요청마다 `build_tree_from_logits` 를 부르고,
그 안에서 `.to("cpu")` 가 두 번(logprob, 토큰 id) 났다. 즉 **D2H 가 배치당
2n회**. 각 복사는 앞선 GPU 작업이 끝날 때까지 파이프라인을 세운다.

`tree.topk_from_logits()` 를 새로 만들어 `[..., depth, vocab]` 을 한 번에 받게
하고, **D2H 를 스텝당 2회로 고정**했다. 트리 빌드는 그 뒤 순수 CPU 다.

float 임시본은 청크로 자른다 — 배치 전체를 `[요청 x 깊이 x vocab]` fp32 로
올리면 요청 64 · vocab 151936 에서 583 MiB 다. 상한은
`VLLM_DDTREE_CAST_MIB` (기본 64).

## 20-2. 등가성

`tests/ddtree/t29_batch_topk.py` — 요청별 경로와 배치 경로가 **같은 트리**를
내는지 (`token_ids`/`parents`/`depths`/`dyn_mode`/`e_chain`) 검사한다.
요청 1·2·5, 손잡이 7조합(spine, depth_bonus, dynamic_tau, dynamic_tau+short,
topk_cap=1/4), 청크 경로 강제, 반환 모양 — **전부 통과**.

회귀: `t4_tree_build` 12/12 (참조 구현과 동일), `t17_pad` 통과.
e2e: `cuda16` 이 변경 전후로 **출력 바이트 단위 동일**, 수용도 124/36 로 동일.

## 20-3. 실측

배치 크기별, 이 구간만 (vocab 151936, 깊이 15, 예산 16, A6000):

| 요청수 | 요청별 ms | 배치 ms | 배속 | (배치 내역 topk / 빌드) |
|---|---|---|---|---|
| 1 | 0.40 | 0.39 | 1.04x | 0.35 / 0.06 |
| 4 | 1.56 | 1.14 | 1.37x | 0.99 / 0.17 |
| 16 | 6.08 | 3.83 | 1.59x | 3.25 / 0.61 |
| 64 | **24.49** | **14.96** | **1.64x** | 12.37 / 2.33 |

e2e(단일 요청)는 61.6 → 62.4 tok/s 인데, 같은 설정 반복 측정이 61.0~62.4 로
**±1.2% 흔들린다. 측정 가능한 변화가 없다** — n=1 은 원래 D2H 가 2회라 당연하다.

`VLLM_DDTREE_CAST_MIB` 튜닝 (요청 64, 결과는 상한과 무관하게 동일):

| 상한 | 시간 | 최대 할당 |
|---|---|---|
| 16 MiB | 15.67 ms | 310 MiB |
| **64 MiB** | **12.34 ms** | **406 MiB** |
| 256 MiB | 11.47 ms | 790 MiB |
| 무제한 | 11.13 ms | 1391 MiB |

배치가 큰 상황은 곧 메모리가 빠듯한 상황이라 64 MiB 를 기본으로 뒀다.

## 20-4. 🔴 §19-5 의 진단 정정 — `propose` 는 대부분 '대기'였다

`p_topk` 가 3.006 ms/스텝인데 같은 모양의 마이크로벤치는 0.35 ms 였다.
차이가 어디서 나는지 `t31_chunk.py` 로 쟀다 — GPU 작업을 큐에 넣고 sync 없이
바로 topk 구간의 CPU 벽시계를 재면:

    밀린 GPU 작업  0.00 ms  ->  topk 구간  0.327 ms
    밀린 GPU 작업  2.16 ms  ->  topk 구간  2.514 ms
    밀린 GPU 작업  9.56 ms  ->  topk 구간  9.856 ms
    밀린 GPU 작업 19.28 ms  ->  topk 구간 19.398 ms

**정지분이 밀린 GPU 작업과 정확히 선형이다.** D2H 는 드래프터 forward 가
끝나기를 기다리고 있었다.

`VLLM_DDTREE_TIMESPLIT=1` (계측 전용 — topk 앞에서 한 번 동기화) 로 분리하면:

| 구간 | ms/스텝 | |
|---|---|---|
| p_wait | **1.883** | 드래프터 forward 대기 — GPU 일, 우리 비용 아님 |
| p_topk | 1.228 | |
| p_build | 0.103 | 힙 트리 빌드 (순수 CPU) |
| p_out | 0.033 | |
| mask | 0.197 | |
| accept | 1.842 | ← 이제 여기가 최대 |
| kv_compact | 1.303 | |
| gdn_compact | 0.489 | |
| 합계 | 7.106 | |
| **실제 추가 비용** | **5.222** | 대기를 뺀 값 |

즉 §19-5 의 "7.04 ms 파이썬 오버헤드" 는 과대 계상이었다. 실제는 **5.2 ms**
이고, 남은 최대 항목은 `propose` 가 아니라 **`accept`(1.84) 와
`kv_compact`(1.30)** 다. 다음 작업은 그쪽이다.

## 20-5. 남는 것

- 요청 64 에서 topk 자체가 아직 12.37 ms 다. fp32 캐스트가 트래픽의 대부분인데,
  bf16 topk 로 바꾸면 동점 처리가 달라져 트리가 바뀔 수 있다 (무손실은
  유지되지만 참조 구현과의 대조가 깨진다). 미착수.
- 다중 요청 경로 자체가 아직 안 붙었다 (§18 이후 미해결). 이 변경의 이득은
  그게 붙어야 e2e 에 나타난다.


---

# 21. accept / kv_compact — 정작 큰 건 **DDTree 를 끈 사용자에게 물리던 비용**이었다 (2026-08-31)

§20 에서 다음 표적으로 `accept`(1.84) 와 `kv_compact`(1.30) 를 지목했다.
쪼개 보니 `accept` 는 손댈 게 없었고, 대신 **PR 이 그대로 나갔으면 회귀였을
문제**를 찾았다.

## 21-1. accept 는 이미 공짜다

`VLLM_DDTREE_TIMESPLIT=1` 로 재면:

| 구간 | ms/스텝 |
|---|---|
| a_wait | **1.664** ← 타깃 forward 대기. GPU 일이다 |
| a_meta | 0.050 |
| a_argmax | 0.075 |
| a_loop | 0.039 |
| a_out | 0.017 |
| **실제 작업** | **0.181** |

`accept` 의 89% 가 대기였다. 최적화할 대상이 아니다.

## 21-2. GDN 계층마다 D2H 를 두 번씩 하고 있었다

`qwen_gdn_linear_attn.py` 의 DDTree 훅이 트리 정보를 세울 때마다
`cu_seqlens[...].tolist()` 와 `state_indices[...].tolist()` 를 불렀다.

  - 계층당 2회 x GDN 계층 24개 = **스텝당 48회 D2H**
  - 내용은 계층마다 **동일하다** (같은 스텝의 같은 메타데이터)

고친 것 — `gdn.tolist_cached()` 로 스텝 안에서 한 번만 내린다
(`runtime.begin_step` 이 `gdn.new_step()` 으로 무효화). 48회 → 2회.

### 🔴 정정 — "DDTree 를 꺼도 부담한다" 는 틀린 진단이었다

처음엔 이걸 **밀어낸 커밋에 들어간 회귀**로 적었다. `.tolist()` 가
`_ddtree_gdn()` 의 None 검사보다 **앞**에 있으니 DDTree 를 안 켠 사용자도
스텝당 48회를 문다는 이야기였다. 틀렸다.

`git show HEAD:...qwen_gdn_linear_attn.py` 를 확인하니 밀어낸 코드에는
`.tolist()` 가 이미 `if _info is not None:` **안**에 있다. 무방비 호출은
아직 커밋 안 한 `ddtree-multireq` WIP 의 작업 트리에만 있었다. 즉

  - 회귀는 **밀려나가지 않았다** — 아무에게도 물린 적이 없다
  - 고친 것(`_ddtree_active()` 선검사)은 그 WIP 에만 해당한다

같이 적었던 "DDTree 를 끈 경로가 +3.7% 빨라졌다 (62.5 → 64.8)" 도 근거가 못 된다.
두 값은 **WIP 이 섞인 빌드**에서 나왔고, 그 폭은 컨테이너 간 잡음(DDTree 를 끈 경로 62.3–65.0 tok/s) 안에 통째로 들어간다. 깨끗한 빌드로 다시 재면
DDTree 를 끈 경로는 HEAD 63.0 대 ddtree-perf 63.9 로, 구별되지 않는다.

교훈: **한 프로세스 한 번씩 잰 두 수를 차이의 증거로 쓰지 마라.** 여기서
프로세스 간 편차는 프로세스 내 편차보다 크다 (§22).

## 21-3. kv_compact — 그룹 루프에서 인덱스 재생성, 그룹마다 동기화

  - 수용 경로 인덱스(`as_tensor(acc)` / `arange`)를 **그룹 루프 안**에서 요청마다
    다시 만들었다. 하이브리드는 캐시 그룹이 3개라 그대로 3배. 그룹과 무관한
    값이므로 밖으로 뺐다 — H2D 가 (그룹 x 요청) 번에서 **3번**으로.
  - 순서 안전성 검사 `int((_dst > _src).sum().item())` 을 그룹마다 불러 스텝당
    3회 동기화했다. 전부 `torch.stack` 으로 모아 **1회**로.

## 21-4. 결과

구간별 (ms/스텝, TIMESPLIT):

| 구간 | 이전 | 이후 | 차이 |
|---|---|---|---|
| c_idx | 0.331 | 0.167 | −0.164 |
| c_viol | 0.167 | 0.147 | −0.019 |
| c_kernel | 0.700 | 0.611 | −0.089 |
| gdn_compact | 0.492 | 0.481 | −0.011 |
| (그 외 변화 없음) | | | |
| 합계 | 7.153 | 6.744 | −0.408 |
| **대기 제외 실제 CPU** | **3.606** | **3.192** | **−0.414** |

e2e (트리 폭 17, 반복 측정):

| | tok/s | GDN 혼합 ms |
|---|---|---|
| 이전 (n=5) | 61.62 ± 0.51 | 17.31 ± 0.16 |
| 이후 (n=4) | **62.94 ± 0.86** | **16.64 ± 0.19** |

GDN 은 편차 밖의 확실한 개선(−0.67 ms)이고, tok/s 는 +2.1% 로 편차의 1.5~2배라
방향은 맞지만 단독으로는 결정적이지 않다.

## 21-5. 무손실 / 회귀

  - 출력이 DFlash 사슬(`flat15`) 과 **바이트 단위 동일**
  - 수용 124/36, `kv_groups` 108, `kv_rows` 480 — 변경 전과 완전 동일
  - `compact_unsafe` 0, `gdn_compact_err` 없음
  - `t5_compact` 7/7, `t29_batch_topk` 전부, `t4_tree_build` 12/12, `t17_pad` 통과

## 21-6. 남은 그림

폭 17 기준 스텝 ≈ 42.4 ms(forward) + 3.2 ms(우리 CPU) + 대기.
DFlash 사슬은 45.0 ms 에 우리 CPU 가 없다. 즉 **forward 는 우리가 2.6 ms 싸고,
CPU 로 3.2 ms 를 도로 낸다.** 남은 항목은 `p_topk` 1.22 (요청 1개에도 나는
topk 지연), `c_kernel` 0.61 (Triton 런치 x 3그룹), `gdn_compact` 0.48 이다.
전부 한 자릿수 % 수준이라, 여기서 더 짜내는 것보다 **다중 요청 경로**를 붙이는
쪽이 이득이 크다 (배치 topk 는 n=64 에서 1.64배로 이미 준비돼 있다).

# 21-7. 🔴 용어 — 기준선을 "순정" 이라 부르지 않는다

§16~§27 에서 기준선을 "순정" 이라고 뭉뚱그려 불렀다. 두 가지가 잘못이었다.

  1. 축마다 **다른 것**을 가리켰다.
  2. "순정" 은 **스펙 디코딩(MTP) 자체를 끈 빌드**로 읽히기 쉽다. 실제로는
     전부 스펙 디코딩이 켜진 구성이고, 다른 것은 **검증 모양(사슬/트리)** 뿐이다.

그래서 기준선을 드래프터 이름 + 검증 모양으로 부른다:

| 축 | 기준선 | 이름 |
|---|---|---|
| 4B 하이브리드 | DFlash 드래프터를 **사슬로** 검증 | **DFlash 사슬** |
| 27B 하이브리드 | DFlash2 드래프터를 **사슬로** 검증 | **DFlash2 사슬** |
| 0.6B·8B 순수 어텐션 | vLLM `NgramProposer` 를 사슬로 검증 | **ngram 사슬** |
| (참조) 스펙 디코딩 없음 | 드래프터 자체가 없음 | **base** |

하이브리드 축에서 두 팔은 **같은 드래프터를 쓴다.** 다른 것은 그 드래프터의
깊이별 출력을 사슬로 검증하느냐 트리로 검증하느냐뿐이다. 즉 그 비교는
"vLLM 대 DDTree" 가 아니라 **"사슬 검증 대 트리 검증"** 이다.

## 21-7-1. 4B 드래프터는 원본이 아니다

`/hf/drafter-4b-full` 은 `z-lab/Qwen3.5-4B-DFlash` 가 아니라 **우리가 변환한
사본**이다. config 에 흔적이 있다:

    _ddtree_note: layer_types [sliding x5, full] -> all full;
                  valid only for seq_len <= 4096

V1 러너가 SWA/full 혼합을 거부해서 전부 full 로 바꿨다. 측정은
`max_model_len=1024` 라 범위 안이지만, 4096 을 넘기는 실험에는 쓸 수 없다.

## 21-7-2. DFlash 와 DFlash2 는 다른 체크포인트다

| | DFlash (4B) | DFlash2 (27B) |
|---|---|---|
| 클래스 | `DFlashDraftModel` | `DFlash2DraftModel` |
| 층 / hidden | 6 / 2560 | 5 / 5120 |
| `num_target_layers` | 32 | 64 |
| 어텐션 | full x6 (변환본) | sliding x5, window 2048 |
| `is_causal` | (없음) | **False** — 블록 내 양방향 |
| `tie_word_embeddings` | True | False |
| **`dflash_config`** | `block_size: 16`, `mask_token_id`, `target_layer_ids[8]` | **없음** |

🔴 마지막 줄이 측정에 직접 영향을 준다. `dflash.py` 는
`drafter_k = (dflash_config or {}).get("block_size") or (1 + num_speculative_tokens)`
에서 1을 뺀다. 즉

  - **4B**: `block_size=16` 이 있으므로 예산을 키워도 `drafter_k=15` 로 고정된다.
  - **27B**: `dflash_config` 가 아예 없으므로 **`drafter_k = num_speculative_tokens`**.
    예산을 키우면 드래프터의 지평도 같이 커진다.

같은 파일의 주석이 이 위험을 이미 경고하고 있다 — *"결합 상태에서 예산을 키우면
크래시 없이 conv 만 조용히 어긋난 폭으로 동작한다."*

# 22. DFlash 사슬 대 DDTree 트리 — 1% 를 가르는 데 든 것

> 🔴 **이 절의 +1.2% 는 실배치를 대표하지 못한다. §23 을 먼저 읽어라.**
> 여기 숫자는 전부 `enforce_eager` + 계층 프로파일러 훅이 붙은 상태에서 나왔다.
> 두 조건 모두 CPU 여유를 만들어 DDTree 의 비용을 숨긴다. cudagraph 를 켜면
> 같은 비교가 **-31.6%** 로 뒤집힌다.


§19-4 에서 CUDA GDN 트리 커널을 켠 뒤 DDTree 는 DFlash 사슬과 "비슷해" 보였다.
비슷하다는 말로 끝내면 안 되는 이유는, 그 전까지 우리가 단발 측정 두 개를
빼서 +3.7% 니 -14% 니 하는 이야기를 몇 번 했기 때문이다 (§21-2 정정).

## 22-1. 왜 n=3 으로는 안 갈렸나

이 벤치의 분산 구조:

| | sd (tok/s) |
|---|---|
| 컨테이너 **안** (한 프로세스에서 5회) | 0.50 |
| 컨테이너 **사이** (프로세스 평균끼리) | 0.75 |

프로세스 간 편차가 더 크다. 모델 적재 배치, 할당자 상태, 그 순간의 클럭·열이
프로세스마다 고정되고 그 안의 5회는 그 상태를 **공유한다** — 독립 표본이
아니다. 그래서 "한 팔에 15회" 는 실제로는 n=3 이고, 0.9 tok/s 차이에
SE 0.55 가 붙어 d/SE 1.7 에서 멈춘다.

## 22-2. 실패한 설계 — 프로세스 안에서 팔 바꿔 끼우기

프로세스 항을 없애려고 `runner.ddtree` 를 끼웠다 뺐다 해봤다. 모든 사용처가
`self.ddtree is not None` 을 호출 시점에 보므로 될 것 같았다. 두 번째 전환에서
죽는다:

    scheduled_spec_decode_tokens={...: [-1] x 16}
    IndexKernel.cu:111: Assertion `index < sizes[i]` failed

DDTree 를 껐다 다시 켜면 요청별 노드 테이블과 `q_start` 가 앞 스텝 것으로
남아, 재활성 스텝이 전부 패딩(-1)인 트리를 내놓는다. **참조를 끊는 것과
상태를 되감는 것은 다르다.** 계측 편의를 위해 실코드의 상태 머신을 건드릴
일은 아니라 접었다.

여기서 건진 것: 이 검증(DFlash 사슬 팔에서 `ddtree_steps == 0` 인가)을 배치 시작
**전에** 게이트로 걸어놨다는 것. 안 걸었으면 깨진 실행이 남긴 결과를
"차이 없음" 으로 읽었을 것이다.

## 22-3. 쓴 설계 — 컨테이너 쌍, 쌍 안에서 순서 뒤집기

컨테이너를 F/T 로 번갈아 띄우되 **쌍**으로 묶고, 쌍 안의 순서를 번갈아
뒤집었다 (F T / T F). 쌍차로 검정하면 배치 내내 도는 열 흐름(GPU0 이
38°C 에서 60°C 로 데워졌다)이 상쇄된다. 순서효과도 없었다 — F 먼저인
쌍의 평균 쌍차 +0.82, T 먼저인 쌍 +0.65.

## 22-4. 결과 (폭 17 트리 대 DFlash 사슬, 같은 빌드, 쌍 11개)

| | tok/s |
|---|---|
| DFlash 사슬 | 63.73 ± 0.75 |
| DDTree 트리 (CUDA 커널) | 64.51 ± 0.53 |

    쌍차 평균  +0.787 tok/s  (+1.23%)
    95% 구간   [+0.21%, +2.26%]        쌍 t 검정, t=2.71, df=10

큰 쌍차 두 개(+2.05, +2.75)에 기댄 결과가 아닌지 봤다:

| 검정 | 결과 |
|---|---|
| 양끝 1개씩 절사 후 쌍 t | **+1.09%**, 95% [+0.24%, +1.94%] — 오히려 좁아진다 |
| 순열검정 (부호뒤집기 2048회 전수) | 양측 **p = 0.016** |
| 부호검정 (8/11 양수) | 양측 p = 0.227 — **유일하게 안 갈린다** |
| 이번 배치만 (쌍 4~11) | +1.16%, 95% [-0.16%, +2.47%] — 0 을 걸친다 |

부호검정은 크기를 버리고 부호만 보므로 n=11 에서 검정력이 거의 없다.
절사 평균이 원 평균과 같고 순열검정이 0.016 이면, 이상치가 만든 결과가
아니라고 봐도 된다.

## 22-5. 그래서 뭐라고 말할 수 있나

  - **0 은 배제된다.** DDTree 는 DFlash 사슬보다 느리지 않다.
  - **크기는 1% 안팎이다.** 하한이 +0.2% 라, "의미 있게 빠르다" 고 팔 만한
    폭은 아니다. 상한도 +2.3% 다.
  - 이건 **배치 1, 프롬프트 1개, 128토큰, enforce_eager** 의 숫자다. eager 는
    스텝이 통째로 런치 바운드라 (§19-2) DDTree 의 파이썬 비용이 최대로
    드러나고, 동시에 GPU 절감분도 가려진다. cudagraph 를 켠 실배치에서
    이 1% 가 어느 쪽으로 움직일지는 **재보지 않았다**.
  - DFlash 사슬 팔은 폭 16(flat15), DDTree 팔은 폭 17 이다. 프로세스 안에서
    `num_speculative_tokens` 를 못 바꿔서 정확히 맞추지 못했다.

## 22-6. 방법론으로 남길 것

  - **단발 두 수의 차이를 결론으로 쓰지 마라.** 이 벤치에서 컨테이너 하나의
    값은 ±1.5 tok/s(2%) 로 흔들린다. 우리가 이걸로 세 번 틀렸다.
  - 반복은 **프로세스를 새로 띄워야** 반복이다. 한 프로세스 안의 5회는
    n=5 가 아니다.
  - 잡음이 큰 쪽으로 결론이 기울 때는 검정을 하나 더 대라. 여기선 절사
    평균과 순열검정이 t 검정을 지지했고, 부호검정 하나가 반대였는데 그건
    검정력 문제였다.

# 23. cudagraph 를 켜면 -31.6% — §22 가 잰 것은 조건의 산물이었다

§22 는 쌍 11개를 모아 +1.23% [+0.21%, +2.26%] 를 얻고 "0 은 배제된다" 로
닫았다. 그 문장은 **eager 안에서만** 참이다.

## 23-1. 네 칸

같은 빌드, 같은 프롬프트, 스텝 수를 직접 세서 (`execute_model` 호출 수):

| 조건 | 팔 | tok/s | 스텝 | ms/스텝 |
|---|---|---|---|---|
| eager | DFlash 사슬 | 70.9 | 34 | 53.21 |
| eager | DDTree | 70.7 | 33 | 53.97 |
| cudagraph | DFlash 사슬 | **144.9** | 34 | **26.28** |
| cudagraph | DDTree | **99.1** | 34 | **37.91** |

    eager      단가차  +0.76 ms/스텝 (+1.4%)    처리량  -0.3%
    cudagraph  단가차 +11.63 ms/스텝 (+44.3%)   처리량 -31.6%

컨테이너 4쌍에서 DFlash 사슬 143.05 ± 1.27, DDTree 99.00 ± 0.43. 격차가 sd 의
35배라 검정이 필요 없다. 출력은 네 조합 전부 바이트 단위로 같다 —
cudagraph 에서도 무손실이다.

## 23-2. 트리가 수용을 하나도 못 번다

**둘 다 34 스텝이다.** 같은 128토큰을 같은 스텝 수에 뽑는다. 폭 17 트리가
폭 16 사슬보다 더 수용하지 못한다는 뜻이고, 그러면 DDTree 는 스텝 단가만
더 내고 돌려받는 게 없다.

이건 이 프롬프트/이 설정의 성질일 수 있다 — 온도 0, 짧은 사실질의라 사슬
수용률이 이미 높아 가지가 먹을 자리가 없다. **드래프터가 헷갈리는 입력에서
다시 재야 트리의 값어치를 안다.** 지금 벤치는 트리에게 유리한 조건이 아니다.

## 23-3. 왜 eager 는 숨겼나

DDTree 의 파이썬 구간 (ms/스텝, TIMESPLIT):

| 구간 | eager | cudagraph |
|---|---|---|
| propose | 2.739 | 3.782 |
| └ p_wait | 1.875 | 2.930 |
| accept | 1.874 | 2.221 |
| └ a_wait | 1.667 | 2.018 |
| kv_compact | 0.877 | 0.860 |
| gdn_compact | 0.504 | 0.501 |
| **합계** | 6.194 | 7.555 |
| 그중 대기 | 3.542 | 4.949 |
| **실 CPU 작업** | **2.652** | **2.607** |

실 CPU 작업은 **두 조건에서 같다** (2.65 대 2.61). 파이썬이 늘어난 게 아니다.

그런데 eager 단가차는 +0.76 ms 뿐이다 — DDTree 가 실제로 2.65 ms 의 CPU 를
쓰는데도. 나머지가 GPU 배수(drain) 시간에 겹쳐 들어간 것이다. cudagraph
단가차는 +11.63 ms 인데 파이썬은 여전히 2.6 ms 다. **차액 약 9 ms 는 GPU
쪽 추가 작업**(KV 압축 커널, 트리 GDN 커널, custom_mask 어텐션)이고,
eager 에서는 런치 바운드로 비어 있던 GPU 유휴 시간에 공짜로 들어갔다.

정리하면: eager 스텝은 53 ms 중 상당 부분이 런치 대기라 DDTree 가 CPU 로
2.6 ms, GPU 로 9 ms 를 더 써도 거의 티가 안 난다. cudagraph 가 그 유휴를
26 ms 로 압축하면 9 ms 가 전액 청구된다.

## 23-4. 방법론 — §22-6 에 추가할 것

  - **`enforce_eager` 로 잰 속도차를 실배치 결론으로 쓰지 마라.** eager 는
    런치 바운드라 (§19-2) 추가 작업을 숨기는 성질이 있다. 여기서는 그
    성질이 -31.6% 를 +1.2% 로 보이게 만들었다.
  - 계측 훅도 같은 방향으로 작용한다. §22 의 절대 수치(DFlash 사슬 63.7)는
    프로파일러 훅이 붙은 값이고, 훅 없는 eager DFlash 사슬은 70.9 다.
  - **스텝 수를 세라.** DFlash 사슬 팔의 스텝 수를 몰라 38 로 가정하고 ms/스텝 을
    계산했다가 원인 추론을 통째로 틀릴 뻔했다. 실측은 34 였고, 그 순간
    "트리가 수용을 못 번다" 는 진짜 문제가 드러났다.

# 24. 순수 어텐션 재검증 — §17 은 재현되지 않는다

§23 이 하이브리드에서 -31.6% 를 확인한 뒤 남은 희망은 "GDN 없는 순수 어텐션
에서는 이긴다(§17)" 였다. 같은 구성을 **cudagraph 를 켜고, 스텝 수를 세서**
다시 쟀다.

## 24-1. 여섯 칸 + 깊이 훑기

Qwen3-0.6B (순수 어텐션) + ngram 드래프터, 프롬프트 4개 합계초, 3바퀴 최저:

| | cudagraph | 무손실 | eager | 무손실 |
|---|---|---|---|---|
| base (스펙 없음) | **1.045s** | 4/4 | 6.621s | 4/4 |
| ngram 사슬 폭16 | 1.217s | 3/4 | **3.394s** | 4/4 |
| DDTree 깊이8 | 1.469s | 0/4 | 4.063s | 4/4 |
| DDTree 깊이12 | 1.433s | 0/4 | 3.885s | 4/4 |
| DDTree 깊이16 | 1.399s | 1/4 | 3.848s | 4/4 |

**DDTree 는 모든 조건에서 ngram 사슬에 진다** (cudagraph +15%, eager +13%,
깊이 맞춘 기준). cudagraph 만의 문제가 아니다.

## 24-2. cudagraph 는 스펙 디코딩의 값어치 자체를 뒤집는다

    eager      base 6.621 -> ngram 3.394   스펙이 1.95배 이긴다
    cudagraph  base 1.045 -> ngram 1.217   스펙이 진다

0.6B 는 cudagraph 에서 스텝이 2.66 ms 다. 검증 토큰을 16개 더 얹는 비용
(스텝 6.3 ms)이 스텝 절감을 넘어선다. eager 에서는 스텝이 68 ms 라 그 16개가
거의 공짜였다.

§17 이 잰 조건은 §23 의 하이브리드보다 **더** 런치 지배적이었다 (base 가
cudagraph 로 6.3배 빨라진다). 스펙 디코딩이 비용을 숨길 자리가 그만큼 넓었다.

## 24-3. §17 이 왜 뒤집혔나 — 대조군이 강해졌다

§17 의 헤드라인은 "파리 프롬프트에서 ngram 사슬 42.3 < base 55.8, 트리가
117.8 로 뒤집는다" 였다. 지금 ngram 사슬의 파리는 **177.8 tok/s (30스텝)** 다.

vLLM 의 ngram 제안기가 좋아졌는지 당시 값이 이상했는지는 모른다. 어느 쪽이든
**§17 이 기댔던 병리적 대조군이 없어졌다.**

## 24-4. 깊이 제한이 원인의 일부였다

`budget=16, depth_limit=8` 이면 ngram 사슬은 예산 16을 전부 깊이에 쓰는데
DDTree 는 8단에서 잘리고 남은 예산을 분기에 쓴다. **트리 대 사슬이 아니라
8단 대 16단이었다.**

깊이를 16으로 맞추면 "반복" 프롬프트가 15 -> 8스텝으로 ngram 사슬과 같아지고,
트리스텝당 수용이 2.63 -> 3.70 으로 오른다. 합계도 1.469 -> 1.399 로 준다.
그래도 ngram 사슬(1.217)에는 못 미친다 — 깊이를 올릴수록 좋아지고, 극한은 깊이=예산,
즉 **사슬 그 자체**다.

## 24-5. 🔴 그래도 "트리가 졌다" 로 읽으면 안 된다

    깊이16 cg:  전체 664스텝 중 트리가 있는 스텝 165 (24.8%)

**DDTree 는 스텝의 4분의 1에서만 트리를 만든다.** 나머지 75% 는 스펙 디코딩
없이 한 토큰씩 돈다. 즉 이 실험이 비교한 것은 **"DDTree 의 ngram 구현 + 트리"
대 "vLLM 의 ngram 제안기 + 사슬"** 이고, 격차의 상당 부분이 트리가 아니라
드래프터 통합 품질에서 온다. 원인은 §25 에서 계수기로 갈랐다.

## 24-6. 🔴 cudagraph 에서 무손실이 깨진다

eager 에서는 DDTree 가 4/4 무손실인데 cudagraph 에서는 0~1/4 이다. 원인 후보를
지웠다:

  - vLLM 공통 수치 문제가 아니다 — ngram 사슬은 cudagraph 에서도 3/4 을 지킨다.
  - KV 압축 안전 경로가 아니다 — 깊이16 은 `compact_unsafe=0` 인데도 깨진다.
  - 하이브리드에서는 안 난다 — §23 의 네 조합은 바이트 단위로 같았다.

**cudagraph + 순수 어텐션 조합의 DDTree 고유 문제**다. 마스크 공급자 쪽이
유력하다. 미조사.

## 24-7. 지금 서 있는 자리

| 축 | 결과 |
|---|---|
| GDN 하이브리드 + DFlash (§23) | cudagraph -31.6%, 트리의 수용 이득 0 |
| 순수 어텐션 + ngram (§24) | eager -13%, cudagraph -15% (ngram 사슬 대비) |

두 축 모두 부정적이다. 다만 어느 쪽도 **트리를 공정하게 평가하지 못했다** —
하이브리드는 드래프터(DFlash)가 너무 정확해 가지가 먹을 자리가 없었고 (§16),
어텐션은 트리를 스텝의 25% 에서만 만들었다 (§24-5).

# 25. 트리를 못 만들던 원인 — n-gram 길이를 하나만 봤다

§24-5 의 "트리가 스텝의 25% 에서만 만들어진다" 를 계수기로 갈랐다.

## 25-1. 원인 — n-gram 길이를 하나만 본다

`propose()` 의 포기 지점마다 계수기를 달았다 (깊이16, cudagraph):

    ng_nomatch    486   n-gram 이 아예 안 맞음
    ng_underfill    0   예산을 못 채워 트리 포기
    ng_ok         178

**전부 불일치다. 예산 미달 포기는 0회다.**

🔴 처음엔 §17-4 를 근거로 `pad_to_budget` 이 예산을 못 채우면 트리를 버리는
   것이라고 적었다. **틀렸다.** 그 경로는 이 워크로드에서 한 번도 타지 않는다.
   `VLLM_DDTREE_SHORT=1` 로 짧은 트리를 허용해도 통계가 **완전히 동일**하다
   (1.428s 대 1.421s, ng_ok 178 그대로).

진짜 원인은 `_ngram_distributions` 가 `self.ngram_n = 3` **하나만** 본다는
것이다. vLLM 의 ngram 제안기는 `prompt_lookup_min..max` (2~4) 를 긴 쪽부터 훑어
처음 맞는 길이를 쓴다. 3-gram 이 안 맞을 때 vLLM 제안기는 2-gram 으로 내려가 드래프트를
내는데, DDTree 는 그대로 포기한다.

## 25-2. 고친 뒤 (VLLM_DDTREE_NGRAM_SPAN=2-4)

n 을 2~4 로 훑게 하니 트리 있는 스텝이 25% -> 32% 로 오르고, 스텝 수에서
ngram 사슬을 이겼다 (167 대 193). 합계도 1.421 -> 1.128s.

> 🔴 **이 1.128s 는 무효다.** 같은 실행의 무손실이 1/4 였고, 원인은 §26 의
> RoPE 버그였다. 깊이 위치가 무시된 채 검증하고 있었으니 수용이 부풀려진
> 것이고, 애초에 유효한 측정이 아니다. 고친 뒤 수치는 §26-3 을 보라.

## 25-3. 남는 것

n-gram 길이를 하나만 보는 것은 **진짜 결함이 맞다** — vLLM 제안기와 맞추면
트리를 만드는 스텝이 늘어난다. 다만 그것만으로 ngram 사슬을 이기지는 못한다
(§26-3).

# 26. 🔴 cudagraph 에서 깊이 RoPE 가 통째로 무시되고 있었다

§24-6 의 "cudagraph 에서만 무손실이 깨진다" 를 잡았다.

## 26-1. 원인

`gpu_model_runner.py` 의 forward 준비:

    positions = self.positions[:num_input_tokens]      # 영속 정적 버퍼의 뷰
    if self.ddtree is not None and self.ddtree.active:
        _rp = self.ddtree.rope_positions(self.positions, num_input_tokens)
        if _rp is not None:
            positions = _rp                            # ← 별도 텐서

cudagraph 로 캡처된 조각은 **캡처 당시의 버퍼 주소**를 읽는다. 새 텐서를 만들어
지역 변수만 갈아끼우면 그래프는 그걸 못 보고 원래 `self.positions`(연속 위치)를
쓴다. **깊이 RoPE 가 조용히 무시된다.** eager 는 인자로 전달되므로 정상이다.

고친 것: 정적 버퍼에 **제자리로** 쓴다. 드래프터가 뒤에서 `self.positions` 를
다시 읽으므로(`propose` 의 `_get_positions`) 원본을 들고 있다가 타깃 forward
직후 되돌린다.

## 26-2. 검증

| 순수 어텐션, cudagraph | 합계 | 무손실 |
|---|---|---|
| 수정 끔 | 1.140s | **1/4** |
| 수정 켬 | 1.282s | **4/4** |

eager 에서는 수정 켬/끔의 **출력과 통계가 바이트 단위로 동일**하다 (4.116 대
4.131s, 컨테이너 잡음). 의도대로 eager 에서는 무연산이다.

## 26-3. 🔴 그 버그가 §25 의 "승리" 를 만들었다

고치면 수용이 트리스텝당 4.43 -> 2.84 로 떨어진다. 형제 노드가 전부 연속 위치를
받고 있었으니 트리가 아닌 무언가를 검증하며 수용을 부풀리고 있었다.

정정된 그림 (순수 어텐션, cudagraph):

| | 합계 | 무손실 |
|---|---|---|
| base | **1.045s** | 4/4 |
| ngram 사슬 | **1.217s** | 3/4 |
| DDTree (n범위 + RoPE 수정) | 1.282s | 4/4 |

**DDTree 는 정확하게 만들면 ngram 사슬에 5.3% 진다.** 스텝 수는 여전히 적지만
(184 대 193) 그 이득이 스텝 단가를 못 덮는다.

§23~§26 을 통틀어 **DDTree 가 정당하게 이긴 조건은 아직 없다.**

## 26-4. ngram 사슬의 "무손실 3/4" 는 버그가 아니다

vLLM 자체 문제인지 확인했다. base 는 eager/cudagraph 에서 4/4 동일하므로 기준은
흔들리지 않는다. 갈리는 건 cudagraph + 스펙 디코딩 + "코드" 프롬프트 하나이고,
`fibonacci(5)` 대 `fibonacci(10)` 토큰 하나다.

그 자리의 1위-2위 로짓 간격을 쟀다:

    #36  ' Output' vs ' Expected'   0.0625
    #22  'print'   vs 'def'         0.1250
    #27  '5'       vs '1'           0.1250   ← 여기서 갈림
    중앙값                          5.6250

간격이 전부 0.0625 의 배수다 — 값이 bf16 격자에 올라앉아 있다. 갈린 자리는
**2 ulp**, 전형적인 자리는 90 ulp 다. 96자리 중 0.25 이하가 8개뿐이고 그중
하나가 뒤집혔다.

스펙 디코딩은 그리디에서 수학적으로 등가지만, 검증 forward 가 토큰 17개를 한
번에 통과시키므로 GEMM 감산 순서가 base(1개)와 달라져 로짓의 마지막 비트가
바뀐다. cudagraph 는 inductor 가 모양별로 다른 커널을 컴파일해 차이를 키운다.
§18 의 "폭에 따라 반올림이 달라진다" 와 같은 원인이다.

**함의: "base 와 텍스트 동일" 은 과한 기준이다** — 사슬 검증도 통과 못 한다. 앞으로
무손실 판정은 같은 조건의 같은 드래프터의 사슬 검증과 비교해야 한다. 단, §26-1 의 RoPE
버그는 이것과 성격이 다르다. 2 ulp 뒤집힘이 아니라 4개 중 3개가 갈렸고 고치니
4/4 가 됐다 — 진짜 버그가 맞다.

# 27. 모델을 키워봤다 — 8B 순수 어텐션과 27B 하이브리드

§26 까지의 결론이 "0.6B 는 스펙 디코딩 자체가 손해라 판정 불가, 4B 하이브리드는
GDN 커널 때문에 -31.6%" 였다. 판정 가능한 크기로 올려 다시 쟀다.

## 27-1. Qwen3-8B 순수 어텐션 + ngram

먼저 관문: 스펙 디코딩이 값을 하는가.

| 8B cudagraph | 합계 | 스텝합 | ms/스텝 |
|---|---|---|---|
| base | 9.062s | 388 | 23.3 |
| ngram 사슬 | **5.926s (-34.6%)** | 223 | 26.4 |

통과한다. 검증 토큰 16개를 얹는 비용이 **13%** 뿐이다 (0.6B 는 2.4배였다).
모델이 커질수록 폭이 싸진다.

DDTree (RoPE 수정 + NGRAM_SPAN=2-4 + 깊이16):

| | 합계 | 대 기준선 | 트리스텝당 수용 |
|---|---|---|---|
| ngram 사슬 | 5.926s | — | — |
| DDTree topk=1 (순수 사슬) | 5.953s | **+0.5%** | 2.78 |
| DDTree topk=4 | **5.854s** | -1.2% | 2.94 |
| DDTree topk=64 | 5.877s | -0.8% | 2.94 |

    ngram 사슬 -> topk=1    +0.5%   DDTree 기계장치 자체 비용
    topk=1 -> topk=4  -1.7%   분기의 순효과

🔴 **순수 어텐션에서 DDTree 오버헤드는 +0.5% 다.** 그리고 **분기는 실제로
   값을 한다.** §24-4 에서 깊이 훑기로 "분기가 손해" 라고 읽었던 것은 틀렸다 —
   깊이를 줄이면 분기가 늘지만 깊이도 같이 줄어 둘이 안 갈린다. 깊이를 고정하고
   분기만 끄면 반대가 나온다.

못 이긴 이유는 다른 데 있다:

    전체 671 스텝 중 481 (72%) 이 n-gram 불일치 -> 트리 이전에 드래프트가 없다
    남은 28% 에서 분기가 수용을 +5.8% 올리지만, 전체로는 -1.7% 로 희석된다

**병목은 DDTree 가 아니라 드래프터 커버리지다.**

## 27-2. Qwen3.8-27B-AWQ-INT4 하이브리드 + DFlash2

"모델을 키우면 DDTree 의 고정 오버헤드가 희석된다" 는 가설을 검증했다.
**기각됐다.**

| | 스텝 단가 | DDTree 추가분 |
|---|---|---|
| 4B 하이브리드 | 26.3 -> 37.9 ms | +44% |
| **27B 하이브리드** | 44.8 -> 66.6 ms | **+49%** |
| 8B 순수 어텐션 | — | **+0.5%** |

절대량은 +11.6 -> +21.8 ms 로 **커졌고** 비율은 그대로다. 오버헤드는 고정이
아니라 **모델에 비례한다** — 트리 GDN 커널과 gdn_compact 가 계층 수·은닉
차원에 비례하기 때문이다. 8B 순수 어텐션의 +0.5% 와 대조하면 원인이 GDN
이라는 것이 분명하다 (§19 와 일치).

| 27B cudagraph | tok/s | ms/스텝 | 토큰/스텝 |
|---|---|---|---|
| DFlash2 사슬 flat7 (폭8) | **78.9** | 44.8 | 3.54 |
| DDTree 폭8 | 50.7 (-35.8%) | 66.6 (+48.6%) | 3.38 |
| DDTree 폭17 | 49.5 (-37.3%) | 73.6 (+64.1%) | 3.63 |

수용 이득은 폭 17 에서도 +2.5% 뿐이다 (3.54 -> 3.63). §16 의 "DFlash2 가 너무
정확해 가지가 안 먹힌다" 가 27B 에서도 그대로다.

### 🔴 폭 17 행은 근거로 쓰면 안 된다

§21-7-2 대로 27B DFlash2 에는 `dflash_config` 가 없어 **`drafter_k` 가
`num_speculative_tokens` 를 따라간다.** 예산 16 으로 돌리면 드래프터에게 16개
위치를 요구하는데, 이 체크포인트의 학습 지평은 8 이다 (그래서 사슬이 폭 8 이다).
`dflash.py` 주석이 경고한 "크래시 없이 conv 만 조용히 어긋난 폭으로 동작" 하는
상태다.

  - **폭 8 행(-35.8%)은 유효하다** — 사슬과 폭·지평이 모두 일치한다.
  - **폭 17 행(-37.3%)은 무효다** — 드래프터를 학습 지평 밖으로 밀어냈다.

결론은 폭 8 행이 지탱한다. 폭 17 을 제대로 재려면 `drafter_k` 를 8 로 고정한
채 트리 예산만 16 으로 키워야 한다 (4B 는 `block_size` 가 있어 이게 자동으로
된다). 미실행.

## 27-3. 🔴 갈린 것은 DDTree 가 아니라 vLLM 의 fused GDN 커널이었다

27B 에서 DDTree 출력이 DFlash2 사슬과 달라 무손실을 의심했다. §18-5 가 지목한 무손실
기준(`VLLM_GDN_DECODE_KERNEL=triton`)으로 대조했다.

| | CUDA사슬 | CUDA트리8 | CUDA트리17 | triton사슬 | triton트리8 |
|---|---|---|---|---|---|
| CUDA 사슬 | — | False | False | False | False |
| CUDA 트리8 | False | — | True | True | True |
| CUDA 트리17 | False | True | — | True | True |
| triton 사슬 | False | True | True | — | True |
| triton 트리8 | False | True | True | True | — |

**다섯 중 넷이 일치하고 홀로 다른 것은 CUDA 사슬이다.** 폭 8 은 우리 트리
커널을 타지 않으므로(8 > MAX_FUSED_GDN_MTP_TOKENS 가 거짓) vLLM 자체의 fused
GDN MTP 커널로 간다. 그 커널이 자기 triton 경로와 갈린다.

**DDTree 는 무손실이다.** 기준으로 삼았던 쪽이 틀렸다. 같은 커널끼리(triton)
비교해도 성능 결론은 같다 — DDTree -35.6%, 스텝 단가 +65.7%.

## 27-4. 지형 정리

| 축 | DDTree 대 기준선 | 병목 |
|---|---|---|
| 4B GDN 하이브리드 | -31.6% | GDN 커널 (스텝 +44%) |
| 27B GDN 하이브리드 | **-36%** | GDN 커널 (스텝 +49%) |
| 0.6B 순수 어텐션 | +5.3% | 모델이 작아 스펙 자체가 손해 |
| 8B 순수 어텐션 | **-0.8% (동률)** | 드래프터 커버리지 28% |

  - **GDN 하이브리드는 구조적으로 불가**하다. 오버헤드가 모델에 비례하므로
    키워도 안 줄어든다.
  - **순수 어텐션은 오버헤드가 없다** (+0.5%). 분기도 값을 한다 (-1.7%).
    남은 문제는 드래프터가 스텝의 72% 에서 아무것도 못 준다는 것이다.
  - 필요한 조합은 **순수 어텐션 + 커버리지 높고 불확실한 드래프터** 인데,
    가진 드래프터 둘 다 아니다 (ngram=커버리지 28%, DFlash2=너무 정확).

# 28. 🔴 러너 축을 놓치고 있었다 — 운영은 V2, 우리 측정은 전부 V1

## 28-1. 사실

    운영 컨테이너 로그:
      INFO [gpu_worker.py:397] Using V2 Model Runner
      speculative_config=SpeculativeConfig(method='dflash', num_spec_tokens=7)
      cudagraph_mode: FULL_AND_PIECEWISE, enable_prefix_caching=True, kv_cache_dtype=fp8

`vllm/config/vllm.py:_get_v2_model_runner_unsupported_features()` 기준:

  - **지원**: eagle, eagle3, mtp, **dflash**, dspark, extract_hidden_states
  - **미지원**: ngram / ngram_gpu -> V1 로 폴백

즉 **하이브리드 축(dflash)은 V2 가 기본값**이다. 우리 하네스가
`VLLM_USE_V2_MODEL_RUNNER=0` 을 강제하고 있어서 §22~§27 의 하이브리드 측정이
전부 실배치와 다른 러너에서 이뤄졌다. 어텐션 축(ngram)은 어차피 V1 폴백이라
영향이 없다.

## 28-2. 기준선이 7% 올라간다

27B DFlash2 사슬, cudagraph, 마운트 없는 이미지:

| | tok/s | 토큰 |
|---|---|---|
| V1 (우리 마운트) | 78.9 | 85 |
| V1 (마운트 없음) | 79.3 | 85 |
| **V2** | **84.9** | **98** |

우리 마운트는 결백하다 (78.9 대 79.3, 출력·스텝 동일). 러너만 바꿔 **+7%**.
그러면 §27-2 의 비교가 이렇게 바뀐다:

    문서에 적힌 것:  DFlash2 사슬(V1) 78.9  대  DDTree(V1) 50.7  -> -35.8%
    실배치 기준:     DFlash2 사슬(V2) 84.9  대  DDTree(V1) 50.7  -> -40.3%

DDTree 는 V2 에서 아예 못 돈다(훅이 없다). 따라서 지금 숫자가 답하는 질문은
**"V2 를 포기하고 V1+DDTree 를 쓰면 얼마나 손해인가"** 이고, 답은 -40% 다.

## 28-3. 🟢 §27-3 의 무손실 결론이 강화된다

V2 의 출력(98토큰)이 **V1+triton GDN, DDTree 폭8, DDTree 폭17 과 모두 일치**한다.
홀로 다른 것은 여전히 V1 의 fused GDN MTP 커널(85토큰) 하나다.

    V2 == V1+triton     True
    V2 == DDTree(V1)    True
    V2 == V1 CUDA fused False

**DDTree 는 무손실이 맞고, 어긋난 것은 V1 의 그 커널이다.** 운영은 V2 를 쓰므로
이 문제를 겪지 않는다.

## 28-4. 포팅을 해도 이길 자리에 못 놓인다

| | V2 지원 | 오늘 결과 |
|---|---|---|
| dflash (하이브리드) | 지원 | GDN 커널로 -36~40%, 구조적 사망 |
| ngram (순수 어텐션) | **미지원** | DDTree 가 동률까지 갔던 유일한 축 |

V2 포팅은 DDTree 를 **못 이기는 축에만** 올려놓고, 가능성 있던 축은 **아예 못
올린다.** 그래서 "V2 포팅" 만으로는 실험의 단위가 안 된다 —
**V2 + 순수 어텐션 타깃 + 모델 드래프터(eagle3/mtp)** 셋을 함께 바꿔야
오늘 찾은 두 병목(GDN 커널, ngram 커버리지 72% 불발)이 동시에 사라진다.

# 29. EAGLE3 — 트리가 값을 할 구조가 처음으로 잡혔다

§28-4 의 결론은 "V2 포팅만으로는 실험이 안 된다, **V2 + 순수 어텐션 + 모델
드래프터** 셋을 같이 바꿔야 한다" 였다. 그 조합을 실제로 만들어 확인했다.

구성: Qwen3-8B + `AngelSlim/Qwen3-8B_eagle3` (`LlamaForCausalLMEagle3`, 1층,
draft_vocab 32000), V2 러너(`runner_module: vllm.v1.worker.gpu.model_runner` 로
확인), cudagraph, 프롬프트 4개 합계초.

## 29-1. 스펙 디코딩이 값을 한다

| | 합계 | 스텝합 | 스텝당 토큰 | ms/스텝 |
|---|---|---|---|---|
| base | 9.261s | 388 | 0.99 | 23.9 |
| ngram 사슬 (V1) | **5.926s** | 223 | 1.72 | 26.6 |
| EAGLE3 사슬 (V2, 예산 7) | 6.695s | 180 | 2.13 | 37.2 |

EAGLE3 는 base 대비 -27.7%. 다만 **ngram 사슬보다는 느리다** — 스텝이 180 대
223 으로 적은데 단가가 37.2 대 26.6 ms 다. 매 스텝 드래프트 모델을 돌리는 값이다.

## 29-2. 🟢 예산 5 에서 수용이 포화한다

| 예산 | 합계 | 스텝합 | 스텝당 토큰 | ms/스텝 |
|---|---|---|---|---|
| 3 | **5.761s** | 192 | 2.00 | 30.0 |
| 5 | 5.987s | 178 | **2.16** | 33.6 |
| 7 | 6.695s | 180 | 2.13 | 37.2 |
| 10 | 7.740s | 180 | 2.13 | 43.0 |

**예산 5, 7, 10 의 스텝합이 178~180 으로 같다.** 예산을 두 배로 늘려도 한 토큰도
더 못 번다. 반면 단가는 33.6 -> 43.0 ms 로 선형으로 오른다.

**예산 5 를 넘는 몫을 깊이에 쓰면 100% 낭비다.** 그 몫을 폭(가지)으로 돌리는 것이
정확히 DDTree 가 하는 일이다.

지금까지 쟀던 드래프터와 대조:

| 드래프터 | 커버리지 | 성질 | 트리 여지 |
|---|---|---|---|
| ngram (8B) | 28% | 맞을 땐 길게 맞음 | 발동 자체가 안 됨 (§27-1) |
| DFlash2 (27B) | 100% | 폭 8 에서 3.54 수용 | 너무 정확, 가지 안 먹힘 (§16) |
| **EAGLE3 (8B)** | **100%** | **예산 5 에서 포화** | **초과분이 전부 유휴** |

## 29-3. 무손실 2/4 는 완전 동점이다

base 로짓의 자리별 1위-2위 간격을 재고, EAGLE3 가 갈린 자리와 대조했다:

| 프롬프트 | base 와 동일 | 갈린 자리 간격 | 96자리 중 순위 | 중앙값 |
|---|---|---|---|---|
| 파리 | ✗ | **0.0000** | **1위** | 7.000 |
| 소수 | ✗ | **0.0000** | **1위** | 10.500 |
| 반복 | ✓ | (완전 동점 없음, 최소 0.875) | — | — |
| 코드 | ✓ | (완전 동점 없음, 최소 0.125) | — | — |

**완전 동점이 있는 프롬프트에서만, 그 자리에서 갈린다.** 동점이 없는 프롬프트는
바이트 단위로 같다. 구현 문제가 아니다 (§26-4 와 같은 성질).

## 29-4. 그래서 포팅한다 — 그리고 남는 가정

    목표선: EAGLE3 최적(예산 3) 5.761s, ngram 사슬 5.926s
    산술:   예산 5(단가 33.6ms, 수용 2.16)에서 수용이 2.8 로만 오르면
            스텝 178 -> 137, 합계 약 4.6s -> 둘 다 넘어선다

🔴 **"남는 예산을 가지에 쓰면 수용이 오른다" 는 아직 가정이다.** 포화가
드래프터가 그 지점에서 **확신을 잃어서**라면 가지가 먹지만, 타깃이 드래프터와
**근본적으로 다른 방향**으로 가서라면 가지를 쳐도 전부 빗나간다. 지금 데이터로는
못 가른다 — DDTree 를 올려야 안다. 그게 포팅의 값어치이자 리스크다.

# 30. V2 포팅 1단계 — 훅 1·2·3·5

§29 로 트리 조건이 성립함을 확인한 뒤 포팅을 시작했다. 1단계는 **러너 배선**
(런타임 생성, 스텝 경계, 깊이 RoPE, 드래프터 가드)이고, 트리를 실제로 만드는
훅 6 과 수용하는 훅 4 는 2단계다.

## 30-1. 훅 대응

| 훅 | V1 | V2 | 상태 |
|---|---|---|---|
| 1 런타임 생성·공급자 등록 | `__init__` 617 | `__init__`, speculator 직전 | 적용 |
| 2 `begin_step` + KV 그룹 | `_prepare_inputs` 2239 | `prepare_attn` 직후, 마스크 생성 직전 | 적용 |
| 3 깊이 RoPE | forward 준비 3723 | `model_inputs` 조립 직전 | 적용 |
| 3′ 원복 | `_sample` 3777 | forward 직후 | 적용 |
| 5 드래프터 가드 | 5136 | `use_workspace_lane` 에 컨텍스트 매니저 | 적용 (2곳) |
| 4 수용 + KV 압축 | `_sample` 3777 | `sample()` — `batch_sharder` 주의 | **2단계** |
| 6 트리 제안 | 5438 | speculator | **2단계** |

바뀐 것:

  - **`use_ngram_drafter` 를 무조건 False** 로 둔다. V2 는 ngram 을 지원하지
    않으므로(§28) 그 경로가 존재할 수 없다.
  - **KV 그룹 표현이 다르다.** V1 은 `block_table` 객체 목록이었는데 V2 는
    `slot_mappings[그룹, 토큰]` 텐서 + `block_tables.kernel_block_sizes` 다.
  - **드래프터 가드를 컨텍스트 매니저로.** 플래그를 켜고 안 끄면 마스크가
    영구히 죽는다. `with use_workspace_lane(...), _ddt_drafter(self):` 로
    범위를 정확히 잡았다.

## 30-2. 🔴 FULL cudagraph 가 §26 을 더 확실하게 만든다

    if batch_desc.cg_mode == CUDAGraphMode.FULL:
        # 입력 텐서를 넘기지 않는다. 이미 캡처 버퍼에 복사돼 있다.
        model_output = self.cudagraph_manager.run_fullgraph(batch_desc)

FULL 모드는 입력을 **아예 안 넘긴다.** 새 텐서로 지역 변수를 바꿔치기하는 V1
방식이었다면 100% 무시됐을 것이다. 정적 버퍼 제자리 쓰기 + forward 뒤 원복이
V2 에서는 선택이 아니라 필수다.

## 30-3. 패치는 스크립트로 — 컨테이너 리비전에 적용해야 한다

🔴 저장소의 `gpu/model_runner.py` 를 그대로 얹으면 죽는다:

    ImportError: cannot import name 'DPSyncState' from 'vllm.v1.worker.gpu.dp_utils'

컨테이너의 vLLM 리비전과 저장소 리비전이 다르다 (63줄 차이). **컨테이너에서
뽑은 원본에 패치를 얹어야 한다.** `ddtree-dev/patch_v2_runner.py` 가 그걸 한다 —
앵커가 하나라도 없으면 조용히 넘어가지 않고 실패한다. 반쪽만 적용된 러너로
측정하는 것이 최악이라서다.

## 30-4. 회귀 검증 — DDTree 를 끄면 완전한 무연산

Qwen3-8B + EAGLE3 예산 5, V2, cudagraph:

| | 합계 | 스텝합 | 출력 |
|---|---|---|---|
| 패치 전 (`e3_k5`) | 5.987s | 178 | — |
| 패치 후 (`p1_off`) | **5.977s** | **178** | **4/4 바이트 동일** |

## 30-5. 2단계의 미지수

훅 6 은 **EAGLE3 speculator 가 후보를 어떻게 내는가**에 달려 있다. §28 에서 본
조건부 격자(`[스텝][직전선택][top_k]`)는 **dflash2** speculator 의 것이고,
EAGLE3 는 다른 구현(`spec_decode/eagle/`)이다. 세 갈래다:

  - 같은 격자가 있으면 -> dflash2 와 같은 방식으로 트리 확장
  - 스텝별 단일 분포만 있으면 -> 조건부성이 없어 트리 품질이 떨어진다
  - **자기회귀로 한 토큰씩만 내면 -> 트리를 만들려고 드래프터를 여러 번 돌려야
    하고, 비용 구조가 무너진다** (계획 재수립)

미확인.

# 31. V2 포팅 2단계 — 훅 4·6, 그리고 마스크 짝짓기 의심

## 31-1. 훅 6 (트리 제안) — 드래프터를 안 건드려도 된다

§30-5 에서 세 갈래로 나눴던 미지수의 답: **EAGLE3 는 자기회귀**
(`for step in range(1, num_speculative_steps)`, 깊이마다 드래프터 1회, 요청당
토큰 1개)라 dflash2 의 조건부 격자가 없다. 그런데 드래프터를 다시 쓸 필요는
없었다 — V2 의 `BaseSpeculator` 가 이미 버퍼를 갖고 있다:

    self.draft_logits = torch.full(
        (max_num_reqs, num_speculative_steps, vocab_size), ...)

**V1 에서 DDTree 가 소비하던 `[요청, 깊이, 어휘]` 와 같은 모양**이다.
`propose()` 뒤에 읽어 `propose_from_drafter_logits()` 에 그대로 넘긴다.

  - `draft_sample_method="probabilistic"` 이어야 할당된다. 그리디면
    `_greedy_sample_draft` 로 빠져 `None` 이다 — 그 경우 **명시적으로 죽인다.**
  - EAGLE3 의 `draft_vocab_size=32000` 은 문제가 안 된다. `compute_logits` 가
    `draft_id_to_target_id` 로 이미 타깃 어휘(151936)로 확장해서 돌려준다.

🔴 **한계**: 이렇게 만든 가지는 조상 선택에 조건부가 아니다. 자기회귀
   드래프터의 깊이 d logits 는 "사슬이 고른 <d 토큰들"에 조건부인데, 2순위로
   가지를 치면 그 가지의 후손 분포가 아니다. V1 의 DFlash 도 하던 근사지만,
   dflash2 의 조건부 격자(§28)보다는 거칠다.

## 31-2. 훅 4 (트리 수용) — V2 는 회계까지 요구한다

V1 은 `SamplerOutput(sampled_token_ids, logprobs_tensors)` 만 돌려주면 됐는데
V2 는 `num_sampled` / `num_rejected` 도 요구한다. 방출 수에서 역산했다:

    방출 = (_tok >= 0).sum()
    기각 = 스케줄된_드래프트수 - (방출 - 1)

또 `accept()` 이 기대하는 V1 `SpecDecodeMetadata` 를 어댑터로 만들어야 했다:

| V1 | V2 |
|---|---|
| `cu_num_sampled_tokens` | `cu_num_logits_np[1:]` |
| `cu_num_draft_tokens` | `cumsum(num_draft_tokens_per_req)` |
| `num_draft_tokens` | `num_draft_tokens_per_req` |

numpy 로 만든다 — 텐서로 하면 `.tolist()` 가 GPU 동기화를 일으킨다 (§20-4).

## 31-3. 첫 완결 실행 — 배선은 살고 정확성은 죽었다

Qwen3-8B + EAGLE3, V2, cudagraph, 예산 16 / 드래프터 깊이 5:

| | 값 |
|---|---|
| 트리 생성 | 324/339 스텝 (96%) |
| 마스크 발동 | 327 |
| `rope_skipped` | 0 |
| 스텝합 | **157** (사슬 178) |
| **출력** | **0/4** |
| 수용/트리스텝 | **1.61** (사슬 2.16) |
| `compact_unsafe` | 61 |

노드를 16.1개 걸어놓고 1.61 만 먹는다. 트리가 제대로 검증되지 않는다.

## 31-4. 이분법으로 범위를 잘랐다

`VLLM_DDTREE_NOACCEPT=1` 은 트리·마스크·RoPE·압축을 다 켠 채 **루트 logits 만**
방출한다. 배선이 옳다면 결과는 반드시 base greedy 와 같아야 한다.

| 구성 | base 와 일치 | 갈린 자리의 1위-2위 간격 |
|---|---|---|
| NOACCEPT + NOMASK | 2/4 | **0.0000** — 완전 동점 |
| NOACCEPT | 1/4 | **0.1250** — 2 ulp, 96자리 중 3번째로 좁음 |

🔴 **처음엔 이걸 "문장 초반부터 의미가 갈린다 -> 컨텍스트/KV 오염" 으로 읽고
   마스크 짝짓기를 범인으로 지목했다. 틀렸다.** 텍스트만 보고 로짓 간격을
   확인하지 않았다. 같은 실수를 §26-4·§29-3 에서 세 번이나 피해놓고 여기서
   건너뛰었다.

갈린 자리는 둘 다 **동점/근접 동점**이다. 마스크를 켜면 트리 노드의 어텐션이
달라져 로짓 마지막 비트가 바뀌고, 그래서 2 ulp 자리가 하나 더 뒤집힌 것뿐이다.

## 31-5. 확인된 것 — 배선은 정상이다

세그먼트 짝짓기를 직접 찍었다 (`VLLM_DDTREE_DEBUG`):

    [mask] segs=[(17,35),(17,35),(17,35),(17,35)] trees=[(0,17),(1,17),(2,17),(3,17)] paired=4
    [mask] segs=[(17,36),(1,20)]                  trees=[(0,17)]                      paired=1
    [mask] segs=[(17,38)]                         trees=[(0,17)]                      paired=1

세그먼트와 트리가 1:1 로 맞는다. V2 의 패딩이 여분 세그먼트를 만들지 않는다.

스텝 회계도 정상이다 — NOACCEPT 에서 `kv_len` 증가폭이 **113회 전부 +1** 이다
(방출이 스텝당 1토큰이므로 정확하다). 처음에 "35 -> 36 -> 38 로 +2 가 있다" 고
읽었는데 그건 프롬프트 경계(38 -> 22)를 잘못 본 것이었다.

정리하면 **훅 1·2·3·5·6, 마스크, 깊이 RoPE, KV 압축, 롤백 회계가 전부 정상**이다.

## 31-6. 🔴 성능 수치를 인용하면 안 되는 상태다

한때 "DDTree 157스텝 대 사슬 178스텝, 스텝당 2.45 대 2.16 (+13%) — §29 의
가정이 실측으로 확인됐다" 고 적었다. **철회한다.** 그 실행은 출력이 0/4 로
틀린 상태였고, 틀린 출력은 다른(짧은) 텍스트를 생성하므로 스텝 수 비교가
성립하지 않는다. §26-3 에서 RoPE 버그가 만든 "+7.3% 승리" 와 **같은 실수**다.
교훈을 적어놓고 같은 절에서 반복했다.

**규칙: 무손실이 깨진 실행의 성능 수치는 읽지 않는다.**

## 31-7. 버그는 '2토큰 이상 방출' 에 있다

| 구성 | base 와 일치 | 갈린 자리의 간격 |
|---|---|---|
| NOACCEPT (루트만) | 1/4 | 0.125 — 동점 |
| **topk=1 (사슬 수용)** | **0/4** | 4.1 / 8.9 / 8.3 — 진짜 버그 |
| topk=64 (트리) | 0/4 | 6.9 / 6.6 / 11.4 |

`topk=1` 은 DDTree 의 제안·마스크·RoPE·수용·압축을 전부 통과하되 모양만
사슬이다. 그것도 깨진다 — **분기는 무죄**다.

무죄 확정: 훅 1·2·3·5·6 배선, 마스크 짝짓기(세그먼트를 직접 찍어 1:1 확인),
깊이 RoPE, KV 압축, 롤백 회계(`num_rejected = query_len - 방출수` 가
`post_update` 커널 규약과 일치함을 코드로 확인).

출력의 성격: 실제 텍스트인데 **퇴화**한다.

    파리: ' Paris. The capital of France is Paris. The capital of France is...'
    소수: ' 2, 222,2, 2222222222222'

## 31-8. 확인한 것과 헛다리

`draft_logits` 는 **정상이다** — `[요청, 깊이, 151936]` bf16, 비영 3.0M/3.04M,
max 15.375, min -inf (EAGLE3 의 draft 어휘 확장에서 매핑 없는 자리).

헛다리 세 개:

  1. **"채워지지 않은 깊이 열이 쓰레기"** — 훅 1 이 이미 `depth_limit` 으로
     트리 깊이를 묶어 그 열은 참조되지 않는다. 수정은 무연산이었다.
  2. **"`draft_logits` 를 배치 앞 n행으로 읽어 빈 슬롯을 본다"** — 요청이
     1~2개라 `idx_mapping` 이 항등이고 같은 행이다. (그래도 `idx_mapping`
     으로 모으는 것이 맞으므로 수정은 남겼다.)
  3. **"마스크 짝짓기가 V2 패딩과 어긋난다"** — 직접 찍어보니 1:1 이다.

🔴 그리고 **표본을 잘못 잡았다.** `VLLM_DDTREE_TRACE=N` 은 처음 N개 트리
   스텝을 잡는데, 하네스가 맨 앞에서 4프롬프트 워밍업을 돌린다. 지금까지 본
   trace 는 전부 워밍업 구간이다 (`lgdbg` 의 `shape=(4,...)` 와 일치).
   본 측정 구간의 trace 를 보려면 워밍업 이후부터 잡아야 한다.

## 31-9. 노드 문맥을 정의로 검증했다

추측이 여섯 번 연속 빗나간 뒤(깊이 열 / 슬롯 인덱싱 / 마스크 짝짓기 /
`num_rejected` / `block_size` / trace 표본) 방법을 바꿨다. `t20_node_context.py`
를 8B·EAGLE3 용으로 옮긴 `t40_node_ctx.py` 로 **불변식을 직접 검증**한다:

> 노드 i 에서 타깃이 뽑은 `sampled[i]` 는, 그 노드의 조상 경로를 **평범한
> (마스크 없는) 시퀀스로 이어붙여** 돌린 결과의 argmax 와 같아야 한다.
> 이게 트리 마스크와 깊이 RoPE 의 정의 그 자체다.

대조군은 **패치도 스펙 디코딩도 없는 기준선 vLLM** 이다. 결과(`topk=1`, 6스텝,
노드 36개):

    루트 불일치  0 / 6스텝
    척추 불일치  5 / 36 노드
    분기 불일치  0        (topk=1 이라 분기 없음)

**루트가 6/6 일치**한다 — 접두사·KV 압축·스텝 전이가 정상이라는 직접 증거다.

## 31-10. 불일치 5건은 전부 '상위 2개 스왑' 이다

| 노드 | 깊이 | 대조군의 1위-2위 간격 | DDTree 토큰의 대조군 순위 |
|---|---|---|---|
| 4 | 4 | 3.875 | **1 (2위)** |
| 1 | 1 | 0.125 | **1** |
| 5 | 5 | 0.750 | **1** |
| 5 | 5 | 0.125 | **1** |
| 3 | 3 | 0.750 | **1** |

**다섯 건 모두 순위 1.** 문맥이 틀렸다면 대조군의 선택이 무작위 순위에
흩어져야 한다. 상위 2개만 뒤바뀌고 그중 넷이 좁은 간격(<=0.75, 중앙값은 약 7)
이라는 것은 **수치 경로 차이**(트리 마스크 + 깊이 RoPE + 배치 검증 대 평범한
시퀀스)의 특징이다.

🔴 다만 이것은 "무해하다" 의 증명이 아니다. 스펙 디코딩에서 노드 하나의
   argmax 가 뒤집히면 수용 판정이 바뀌고 거기서부터 텍스트가 갈린다.
   5/36 = 14% 는 낮지 않아, 출력이 base 와 0/4 로 갈리는 것이 이걸로 설명된다.
   **버그가 아닐 수 있지만, 결과적으로 무손실은 못 지킨다.**

🔴 그리고 V1 선례와 어긋난다. §23 에서 4B 하이브리드 폭 17 의 DDTree 는 사슬과
   **바이트 단위로 일치**했다. 트리 수치가 일치할 수 있다는 뜻이므로, 여기서
   14% 가 뒤집히는 것은 EAGLE3/8B/V2 조합 특유의 무언가다. 정밀도만의 문제라고
   닫을 수 없다.

## 31-11. 남은 두 가지

  1. 깊이 4 의 **간격 3.875** 건. 나머지 넷(<=0.75)과 성격이 다르다. 이것만
     따로 규명해야 "전부 수치" 로 닫힌다.
  2. **V1 선례와의 차이.** 4B 하이브리드에서는 바이트 일치가 됐는데 왜 여기서는
     안 되는가.

## 31-12. 죽은 가설 셋 (무손실 수리 시도)

  - **cudagraph 가 마스크 텐서를 캡처 시점 주소로 고정** (§26 의 RoPE 버그와
    같은 형태를 의심). `mask_provider` 가 매 스텝 새 텐서를 만드는 건 사실이나,
    **eager 에서도 1/4 로 깨진다.** 기각.
  - **기준선 오선택.** §18-5 는 "무손실 판정은 base 가 아니라 사슬 스펙
    디코딩을 기준으로" 라고 했고 §23 의 V1 바이트 일치도 `flat15` 와의 비교였다.
    그런데 DDTree 는 **base 0/4, EAGLE3 사슬 0/4** 로 둘 다 안 맞는다
    (사슬 자체는 base 와 3/4 일치). 기각.
  - **`logits_indices` 대응 오류.** V2 는 `combine_sampled_and_draft_tokens` 로
    `req_states.draft_tokens` 에서 `input_ids` 를 만드는데, 훅 6 이 쓴 트리가
    스케줄러(`set_draft_tokens`)에도 그대로 전달됨을 확인했다. 일치한다. 기각.

## 31-13. 🔴 정정 — 예산 문턱은 없다. 생성 길이다

처음엔 예산 1(3/4)과 예산 5(0/4)만 보고 "버그가 예산 >= 2 에서 켜진다" 고 적었다.
**틀렸다.** 사이를 채우니 평평하다.

| | 32토큰 | 96토큰 |
|---|---|---|
| 예산 1 | 2/4 | — |
| 예산 2 | **2/4** | 1/4 |
| 예산 3 | 2/4 | — |
| 예산 4 | 2/4 | — |
| 예산 5 | **2/4** | 0/4 |

예산 1~5 가 같은 길이에서 **전부 2/4** 이고, **첫 불일치 자리와 간격도 동일**하다
(파리 0.125, 소수 4.125). 예산은 무관하고 **길이가 길수록 더 갈린다** — 근접
동점을 만날 기회가 누적된다.

## 31-14. 그래서 이것은 '버그' 인가

정황은 **불가피한 수치 차이** 쪽이다:

  - 노드 문맥 검증의 불일치 5건이 **전부 상위 2개 스왑**이고 간격이 좁은 자리에
    몰려 있다 (§31-10).
  - 좁은 자리 비율(간격 <=0.25 가 96자리 중 8개, 약 8%)과 불일치율(5/36 = 14%)이
    같은 자릿수다 — **모든 근접 동점이 뒤집힌다**는 뜻.
  - 예산·트리 모양과 무관하게 같은 자리에서 갈린다.

원인은 경로 차이다. 트리 검증은 FlashInfer 의 **`custom_mask` 프리필 경로**를
타고 대조군은 평범한 경로를 탄다. 감산 순서가 달라 1 ulp 수준 차이가 상시
발생하고, 그것만으로 근접 동점이 뒤집힌다. §18-5 가 하이브리드에 대해
"비트 일치는 성립하지 않는다" 고 한 것과 같은 상황이 어텐션에서도 나온다.

🔴 **다만 전부 설명되지는 않는다.** 96토큰 실행에서 '반복'(간격 8.875),
   '코드'(8.250) 처럼 **넓은 간격 자리의 첫 불일치**가 있다. 동점 뒤집힘으로는
   설명 안 된다. 그리고 '소수' 의 4.125 는 32토큰에서도 예산과 무관하게 항상
   나온다. 이 둘이 미해결이다.

## 31-15. 🔴 철회 — '소수 루트 오염' 은 검증기 인공물이었다

소수 프롬프트 단독 검증에서 루트 불일치 4/10, 순위 -1 이 5건 나와
"수치가 아니라 실제 오염" 이라고 적었다. **철회한다.**

검증기(`t40_node_ctx.py`)는 각 스텝의 접두사를 재구성해야 하는데, 그 방법이
둘 다 신뢰할 수 없다:

| 정렬 방식 | 소수 불일치 | 루트 불일치 |
|---|---|---|
| `emitted_before + OFF` (추정) | 11/30 | 4/10 |
| `prefix_len` (기록값) | **16/30** | **8/10** |

**방식을 바꾸면 결과가 바뀐다.** 그런 지표로는 아무것도 판정할 수 없다.

원인은 둘 다 정확한 접두사가 아니라는 것이다:

  - `emitted_before + OFF` 의 OFF 는 `gen[emitted_before+OFF] == sampled[0]` 이
    모든 스텝에서 성립하는 값을 찾는 **추정**이다. 소수처럼 토큰이 세 종류
    (`220=' '`, `18='3'`, `11=','`)뿐이면 **잘못된 정렬도 우연히 통과**한다.
  - `prefix_len` 은 `num_computed` 인데 비동기 스케줄링에서 **낙관적 값**이다
    (스케줄한 만큼 먼저 올리고 거부된 만큼 나중에 되감는다). 실측에서도
    한 스텝씩 밀린다.

파리는 토큰이 다양해 우연한 통과가 안 되므로 **파리 결과(5/36, 전부 상위 2개
스왑)는 유효**하고, **소수 결과는 무효**다.

## 31-16. 압축은 결백하다

같은 실행에서 `compact()` 입출력을 직접 찍었다 (`VLLM_DDTREE_CDBG=1`):

    [compact] paths={0:[0]}     a=[0]     i=[0]     | src=[21]        dst=[21]
    [compact] paths={0:[0,1]}   a=[0,1]   i=[0,1]   | src=[23,24]     dst=[23,24]
    [compact] paths={0:[0,1,2]} a=[0,1,2] i=[0,1,2] | src=[27,28,29]  dst=[27,28,29]

21스텝 전부 `src == dst` — **완전한 무연산**이다 (`topk=1` 사슬이면 수용 경로가
항상 연속이라 당연하다). 슬롯 진행도 긴 구간에서 단조 증가하고, 튀는 지점
(29->37, 47->16)은 블록 경계와 요청 경계다.

## 31-17. 근본 수정이 필요하다

trace 가 **접두사 토큰 ID 자체**를 기록해야 한다. 지금은 개수
(`emitted_before`)와 길이(`prefix_len`)만 남기는데 둘 다 비동기 스케줄링의
낙관적/추정 값이라 사후에 정확한 접두사를 복원할 수 없다. 주석은 이 문제를
알고 `prefix_len` 을 추가했지만 그것도 낙관적이라 해결되지 않았다.

그 전까지 **노드 문맥 검증 결과는 토큰이 다양한 프롬프트에서만 유효**하다.

## 31-18. 현재 상태

**유효한 것**

  - 훅 1·2·3·5·6 배선, 마스크 짝짓기, KV 압축(`src==dst` 무연산 확인),
    롤백 회계, `draft_logits` 버퍼, `block_size` 매핑 — 전부 검증됨
  - DDTree 끄면 완전한 무연산 (5.977 대 5.987s, 출력 4/4 동일)
  - 예산 문턱 없음 (1~5 동일), 변수는 **생성 길이**
  - 파리 노드 검증: 불일치 5/36 이 전부 상위 2개 스왑 -> `custom_mask` 경로의
    1 ulp 수준 차이로 설명됨

**미해결**

  - 무손실이 안 선다 (32토큰 2/4, 96토큰 0~1/4)
  - 그것이 전부 수치 차이인지, 진짜 버그가 섞였는지 **아직 못 가린다** —
    검증기의 접두사 재구성이 신뢰할 수 없기 때문
  - 2단계 패치는 이 때문에 미커밋 (`patch_v2_runner.py` /
    `patch_v2_speculator.py` 로 재현 가능)

# 32. V2 무손실 붕괴의 진짜 원인 — RoPE 기준 위치 (2026-09-01)

## 32-1. 원인: `num_computed_tokens_np` 는 낙관적 상한이다

vLLM 자신이 그렇게 적어놨다.

```python
# vllm/v1/worker/gpu/states.py:61
# Optimistic CPU mirror of num_computed_tokens (upper bound on GPU value).
self.num_computed_tokens_np = np.zeros(self.max_num_reqs, dtype=np.int32)
```

스케줄러는 드래프트 **전량 수용**을 가정해 이 값을 미리 올려두고, 실제 보정은
GPU 에서만 한다 (`post_update` 가 `num_rejected` 로 되감는다). CPU 미러는 안
고친다.

그런데 DDTree 의 `rope_positions()` 는 이걸 트리 노드의 RoPE 기준으로 썼다.

```python
base = int(self.num_computed[idx])          # ← 낙관적 상한
self._rope[s : s + tree.num_nodes] = depths + base
```

러너 자신은 권위값을 쓴다 — `prepare_pos_seq_lens(..., num_computed_tokens.gpu,
positions, seq_lens)`. 즉 **러너가 채운 positions 는 옳은데 DDTree 가 그걸 틀린
값으로 덮어썼다.**

## 32-2. 실측 (8B + EAGLE3 + 예산 5, 96토큰, V2)

```
rope_base_n=227  rope_base_bad=214 (94%)  누적 편차 +887
[pdbg] step=3  np=13 gpu=8   delta=+5
[pdbg] step=5  np=20 gpu=10  delta=+10
[pdbg] step=12 np=14 gpu=10  delta=+4
```

트리 스텝의 94% 에서 **노드 전체가 4~10 위치 앞으로 밀린 채** KV 에 박혔다.
편차가 대체로 일정해서 상대 거리는 근사적으로 보존된다 — 그래서 출력이 완전히
깨지지 않고 "그럴듯한데 조금씩 다른" 텍스트가 나왔다. 오염이 캐시에 누적되므로
**생성 길이에 비례해 발산**한다. 32토큰 2/4, 96토큰 0~1/4 이라는 관측이 정확히
이 모양이다.

## 32-3. 왜 V1 에서는 안 터졌나

V1 은 CPU 값도 되감는다.

```python
# ddtree-dev/mnt_perf/gpu_model_runner.py:1636
correction = optimistic_num_accepted - num_accepted
req_state.num_computed_tokens -= correction
self.input_batch.num_computed_tokens_cpu[cur_req_index] -= correction
```

이 보정이 스텝 끝에 돌기 때문에 다음 `begin_step` 에서 읽는 값은 옳다. V2 는
같은 보정을 GPU 에서만 한다. **V1 에서 V2 로 옮기면서 생긴 회귀**이고, V1
코드를 그대로 옮겼기 때문에 눈에 안 띄었다.

역설적으로 `runtime.py:367` 의 주석은 이 위험을 **이미 알고 있었다** — 마스크
매칭에 `kv_len` 을 쓰지 말라는 근거로 "낙관적 값"을 인용해놨다. 같은 함수 30줄
위에서 그 값을 RoPE 기준으로 쓰고 있었다.

## 32-4. 수정: 기준을 positions 에서 가져온다

```python
self._rope[s : s + tree.num_nodes] = depths_gpu + positions[s]
```

`positions` 는 `num_computed_tokens.gpu` 로 채워진 권위값이고, `positions[s]`
는 그 요청 쿼리의 첫 자리 = 실제 `num_computed` 다. GPU 스칼라라 D2H 동기화도
없다. 훅 3 은 `positions` 를 아직 안 건드린 시점에 호출되므로 (`copy_` 는 그
뒤) 원본을 읽는 게 보장된다.

## 32-5. 검증기도 같은 이유로 틀려 있었다

`t40_node_ctx.py` 는 접두사를 **추정**했다 — 처음엔 `emitted_before + OFF`,
다음엔 기록된 `prefix_len`. 둘 다 틀렸다. `prefix_len` 이 바로 이 낙관적 값을
찍은 것이었다 (12스텝 중 11스텝에서 예산만큼 어긋나 있었다).

근본 수정: DDTree 가 `all_token_ids` / `num_computed_tokens.gpu` 에서 **접두사
토큰 ID 를 그대로** 찍는다 (`set_prefix_probe` -> `_read_prefix`, trace 켤
때만). `prefix_ids[:-1]` 이 KV 접두사, `[-1]` 이 루트 토큰이다 (루트는 직전
스텝이 낸 토큰이라 KV 가 없고 쿼리 맨 앞에 실리므로 `tree.token_ids` 에 없다).

## 32-6. 수정 후 결과

노드 문맥 검증 (8B + EAGLE3 + 예산 5, 96토큰, 스텝당 노드 6개):

| 프롬프트 | 검사 | 불일치 | 성격 |
|---|---|---|---|
| 소수 | 72 | **0** | — |
| 파리 | 72 | 3 | 1위와의 간격 0.0000 / 0.0000 / 0.1250 |

파리의 3건은 전부 **정확한 동점 또는 2 ulp** 다 (bf16 1 ulp = 0.0625). 대조군은
평범한 디코드 커널, DDTree 는 `custom_mask` 프리필 커널이라 리덕션 순서가 다르고,
그 자리는 커널만 바꿔도 뒤집힌다. **구조 오류는 0건.**

§31-15 의 "소수 루트 문맥 오염" 은 이것으로 최종 정리된다 — 원인은 두 개가
겹친 것이었다. (1) RoPE 기준이 실제로 틀려 있었고, (2) 검증기의 접두사 복원도
틀려서 그 진단을 못 했다. 소수가 파리보다 나빠 보였던 건 토큰 종류가 세 개뿐
(`' '`, `'3'`, `','`)이라 틀린 정렬도 우연히 통과했기 때문이다.

## 32-7. 판정 기준도 고쳤다

불일치 **개수**로 판정하면 안 된다. 1위와의 간격이 ulp 수준이면 커널 차이만으로
뒤집히는 자리다. `t40_node_ctx.py` 는 이제 간격이 4 ulp 를 넘는 것만 구조
오류로 센다.

## 32-8. 남은 것

  - V1(`mnt_perf`)도 같은 코드지만 32-3 의 보정 덕에 증상이 없다. 그래도
    기준을 positions 에서 가져오는 게 맞다 — 보정 순서에 의존하지 않는다.
  - 파리의 동점 뒤집힘은 **원리상 제거 불가**다. 트리는 프리필 커널을 타야 하고
    그건 사슬과 리덕션 순서가 다르다. §18-5 가 하이브리드에서 내린 결론과 같다.

# 33. 수정 후 정량 — V2 + 8B + EAGLE3 (2026-09-01)

## 33-1. 무손실: 사슬과 완전히 동일하다

기준은 **스펙 디코딩 없는 그리디**(`base`). 같은 조건에서 vLLM 자신의 EAGLE3
사슬 스펙 디코딩과 DDTree 를 나란히 놓는다 (96토큰, cudagraph, V2).

| 예산 | 사슬(EAGLE3) | 첫 분기 | DDTree | 첫 분기 |
|---|---|---|---|---|
| 3 | 2/4 | 파리@12, 소수@46 | 2/4 | 파리@12, 소수@46 |
| 5 | 3/4 | 파리@12 | 3/4 | 파리@12 |
| 8 | 2/4 | 파리@12, 소수@46 | 2/4 | 파리@12, 소수@46 |

**프롬프트도 위치도 예산 패턴도 전부 같다.** 즉 남은 불일치는 DDTree 의 것이
아니라 **vLLM 스펙 디코딩 전체의 성질**이다 — 드래프트가 붙으면 쿼리 폭이 1
이 아니게 되어 프리필 커널을 타고, bf16 에서 리덕션 순서가 바뀌면 동점 자리가
뒤집힌다. §32-6 의 노드 검증(구조 오류 0건, 전부 간격 0.0000~0.1250)과 정확히
같은 이야기다.

판정 기준을 다시 세운다: **"스펙 디코딩 없는 그리디와 비트 동일"은 사슬도 못
지킨다.** DDTree 의 무손실 기준은 **같은 조건의 사슬 스펙 디코딩과 동일**이고,
그건 지금 **만족한다**.

## 33-2. 속도: base 는 이기고 사슬에는 진다

| 예산 | 구성 | 토큰/스텝 | 96토큰×4 (s) | base 대비 |
|---|---|---|---|---|
| — | base (스펙 없음) | 0.99 | 9.27 | 1.00x |
| 3 | 사슬 | 2.00 | 5.76 | 1.61x |
| 3 | DDTree | 2.00 | 6.29 | 1.47x |
| 5 | 사슬 | 2.16 | 6.02 | 1.54x |
| 5 | DDTree | 2.06 | 6.81 | 1.36x |
| 8 | 사슬 | 2.15 | 7.07 | 1.31x |
| 8 | DDTree | 2.09 | 7.91 | 1.17x |

길이 192 에서도 같다 (DDTree 1.53x / 1.40x / 1.25x).

읽을 것 두 가지.

  1. **토큰/스텝에서 트리가 사슬을 못 이긴다** (2.00 vs 2.00, 2.06 vs 2.16,
     2.09 vs 2.15). 예산을 분기에 쓰나 깊이에 쓰나 수용이 같거나 오히려 낮다.
     §29 에서 EAGLE3 가 예산 5 이상에서 포화(수용 178~180 고정)한다고 봤는데,
     그 남는 예산을 분기로 돌려도 **회수가 안 된다**. 드래프터 1위가 이미
     충분히 맞아서 2·3위 형제가 헛돈다.
  2. **스텝 단가는 트리가 8~12% 비싸다.** 순수 어텐션이라 §27 의 GDN 재앙
     (+44~49%)은 없지만 0 도 아니다.

두 개가 겹쳐 DDTree 가 사슬에 8~12% 진다.

## 33-3. 그래서 무엇이 남았나

배선·정확성 문제는 **끝났다**. §31 에서 미해결로 남겼던 "수치냐 버그냐" 는
버그였고(§32 의 RoPE 기준), 고친 뒤엔 사슬과 동일하다.

남은 건 **알고리즘 문제 하나**다: 이 드래프터·이 모델에서 분기가 깊이보다
나은 구간이 안 나온다. §29 의 포화는 "분기가 값을 할 조건"으로 읽었는데,
실측은 반대다 — 포화한 예산을 분기에 써도 수용이 안 오른다.

다음에 볼 것은 **드래프터 1위 정확도가 낮은 구간**이다. 온도 > 0, 또는 1위
확률이 낮은 자리에만 분기를 여는 적응형 폭. 지금처럼 모든 자리에 균일하게
분기를 여는 건 예산 낭비다.

## 33-4. V1 회귀 검사

같은 수정을 V1(포크 본체)에도 넣고 하이브리드 + ngram + 예산 8 + 96토큰으로
수정 전후를 돌렸다. **출력 바이트 동일** (무손실 1/4, 분기 위치 파리@63 /
소수@21 / 코드@6 모두 같음). 32-3 대로 V1 에서는 무연산이다 — 그래도 보정
순서에 의존하지 않게 되므로 함께 커밋했다 (`756c42ad82`).

그 1/4 은 이 조합의 **기존 상태**이고 이번 수정과 무관하다 (§18-5: 하이브리드는
비트 동일이 원리상 안 된다).

# 34. 적응형 폭 — 구현하고, 측정하고, 기각했다 (2026-09-01)

## 34-1. 왜 임계값을 감으로 정하면 안 되는가

기대 수용 길이 = Σ_v P(v 가 수용 경로 위에 있음). 수용된 노드는 경로 하나를
이루므로 지시함수의 합이 곧 길이다 — **독립 가정이 필요 없다**. best-first 는
매번 가중치 최대 노드를 넣는데, 부모가 이미 들어 있어야 자식이 후보가 되므로
그리디가 곧 최적이다. 사슬은 가능한 트리 하나일 뿐이다.

따라서 **드래프터 확률이 맞다면 트리는 사슬에 질 수 없다.** 그런데 진다
(§33-2). 그러면 틀린 건 확률이다. 그래서 임계값을 정하기 전에 **무엇이 얼마나
틀렸는지** 먼저 쟀다.

## 34-2. 편향 없는 측정

형제는 best-first 가 고를 때만 트리에 존재한다 — 그 표본으로 잰 적중률은
선택 편향이 있다. 모양을 강제해 없앴다.

  - `VLLM_DDTREE_DEPTH=1` -> 예산 5 전부가 루트 자식(rank 0..4). 모든 스텝 관측.
  - `VLLM_DDTREE_TOPK=1` -> 깊이 5 순수 사슬. 모든 스텝 관측.

| | 실제 적중 | 말한 값 대비 |
|---|---|---|
| 폭(깊이1) rank 0/1/2/3/4 | 0.530 / 0.112 / 0.096 / 0.033 / 0.023 | 0.76 / 1.03 / 1.99 / 1.44 / 1.80 |
| 깊이(사슬) d1..d5 rank0 | 0.648 / 0.449 / 0.323 / 0.393 / 0.390 | 0.92 / 0.64 / 0.46 / 0.57 / 0.66 |

독립 모델로 계산하면 폭 0.794 / 사슬 1.084. 실측은 폭 0.784 / 사슬 1.234 —
**폭은 정확히 맞고 사슬만 14% 초과한다.** 수용이 뭉쳐서 온다(즉시 실패하거나
지평까지 받아낸다).

## 34-3. best-first 가 실제로 필요한 보정

목적함수가 Σ P(수용) 이므로 그 자체를 재면 된다. C = 실제 P(수용) / 말한 값:

| 깊이 | rank0 | rank1 | rank2+ |
|---|---|---|---|
| 1 | 0.85 | 0.96 | 1.69 |
| 2 | 0.66 | 0.48 | 0.39 |
| 3 | 0.26 | — | — |
| 4 | 0.20 | — | — |
| 5 | **0.06** | — | — |

log C 가 깊이에 거의 선형(기울기 -0.65)이고 rank 에도 선형(+0.3)이다. 그래서
상수 두 개로 보정된다.

## 34-4. 구현

  - **`VLLM_DDTREE_BETA` (β)** — 깊이마다 lp 에 더한다. 이미 있던 손잡이인데
    **이 축으로 튜닝된 적이 없었다.** log C 의 깊이 기울기가 곧 β 다.
  - **`VLLM_DDTREE_RANKB` (δ)** — 새로 넣었다. 형제로 갈 때마다 가중치에 더한다
    (`build_tree` 의 형제 push 두 곳). rank 쪽 보정.
  - **`VLLM_DDTREE_TOPK`** — topk_cap 을 손잡이로 뺐다. 트리 기계는 그대로 두고
    모양만 사슬로 만들 수 있어야 **오버헤드와 모양 선택을 가른다**.
  - 노드별 조건부 lp / rank 를 `Tree` 와 trace 에 기록(보정 측정용).

폭이 고정 값으로 정해지지 않는 것이 핵심이다 — 보정된 가중치끼리 겨루므로
드래프터가 확신하는 자리에서는 깊이가 이기고 헷갈리는 자리에서만 가지가 열린다.
평균 분포로 검산하면 β=-0.30 이 이론 최적 모양
`[(1,0),(2,0),(3,0),(1,1),(4,0)]` 을 정확히 만든다 (기대 1.230 대 사슬 1.133).

## 34-5. 실측 — 예측한 이득이 안 나온다

192토큰×4, 8B + EAGLE3 + V2 + cudagraph:

| 구성 | 토큰/스텝 | 초 |
|---|---|---|
| vLLM 사슬 (예산 5) | **2.213** | **11.70** |
| vLLM 사슬 (예산 8) | **2.239** | 13.52 |
| DDTree 모양=사슬 (topk=1) | 2.213 | 12.76 |
| DDTree 트리 β=0 | 2.116 | 13.27 |
| DDTree 트리 β=-0.2 | 2.133 | 13.22 |
| DDTree 트리 β=-0.3 | 2.145 | 13.12 |
| DDTree 트리 β=-0.45 | 2.076 | 13.55 |
| DDTree 트리 β=-0.65 | 2.087 | 13.48 |

보정은 **방향은 맞다** (2.116 -> 2.145) 그러나 예측한 +8.6% 가 아니라 +1.4% 고,
사슬 2.213 에 여전히 못 미친다.

예산을 지평 밖으로 늘려 사슬이 남는 예산을 못 쓰게 해도 마찬가지다:

| 구성 (예산 8) | 토큰/스텝 | 초 |
|---|---|---|
| 사슬 (깊이 8) | **2.239** | 13.52 |
| 트리 깊이5 β=0 δ=0 | 2.188 | 13.03 |
| 트리 깊이5 β=-0.3 δ=0 | 2.110 | 13.54 |
| 트리 깊이5 β=-0.3 δ=0.3 | 2.093 | 13.64 |
| 트리 깊이5 β=0 δ=0.3 | 2.098 | 13.63 |

여기서는 보정이 **해가 된다** — 이미 3노드가 강제로 폭에 가 있는데 β 가 더
밀어붙인다.

강제 모양 실험에서 잰 한계 가치가 **혼합 트리로 전이되지 않는다.** 루트 형제의
가치 0.112 는 '항상 넓은' 배치에서 잰 값이고, 실제로 가지를 여는 스텝은
best-first 가 고른 스텝이라 그 자리의 실현 가치가 더 낮다.

## 34-6. 진짜 병목은 오버헤드였다

`VLLM_DDTREE_TOPK=1` 이 이걸 처음으로 깨끗하게 갈랐다. DDTree 모양=사슬은
vLLM 사슬과 **토큰/스텝이 정확히 같다 (2.213 대 2.213)** — 같은 모양, 같은
수용. 그런데 12.76 대 11.70 초, **+9.1%**.

즉 DDTree 가 이기려면 수용에서 **+9% 이상**을 벌어야 하는데, 최선의 보정이
번 것은 +1.4% 이고 그마저 사슬보다 -3% 낮다.

귀속 (예산5, 374스텝, 사슬 모양):

| 구간 | 스텝당 | 초과분(1.01s) 대비 |
|---|---|---|
| propose (트리 구성 + topk D2H) | 0.97 ms | 36% |
| kv_compact | 0.58 ms | 22% |
| rope | 0.10 ms | 4% |
| mask | 0.07 ms | 3% |

`accept` 구간(15.75 ms/스텝)은 **읽으면 안 된다** — `.tolist()` 가 타깃 forward
를 기다리므로 그 대기가 통째로 잡힌다 (코드 주석이 경고하는 그대로). 위 네
구간 합이 초과분의 64% 고, 나머지는 GPU 쪽(`custom_mask` 프리필 경로, 압축
커널 런치)이다.

## 34-7. 결론

**적응형 폭은 구현했고, 기계는 설계대로 동작한다** (평균 분포에서 이론 최적
모양을 정확히 만든다). 그런데 이 드래프터·이 모델에서 **분기는 어떤 예산에서도
깊이를 못 이긴다.** 예산 5 에서도, 지평 밖 예산 8 에서도, 보정을 넣어도.

§29 의 "예산 5 에서 포화하니 남는 걸 폭에 쓰면 된다" 는 최종 기각이다. 192토큰
기준으로는 포화도 완전하지 않다 (사슬 예산 5 -> 8 에서 2.213 -> 2.239).

남은 유일한 공학적 표적은 **+9.1% 오버헤드**다. 사슬 모양일 때는 KV 압축이
무연산이고 마스크도 causal 과 동치인데 둘 다 실행된다 — '퇴화 트리 빠른 경로'
로 대부분 회수할 수 있다. 다만 그건 DDTree 를 **사슬과 같게** 만들 뿐 이기게
하지는 못한다.

이기려면 축을 바꿔야 한다: **드래프터 1위가 자주 틀리는 작업**(온도>0, 창작,
코드 자동완성 같이 분기가 실재하는 입력). 온도 0 짧은 사실질의에서는 EAGLE3
1위가 이미 충분히 맞아서 형제가 놓일 자리가 없다.

## 34-8. 커밋과 회귀 검사

기능 단위로 셋으로 나눴다.

  - `4e80b4d69f` `VLLM_DDTREE_TOPK` — 오버헤드와 모양 선택을 가르는 손잡이
  - `7d04689f95` 노드별 조건부 lp/rank 기록 — 보정 측정의 전제
  - `69302c7cc1` 적응형 폭 (`VLLM_DDTREE_RANKB`)

검사 두 가지.

  - 리포본과 V2 스테이징본(`mnt_v2`)이 같은 모양을 내는가 — 무작위 200 케이스
    (깊이 1~7, topk 1~5, 예산 1~9, β∈{0,-0.3,-0.65}, δ∈{0,0.3,0.6})에서
    parents/token_ids/depths/node_rank 전부 일치.
  - 기본값이 무연산인가 — δ=0 에서 변경 전 코드(`756c42ad82`)와 무작위 300
    케이스 전부 일치. 손잡이를 안 켜면 동작이 그대로다.

# 35. 축을 바꾸니 트리가 처음으로 이긴다 (2026-09-01)

## 35-1. 기본 4종은 드래프터에게 최악의 표본이었다

`t36_attn.py` 의 기본 프롬프트는 사실 조회("The capital of France is"), 짧은
목록("List three prime numbers:"), **문자 그대로의 반복열**("Repeat after me:
alpha beta gamma alpha beta gamma..."), 그리고 피보나치 다음의 팩토리얼이다.
전부 드래프터 1위가 뻔한 자리다. 여기서 "분기가 값을 안 한다"는 결론을 낸 것은
**표본이 결론을 정한 것**이다.

`DDT_SET=hard` 로 개방형 산문·설명·비정형 코드·고유명사·번역·자유 나열 6종을
넣었다. 온도는 0 그대로다 — 타깃은 결정적이되 드래프터가 못 맞히는 자리를
만드는 게 목적이다.

축 변수부터 확인했다 (`VLLM_DDTREE_TOPK=1` 사슬로 편향 없이 측정):

| 깊이 | easy 적중 | hard 적중 |
|---|---|---|
| 1 | 0.648 | **0.331** |
| 2 | 0.449 | 0.249 |
| 3 | 0.323 | 0.203 |
| 4 | 0.393 | 0.179 |
| 5 | 0.390 | 0.163 |

드래프터 1위 적중률이 65% -> 33% 로 떨어졌고 사슬 수용도 2.213 -> 1.506 으로
무너졌다. 의도한 구간이다.

## 35-2. 결과 — 수용에서 트리가 이긴다

192토큰×6, 8B + EAGLE3 + V2 + cudagraph:

| 구성 | 토큰/스텝 | 초 | 사슬 대비 |
|---|---|---|---|
| base (스펙 없음) | 0.995 | 27.86 | 0.93x |
| **vLLM 사슬 예산5** | **1.506** | **25.89** | 1.00x |
| vLLM 사슬 예산8 | 1.510 | 30.17 | 0.86x |
| DDTree 모양=사슬 | 1.506 | 28.13 | 0.92x |
| DDTree 트리 β=0 | 1.580 | 27.00 | 0.96x |
| DDTree 트리 β=-0.3 | 1.625 | 26.16 | 0.99x |
| DDTree 트리 β=-0.55 | 1.609 | 26.31 | 0.98x |
| DDTree 트리 β=-0.3 δ=0.3 | 1.618 | 26.15 | 0.99x |
| **DDTree 예산8/깊이5 β=-0.3** | **1.662** | **25.86** | **1.00x** |

**수용에서 최대 +10.4%** (1.662 대 1.506). §34 에서 easy 로는 어떤 설정도 사슬을
못 넘었던 것과 정반대다. 그리고 §34 에서 만든 보정(β)이 여기서는 실제로 먹는다
(1.580 -> 1.625, +2.8%).

벽시계는 25.86 대 25.89 로 **무승부**다. 산수가 정확히 맞는다: 기계값
오버헤드가 이 표본에서 **+8.6%** 이고(모양=사슬 28.13 대 사슬 25.89, 토큰/스텝은
1.506 으로 동일), 수용 이득 +10.4% 가 그걸 상쇄한다.

## 35-3. 폭은 드래프터에게 공짜다

예산 8 두 팔을 비교하면 구조적 이점이 하나 더 보인다.

  - 사슬 예산 8: 드래프터 forward **8회**, 검증 8토큰 -> 1.510 tok/step, 30.17s
  - 트리 예산 8 / 깊이 5: 드래프터 forward **5회**, 검증 8토큰 -> 1.662, 25.86s

깊이를 하나 늘리면 드래프터 forward 가 하나 더 붙는다. **폭을 하나 늘리면 이미
가진 logits 에서 topk 를 하나 더 뽑을 뿐이다.** 사슬은 예산 5->8 에서 4.28초를
더 쓰고 수용은 1.506->1.510 으로 사실상 그대로다. 트리는 같은 예산을 폭에 써서
수용을 10% 올리면서 시간은 오히려 줄인다.

이건 §29 의 "포화" 를 제대로 쓰는 방법이기도 하다. 드래프터가 깊이에서 포화하면
남는 예산을 **드래프터를 더 돌리지 않고** 폭으로 돌릴 수 있다 — 그게 트리가
사슬로 흉내낼 수 없는 유일한 것이다.

## 35-4. 확인해둔 것

`draft_sample_method: probabilistic` 은 DDTree 가 logits 를 얻으려고 켜는데,
드래프터가 자기 사슬을 샘플링으로 이어가면 사슬 기준선(그리디)과 불공정해진다.
확인 결과 교란이 아니다 — `gumbel_noised_argmax` 는 온도 0 에서 순수 argmax 다
(`"Argmax of logits under Gumbel-max sampling, or plain argmax at temp 0."`).
요청 온도가 0 이므로 드래프트는 그리디와 동일하고 logits 만 추가로 저장된다.

무손실은 이 표본에서 base 대비 사슬 0/6, DDTree 1/6~3/6 이다. §33-1 의 기준대로
**사슬과 같은 급**이고, 오히려 DDTree 쪽이 조금 낫다.

## 35-5. 깊이를 줄이고 폭으로 돌리는 것이 옳다

깊이는 드래프터 forward 를 하나씩 먹고, 폭은 안 먹는다. 그래서 드래프터 깊이를
3으로 낮추고 남는 예산을 전부 폭에 쓰는 게 최적이다 (192토큰×6, β=-0.3):

| 구성 | 드래프터 fwd | 토큰/스텝 | 최저(s) | 사슬3 대비 |
|---|---|---|---|---|
| 사슬 예산3 (3회 반복) | 3 | 1.498 | 23.13 | 1.000x |
| 트리 깊이3 예산8 | 3 | 1.653 | 23.54 | 0.983x |
| 트리 깊이3 예산12 (3회) | 3 | 1.697 | 23.08 | 1.002x |
| **트리 깊이3 예산16 (3회)** | 3 | **1.761** | 23.16 | 0.999x |
| 트리 깊이3 예산20 | 3 | 1.714 | 23.96 | 0.965x |
| 트리 깊이3 예산24 | 3 | 1.689 | 23.95 | 0.966x |
| 트리 깊이5 예산16 | 5 | 1.751 | 25.67 | 0.901x |
| 사슬 예산8 | 8 | 1.510 | 30.17 | 0.767x |

**수용 +17.6%** (1.761 대 1.498) 인데 **벽시계는 정확히 동률**이다 (23.16 대
23.13, 3회 반복 최저). 예산 16 을 넘기면 수용이 되레 떨어진다 — 깊이 3 에서
best-first 가 쓸 후보를 소진하면 `pad_to_budget` 이 쓰레기 노드로 채운다.

## 35-6. 남은 것은 전부 기계값이다 — 정확히 얼마인지 쟀다

`VLLM_DDTREE_TOPK=1` + 깊이 3 으로 드래프터 비용을 3회에 고정하고 검증 폭만
키우면, 기계값과 검증 폭 값을 가를 수 있다.

| 검증 폭 | ms/스텝 | 증분 |
|---|---|---|
| 5 | 33.28 | — |
| 9 | 33.71 | +0.44 |
| 13 | 33.84 | +0.12 |
| 17 | 35.44 | +1.60 |

즉 **검증 토큰 12개를 더 붙이는 값이 2.13 ms 뿐**이다 (토큰당 약 0.18 ms).
배치 1 의 타깃 forward 는 메모리 바운드라 폭이 거의 공짜다.

기준선과 맞대면:

```
사슬        (검증폭 4,  드래프터 3회)  30.08 ms/스텝
DDTree     (검증폭 5,  드래프터 3회)  33.28  -> 기계값 +3.20 ms (+10.6%)
DDTree     (검증폭 17, 드래프터 3회)  35.41  -> 폭 12토큰 +2.13 ms
```

**+5.3 ms 격차의 60% 가 기계값이고 40% 만 진짜 검증 폭이다.**

토큰당 시간으로 환산하면:

| | ms/토큰 |
|---|---|
| 사슬 | 20.08 |
| 트리 (현재) | 20.11 — 동률 |
| 트리 (기계값 0 가정) | **18.29 — 8.9% 빠름** |

**즉 +3.2 ms 기계값이 DDTree 와 약 9% 승리 사이에 정확히 놓여 있다.** §34-6 의
귀속(propose 0.97 ms, kv_compact 0.58 ms, 나머지는 GPU 쪽 `custom_mask` 프리필
경로)이 그대로 표적이 된다. 이건 알고리즘 도박이 아니라 공학 항목이다.

## 35-7. 정리

  - **§34 의 "분기는 값을 안 한다" 는 easy 표본에 한정된 결론이었다.** 표본을
    바꾸니 수용에서 트리가 사슬을 최대 +17.6% 이긴다.
  - §34 에서 만든 보정(β)이 여기서는 실제로 먹는다 (1.580 -> 1.625).
  - **폭은 드래프터에게 공짜, 타깃에게도 거의 공짜**(토큰당 0.18 ms)다. 깊이만
    비싸다. 최적은 얕은 드래프터 + 넓은 트리 (깊이 3, 예산 16).
  - 지금 막고 있는 건 기계값 +3.2 ms/스텝 하나다. 없애면 약 9% 이긴다.
  - 예산 상한이 있다 — 깊이 3 에서 16 을 넘으면 `pad_to_budget` 쓰레기가 들어와
    수용이 떨어진다. 예산을 늘리려면 깊이도 같이 늘려야 한다.

# 36. 퇴화 빠른 경로 — 구현했고, 그리고 더 큰 것이 보였다 (2026-09-01)

## 36-1. 기계값 3.2 ms 의 내역

구간을 통째로 빼서 귀속했다 (hard 표본, 깊이3/예산16. 제거 팔은 무손실이 깨지므로
ms/스텝만 유효하다):

| 구성 | ms/스텝 | 절감 |
|---|---|---|
| 전체 | 35.29 | — |
| 압축 제거 | 34.60 | 0.69 |
| 마스크 제거 | 33.82 | 1.47 |

즉 마스크 1.47 / 압축 0.69 / 나머지 약 1.0 (propose 의 topk D2H 등).

## 36-2. 구현한 것 — 압축 퇴화 경로

수용 경로가 `[0,1,2,...]` 이면(척추를 그대로 받아낸 경우) `src == dst` 라 압축은
완전한 무연산이다. 그런데도 인덱스 H2D 3회, 그룹별 slot gather, 안전성 검사의
**D2H 동기화**, 커널 런치까지 전부 돌고 있었다. 판정은 공짜다 — 수용 경로는
이미 CPU 에 있다.

| 구성 | ms/스텝 | 최저(s) | 건너뛴 스텝 | 출력 |
|---|---|---|---|---|
| 빠른경로 off | 35.53 | 23.24 | 0/1965 | 기준 |
| **빠른경로 on** | **35.14** | **22.98** | **1161/1965 (59%)** | 바이트 동일 |
| on + torch 압축 | 35.51 | 23.22 | 1161/1965 | 동일 |

절감 0.39 ms = 0.59 × 0.69 ms — 산수가 정확히 맞는다. 커밋 `13f42b629b`.

torch 구현은 순서 전제가 없어 안전성 검사(D2H)를 아예 뺄 수 있는데, 실측에서는
커널이 더 느려 순손해였다. 기본값은 triton + 검사 그대로 둔다.

## 36-3. 🔴 §35-6 의 "기계값 0 이면 8.9% 승리" 는 과했다

그 계산은 **3.2 ms 를 전부 없앨 수 있다는 가정**이었다. 실제로 없앨 수 있었던
건 0.39 ms 뿐이다.

  - 마스크 1.47 ms 는 FlashInfer 의 `custom_mask` 커널 자체다. 트리 의미를
    유지하는 한 우리 쪽에서 못 없앤다.
  - 압축 0.69 ms 중 59% 만 무연산이다. 나머지 41% 는 진짜로 옮겨야 한다.
  - propose 약 1.0 ms 는 topk D2H 동기화가 대부분이고, 드래프터 forward 를
    기다리는 것이라 구조적이다.

현재 위치(hard 표본, 깊이3/예산16 대 사슬 예산3):

| 문맥 | 사슬 | 트리 | |
|---|---|---|---|
| maxtok 96 | 11.67s | 11.58s | 트리 **+0.8%** |
| maxtok 384 | 45.38s | 45.58s | 트리 **-0.5%** |

**사실상 동률**이다.

## 36-4. 더 큰 것: 마스크 값이 문맥 길이에 비례한다

평탄 마스크는 `q × (과거 + q)` 다. 트리 구조가 실제로 필요로 하는 건 마지막
`q × q` 블록뿐인데, **과거 구간 전체를 1로 채워서 넘긴다.** FlashInfer 의
`custom_mask` 커널은 그걸 다 읽는다.

| maxtok | 사슬 ms/스텝 | 트리 ms/스텝 | 격차 |
|---|---|---|---|
| 96 | 29.92 | 34.57 | 4.65 |
| 384 | 30.27 | 35.39 | **5.12** |

문맥이 ~100 -> ~390 늘 때 사슬은 +0.35 ms, 트리는 +0.82 ms 늘어 **격차가
+0.47 ms 벌어진다** (문맥 100토큰당 약 0.16 ms). 외삽하면 문맥 2000 에서 약
7.7 ms, 4000 에서 약 10.9 ms 다. **운영 문맥 길이에서는 이게 지배적이 된다.**

고칠 방법은 있다 — 어텐션을 두 번에 나눈다. (a) 과거 구간은 마스크 없이 평범한
경로로, (b) 트리 구간만 `q × q` 마스크로, 그리고 log-sum-exp 로 병합한다
(FlashInfer 의 `merge_state`, cascade 어텐션이 쓰는 그 방식). 그러면 마스크 값이
`O(q · 문맥)` 에서 `O(q²)` 로 떨어진다.

**이게 DDTree 가 운영에서 성립하는지를 가르는 항목이다.** 지금은 짧은 문맥에서
동률인데, 문맥이 길어질수록 벌어진다.

## 36-5. 정리

  - 퇴화 빠른 경로는 구현했고 동작한다 (-1.1%, 스텝의 59%, 출력 동일).
  - 그걸로 트리는 사슬과 **동률**이 됐다 (짧은 문맥 +0.8%, 긴 문맥 -0.5%).
  - §35-6 의 8.9% 전망은 철회한다 — 없앨 수 있는 건 3.2 ms 중 0.39 ms 였다.
  - 다음 표적은 **마스크를 두 번에 나누기**다. 남은 기계값의 절반이고, 유일하게
    문맥 길이에 비례해 커지는 항목이다.

# 37. 마스크 2분할 — 구현했고, 안 된다 (2026-09-01)

## 37-1. 🔴 먼저: §36-4 의 근거가 틀렸다

§36-4 는 "마스크 값이 문맥 길이에 비례한다"고 결론했다. 근거는 `VLLM_DDTREE_NOMASK=1`
로 마스크를 끄고 잰 차이였다 — 문맥 40 에서 0.65 ms, 740 에서 4.43 ms.

그런데 그 팔의 통계를 보면:

```
마스크 켬: masked=349  rope_skipped=0    accepted=310  c_work=147  kv_rows=353
마스크 끔: masked=0    rope_skipped=606  accepted=0    c_work=0    kv_rows=0
```

**마스크를 끄면 트리 파이프라인 전체가 꺼진다.** 마스크를 못 받은 트리는
`accept()` 가 거부하고(무손실 보호), 그러면 `rope_positions` 도 건너뛰고 압축도
아예 안 돈다. 즉 4.43 ms 는 마스크 + RoPE + 수용 + 압축을 합친 값이었는데 전부
마스크 탓으로 돌렸다.

이 손잡이는 구조상 마스크만 떼어낼 수 없다 — 마스크와 트리 수용은 한 몸이다.
**귀속에 쓰면 안 된다.**

## 37-2. 구현한 것

`BatchDCPPrefillWrapper`(DCP 경로)가 이미 정확히 같은 구조였다 — 페이지드
컨텍스트 + ragged 신규토큰 + `merge_attn_states`. 그걸 본떠
`BatchDDTreeSplitPrefillWrapper` 를 만들었다.

  - `_context`: 페이지드, `causal=False`, **kv 길이에서 현재 쿼리 토큰을 뺀다**
    (안 빼면 현재 토큰이 양쪽에서 두 번 셈해지고, 컨텍스트 쪽은 마스크가 없어서
    형제가 서로를 보게 된다)
  - `_local`: ragged, kv = 현재 토큰 자신, `custom_mask` = 세그먼트별 q×q
  - 병합: `merge_state`

런타임에 `local_mask_provider` 를 추가했다 — 과거 구간 없이 q×q 만 만든다.
`VLLM_DDTREE_SPLIT=1` 로만 켜지고 기본은 꺼져 있다.

**정확성은 확인됐다**: 문맥 700, 6프롬프트 중 5개가 바이트 동일이고 1개가 위치
21 에서 갈린다 — §33-1 의 동점 뒤집힘과 같은 급이다 (병합은 리덕션 순서가 다르다).

## 37-3. 결과 — 모든 문맥에서 진다

| 문맥 | 단일 마스크 | 2분할(merge_state) | 2분할(vllm merge) |
|---|---|---|---|
| 0 | 33.74 | 34.86 | — |
| 700 | **40.99** | 41.66 (+0.67) | 41.99 (+1.00) |
| 1500 | **48.95** | 49.12 (+0.17) | — |
| 3000 | **62.53** | 63.83 (+1.30) | 64.05 (+1.52) |

교차점이 없다. 그리고 **문맥이 길수록 격차가 벌어진다**(+0.67 -> +1.30) — 마스크
값이 문맥에 비례한다면 반대로 좁아져야 한다. 37-1 의 오류가 여기서 확인된다.

원인은 단순하다. 병합 경로는 **어텐션 계층마다** 돈다. 8B 면 36계층이라
계층당 (커널 1개 추가 + 병합)이 그대로 36배가 된다. 실측 +0.7~1.3 ms 는
계층당 2회 추가 런치 × 36 ≈ 72 런치와 정확히 맞는 크기다. 반면 없앤 마스크
값은 사실상 0 이었다.

부수 소득: `merge_state` 가 vLLM 의 `merge_attn_states` 보다 일관되게 빠르다
(0.67 대 1.00, 1.30 대 1.52). FlashInfer 래퍼가 내놓는 LSE 를 **layout 도 log2
도메인도 그대로** 받으므로 계층당 전치 → contiguous 복사 → log2에서 ln 변환이
양쪽에 필요 없다 (계층당 커널 6개). 검증: `merge_state` 를 log2 기준으로 계산한
참조와 맞춰보면 오차 0.0008, 자연로그 기준이면 0.34 — log2 가 맞다.

## 37-4. 정리

  - 2분할은 **동작하지만 값을 안 한다.** 배치 1 · 36계층에서 계층당 고정비가
    지배한다. 기본값은 꺼둔다.
  - §36-4 의 "마스크가 문맥에 비례" 는 **철회**한다. 근거가 된 NOMASK 팔이
    트리 파이프라인 전체를 껐다.
  - `VLLM_DDTREE_NOMASK` 는 귀속에 쓸 수 없는 손잡이다 (마스크와 트리 수용이
    한 몸이라 구조상 분리가 안 된다).
  - 남은 기계값 약 2.8 ms 의 내역은 **아직 정직하게 귀속되지 않았다.** 확실한 건
    압축 0.69 ms 뿐이고(구간 자체가 독립적이라 measurable), 그 중 59% 는 이미
    §36-2 로 없앴다.

# 38. TIMESPLIT 으로 다시 귀속 (2026-09-01)

## 38-1. 방법 — 끄지 말고 잰다

§37-1 에서 확인했듯 DDTree 에서 **구간을 끄는 ablation 은 신뢰할 수 없다**.
무손실 보호 때문에 마스크를 끄면 수용도 RoPE 도 압축도 연쇄로 꺼진다.

`VLLM_DDTREE_TIMESPLIT=1` 은 구간 앞에서 한 번 동기화해 '앞선 GPU 작업 대기'와
'우리 작업'을 가른다. 원래 `propose`/`accept` 두 곳에만 있던 것을 `mask`,
`rope`, `kv_compact` 까지 확장했다. (동기화가 파이프라인을 세우므로 이 모드의
**총 시간은 성능 측정에 쓰면 안 된다** — 구간 비율만 읽는다.)

`custom_mask` 커널 값은 CPU 타이머로는 안 잡힌다. 타깃 forward 안에서 일어나
`a_wait` 에 섞이기 때문이다. 그건 다른 방법으로 갈랐다 — 아래.

## 38-2. custom_mask 커널을 파이프라인을 켠 채로 격리

**사슬 트리의 가시성은 causal 과 정확히 같다.** 그러면 마스크를 줄 이유가 없다.
그래서 `Tree.is_chain` 을 보고 사슬이면 마스크를 건너뛰되, `causal_ok` 로 따로
표시해 수용·RoPE 는 정상으로 돌게 했다 (`VLLM_DDTREE_CHAINMASK=0` 으로 끌 수
있다). 이러면 **파이프라인을 전부 켠 채로 마스크만** 뺄 수 있다.

🔴 마스크 제공자가 causal 마스크를 만들어 넘기면 안 된다 — 값이 같아도
FlashInfer 는 그대로 custom_mask 커널로 간다. 진짜 트리 마스크가 하나도 필요
없으면 **None 을 반환**해야 기본 causal 경로를 탄다.

진짜 사슬(예산 5 = 깊이 5, topk=1)에서 셋을 나란히 놓으면 — **수용은 셋 다
1.506 으로 동일**하다:

| 구성 | ms/스텝 | |
|---|---|---|
| vLLM 사슬 | 33.99 | 기준 |
| DDTree 사슬, 마스크 생략 | 35.70 | 기계값 **+1.71** |
| DDTree 사슬, 마스크 유지 | 36.40 | custom_mask 커널 **+0.70** |

출력은 마스크 유지/생략이 **바이트 동일**하다. 순수 기계값은 **2.41 ms**다.

§36-1 이 NOMASK ablation 으로 낸 "마스크 1.47 ms" 는 **0.70 ms** 로 정정한다.

## 38-3. 이기는 설정의 격차 분해

깊이3/예산16 대 사슬 예산3 (드래프터 forward 는 양쪽 3회로 같다):

| | ms/스텝 |
|---|---|
| 총 격차 | 5.13 |
| — 검증 폭 12토큰 추가 | 2.13 (정당한 값 — 트리가 **사려는** 것) |
| — 기계값 | **3.00** |

## 38-4. 기계값 3.00 ms 의 내역

| 항목 | ms/스텝 | 비중 |
|---|---|---|
| propose (topk 커널 + D2H) | 0.664 | 22.1% |
| custom_mask 커널 | 0.700 | 23.3% |
| kv_compact (빠른경로 적용 후) | 0.265 | 8.8% |
| accept (argmax D2H + 트리 walk) | 0.224 | 7.5% |
| rope | 0.044 | 1.5% |
| mask 생성 | 0.018 | 0.6% |
| **귀속 합계** | **1.914** | **63.8%** |
| 미귀속 (기타 GPU 런치·plan 비용) | 1.085 | 36.2% |

`a_wait` 17.65 ms 는 타깃 forward 를 기다린 시간이라 **우리 값이 아니다**.
§34-6 에서 `accept` 를 15.75 ms 로 읽고 경고만 달았던 그 항목이 이걸로 정확히
갈렸다 — 실제 accept 의 값은 **0.224 ms** 다.

## 38-5. 읽을 것

  - 마스크는 생각보다 작다 (0.70 ms, 23%). §36 에서 이걸 1.47 로 보고 2분할에
    투자한 것이 과했다.
  - **가장 큰 항목은 propose 의 topk (0.664 ms, 22%)** 다. 드래프터 logits 에서
    깊이×topk 를 뽑아 CPU 로 내리는 부분이다. 트리를 GPU 에서 만들면 D2H 두
    번이 사라진다 — 지금까지 한 번도 시도 안 한 축이다.
  - 압축은 빠른 경로 덕에 이미 0.265 로 작다 (§36-2 전에는 0.69).
  - 미귀속 1.09 ms 는 plan() 의 `segment_packbits` 와 계층별 추가 런치로
    추정되나 **아직 측정된 값이 아니다.**

## 38-6. 부수 소득 — 사슬 마스크 생략은 그 자체로 이득

사슬 모양일 때 `custom_mask` 를 안 주면 **출력 바이트 동일에 -0.70 ms/스텝
(-1.9%)** 다. 동적 사슬 모드(`VLLM_DDTREE_TAU`)나 드래프터가 확신해 트리가
사슬로 퇴화하는 스텝에서 공짜로 걸린다. 이기는 설정(넓은 트리)에서는 안 걸린다.

# 39. GPU 트리 구성은 표적이 틀렸다 — 진짜는 CPU 디스패치였다 (2026-09-01)

## 39-1. D2H 는 0.056 ms 뿐이다

§38 이 propose 의 topk 를 최대 항목(0.664 ms)으로 짚었고, "트리를 GPU 에서
만들면 D2H 두 번이 사라진다"고 했다. 그 안을 갈라보니 D2H 는 **0.056 ms** 다 —
propose 의 7%, 기계값의 2%. GPU 트리 구성이 없앨 수 있는 건 그것뿐이다.

## 39-2. 🔴 계측이 계속 값을 부풀리고 있었다

같은 연산의 GPU 시간을 세 방법으로:

| 방법 | logsumexp | topk | cast |
|---|---|---|---|
| 격리 마이크로벤치 (진실) | **0.062** | 0.077 | 0.008 |
| 프로덕션, 호출마다 CUDA 이벤트 4개 | 0.295 | 0.141 | 0.039 |
| 프로덕션, 구간마다 `torch.cuda.synchronize()` | 0.327 | 0.164 | 0.052 |

**계측이 3~5배 부풀린다.** 동기화는 파이프라인을 세우고, `torch.cuda.Event`
객체를 호출마다 만들어 기록하는 것도 스트림에 부담을 준다.

따라서 §38 의 구간 표는 **상한**으로만 읽어야 한다. 믿을 수 있는 건 두 가지뿐이다:
**끝에서 끝까지 A/B** 와 **격리 마이크로벤치**.

## 39-3. propose 0.759 ms 의 실제 구성

| | ms/스텝 | 비중 |
|---|---|---|
| GPU 커널 (격리 실측) | 0.147 | 19% |
| D2H | 0.056 | 7% |
| 파이썬 힙 (트리 구성) | 0.098 | 13% |
| p_out | 0.029 | 4% |
| **CPU 디스패치** | **0.429** | **57%** |

**연산 개수가 값이다.** eager PyTorch 에서 디스패치 하나가 30~40 µs 인데,
`float()` -> `topk` -> `logsumexp`(내부에서 max/sub/exp/sum/log 5개) -> 뺄셈 ->
D2H 2회 -> numpy 2회 로 열 개 넘게 던지고 있었다.

## 39-4. 고친 것

`log_softmax` 는 정의가 `logits - logsumexp(logits)` 그 자체다. `dtype` 인자로
캐스트까지 흡수하면 **디스패치 2개**(log_softmax + topk)로 끝난다.

```python
lsm = torch.log_softmax(rows, dim=-1, dtype=torch.float32)
return torch.topk(lsm, k=k_all, dim=-1)
```

D2H 도 하나로 합쳤다. `.to("cpu")` 는 매번 동기화라 두 번이면 두 번 선다.
vocab < 2^24 면 토큰 id 를 float32 로 **정확히** 실을 수 있어 (float32 는 2^24
까지 정수를 손실 없이 표현한다) 한 번에 보낸다. 넘으면 예전처럼 따로 보낸다.

| | 기존 | 새 경로 |
|---|---|---|
| p_topk | 0.511 | **0.276** (-46%) |
| propose − p_wait | 0.664 | **0.427** |
| 전체 ms/스텝 | 35.13 | **35.00** |
| 전체 초 (3회 최저) | 22.98 | **22.89** (-0.38%) |

**출력은 바이트 동일**이다 (수학적으로 같은 식이다).

구간에서 −0.235 ms 를 아꼈는데 끝에서 끝까지는 −0.132 ms 다. TIMESPLIT 없이는
그 CPU 작업의 일부가 GPU 작업과 겹치기 때문이다 — §39-2 가 말하는 그대로다.

## 39-5. 현재 위치

hard 표본, 깊이3/예산16 대 사슬 예산3 (192토큰×6, 3회 최저):

| | 초 | 토큰/스텝 |
|---|---|---|
| vLLM 사슬 예산3 | 23.13 | 1.498 |
| **DDTree** | **22.89** | **1.761** |

**+1.0% 빠르면서 수용은 +17.6%** 다. §35 에서 동률이었던 것이 압축 빠른 경로
(§36-2, −0.39 ms)와 이번 디스패치 절감(−0.13 ms)으로 앞섰다.

## 39-6. 남은 것

  - GPU 트리 구성은 **안 한다**. 없앨 수 있는 D2H 가 0.056 ms 인데 트리 구조를
    GPU 에 두면 `follow_tree`(수용)와 마스크 생성도 GPU 로 옮겨야 한다 — 훨씬
    큰 작업에 훨씬 작은 보상이다.
  - 같은 논리를 다른 구간에도 적용할 수 있다. accept 0.224 ms 와 kv_compact
    0.265 ms 도 대부분 디스패치일 가능성이 높다. 다만 §39-2 대로 **끝에서 끝까지
    A/B 로만** 확인해야 한다.

# 40. accept·kv_compact 도 같은 식으로 — 그리고 배치 1 의 함정 (2026-09-01)

## 40-1. 줄인 것

  - **accept**: 출력 버퍼를 numpy 로 만든다. 예전에는 `torch.full` 한 번 +
    **요청마다** `torch.tensor` 한 번이었다.
  - **kv_compact**: 인덱스 두 개를 이어붙여 H2D 한 번으로. 어차피 바로 뒤에서
    같은 `slot_mapping` 을 두 번 gather 하므로 합쳐 올리면 **gather 도 한 번**,
    int32 캐스트도 한 번이 된다 (예전에는 그룹마다 gather 2회 + 캐스트 2회).

## 40-2. 🔴 배치 1 에서는 이득이 없다

5회×2라운드(n=10)로 재면:

| | 최저 | 중앙 | 새 경로가 빠른 횟수 |
|---|---|---|---|
| 배치 1 | 23.00 대 23.00 | 23.22 대 23.29 | **3/10** |

절감분(~0.05 ms)이 잡음(표준편차 0.135s) 아래다. **이득 없음**이다.

그런데 이건 요청당 도는 비용이라 배치를 키워야 드러난다:

| | 최저 | 중앙 | 새 경로가 빠른 횟수 |
|---|---|---|---|
| 배치 6 | 8.50 → **8.45** | 8.61 → **8.56** | **9/10** |

**-0.59%**, 출력은 배치 1·6 모두 바이트 동일. 부호검정으로 9/10 은 p≈0.011 이다.

## 40-3. 이게 더 큰 이야기다

하네스가 프롬프트를 **하나씩** 돌려서 **DDTree 측정이 전부 배치 1 이었다.**
운영은 `max-num-seqs 64` 다. 배치 1 은 '요청마다 도는 비용' 을 구조적으로
가린다 — 이번 항목이 정확히 그 예다.

지금까지의 모든 결론(수용 +17.6%, 기계값 3.0 ms, 마스크 0.70 ms, 2분할 기각)은
**배치 1 조건에서만 검증된 것**이다. 배치가 커지면:

  - 트리의 검증 폭(17토큰 × 요청수)이 배치를 그만큼 키운다 — 배치 1 에서
    "거의 공짜"였던 폭이(§35-6, 토큰당 0.18 ms) 안 그럴 수 있다.
  - 반대로 요청당 CPU 비용은 배치에 비례해 커지므로 이번 같은 절감이 더 값을 한다.

`DDT_BATCH=1` 을 하네스에 넣었다. **다음 측정은 배치를 축으로 다시 해야 한다.**

## 40-4. 방법론 정리

이 장에서 확인된 것을 §39-2 에 더한다.

  - 계측(동기화·CUDA 이벤트)은 3~5배 부풀린다 → 끝에서 끝까지 A/B 만 믿는다.
  - 끝에서 끝까지 A/B 도 **n=3 으로는 0.5% 를 못 가른다.** 잡음 표준편차가
    0.135s(0.6%)다. n=10 + 부호검정이 필요하다.
  - 그리고 **조건(배치 크기)이 틀리면 n 을 아무리 키워도 못 본다.**

# 41. 배치 축 재검증 — DDTree 의 승리는 배치 1 에서만이다 (2026-09-01)

## 41-1. 결과

hard 표본, 8B + EAGLE3 + V2 + cudagraph, 192토큰, 요청 B개 동시:

| 배치 | 사슬 예산3 | 최선 트리 | 대비 | 스텝단가비 | 수용비 |
|---|---|---|---|---|---|
| 1 | 23.13s | 22.89s (d3/b16) | **+1.0%** | 1.167x | 1.176x |
| 2 | 4.15s | 4.25s (d3/b8) | **-2.4%** | 1.142x | 1.114x |
| 4 | 4.39s | 4.52s (d3/b8) | **-2.7%** | 1.187x | 1.154x |
| 16 | 5.19s | 5.43s (d3/b4) | **-4.4%** | 1.214x | 1.160x |

**최적 예산이 배치와 함께 줄어든다: 16 → 8 → 8 → 4.** 배치 16 에서 예산 16 을
쓰면 8.56s 로 **-39%** 다 (스텝 단가 34.17 → 73.13 ms).

## 41-2. 왜 — 폭이 공짜가 아니게 된다

§35-6 은 "검증 토큰이 토큰당 0.18 ms 로 거의 공짜" 라고 했다. 그건 **배치 1 에서
GPU 가 놀고 있기 때문**이었다. 배치 16 에서 트리의 쿼리 토큰은 16×17 = 272 개다.
그 크기에서 타깃 forward 는 메모리 바운드를 벗어나 연산 바운드로 가고, 추가
토큰이 선형으로 값을 부른다.

사슬은 배치에 거의 공짜로 확장된다 (배치 1 30.00 → 배치 16 34.17 ms/스텝,
16배 일에 +14%). 트리는 안 그렇다.

## 41-3. 기계값도 배치에 비례해 커진다

**같은 예산 = 같은 검증 폭 = 같은 스텝 단가.** 그러면 남는 차이는 모양과 기계값
뿐이다:

| 배치 16 | ms/스텝 | 토큰/스텝 | 초 |
|---|---|---|---|
| 사슬 예산3 | 34.17 | 20.211 | **5.19** |
| 트리 예산3(깊이3) | 38.14 (+11.6%) | 20.211 (동일) | 5.80 |
| 사슬 예산5 | 40.55 | 21.483 | **5.80** |
| 트리 예산5(깊이5) | 45.63 (+12.5%) | 24.000 (+11.7%) | 5.84 |

예산 3 에서는 모양 재배분이 수용을 **하나도** 못 벌면서 기계값 +11.6% 를 낸다.
예산 5 에서는 모양이 +11.7% 를 버는데 기계값 +12.5% 가 정확히 먹는다.

기계값이 배치 1 의 2.41 ms 에서 배치 16 의 3.97 ms 로 커진다 — **요청당 도는
CPU 작업**(accept 루프, 요청별 트리 구성, 요청별 마스크 생성, 압축 인덱스)이기
때문이다. §40 에서 줄인 것이 바로 이 축이고, 더 줄일 여지가 있다.

## 41-4. 판정

**운영은 `max-num-seqs 64` 다.** 배치 16 에서 이미 -4.4% 이고 추세가 단조라
64 에서는 더 나쁘다. 최적 예산이 배치와 함께 줄어 4까지 왔으니, 배치 64 에서는
예산 3 — **즉 사슬 자신** — 이 최적일 가능성이 높다.

**DDTree 는 배치 1 에서만 이긴다.** 저동시성 지연 최적화 배치(요청이 드물게
오는 단일 사용자, 대화형 코딩 도구 등)라면 자리가 있지만, 처리량 배치에서는
없다.

## 41-5. 이 장이 뒤집는 것

§35~40 의 결론은 전부 **배치 1 조건**이었다.

  - §35-3 "폭은 드래프터에게 공짜" — 드래프터 쪽은 여전히 맞다(깊이만 forward 를
    먹는다). 그러나 **타깃 검증 쪽은 배치에서 공짜가 아니다.**
  - §35-6 "검증 토큰 토큰당 0.18 ms" — 배치 1 한정.
  - §35-5 "최적은 깊이3/예산16" — 배치 1 한정. 배치 16 에서는 예산 4.
  - §39-5 "DDTree 가 +1.0% 빠르다" — 배치 1 한정.

§38 의 기계값 귀속(3.0 ms)도 배치 1 값이다. 배치 16 에서는 약 4.0 ms 다.

## 41-6. 남은 것

  - 기계값의 요청당 성분을 더 줄이는 것은 여전히 값을 한다 (§40 이 배치 6 에서
    -0.59% 를 실증했다). 다만 그걸로 폭의 값을 이길 수는 없다.
  - **폭의 값을 줄이는 유일한 길은 검증 토큰 수를 줄이는 것**이고, 그건 트리를
    포기하는 것과 같다.
  - 배치 32/64 를 실제로 재서 추세를 확정하는 것은 남았다. 다만 배치 2 에서
    이미 지고 16 까지 단조 악화라 결론이 바뀔 여지는 작다.

# 42. 운영 모델(27B + DFlash2) 대비 — DDTree 가 34% 진다 (2026-09-01)

## 42-1. 조건

운영 설정을 그대로 맞췄다: `cyankiwi/Qwen3.8-27B-AWQ-INT4` + `dflash` 메서드 +
`z-lab/Qwen3.8-27B-DFlash2`, V2 러너, FlashInfer, cudagraph. 운영은
`num_speculative_tokens: 7` 이고 배치별로 `[[1,4,7],[5,64,3]]` 을 쓴다 —
**운영도 이미 배치가 커지면 예산을 줄인다**(§41 이 찾은 추세와 같다).

hard 표본 6프롬프트 × 192토큰, 배치 1, 3회 최저. 예산은 8 이하로 제한했다
(DFlash2 는 `dflash_config` 가 없어 `drafter_k = num_speculative_tokens` 라
학습 지평 밖으로 키우면 무효다 — §21-7-2).

## 42-2. 결과

| 구성 | ms/스텝 | 토큰/스텝 | 초 | 사슬7 대비 |
|---|---|---|---|---|
| base (스펙 없음) | 27.95 | 0.995 | 32.36 | 0.482x |
| DFlash2 사슬 예산3 | 43.13 | 2.661 | 18.67 | 0.835x |
| **DFlash2 사슬 예산7** | **43.34** | **3.200** | **15.60** | **1.000x** |
| DDTree 예산7 β=0 | 63.47 | 3.080 | 23.74 | 0.657x |
| DDTree 예산7 β=-0.3 | 63.55 | 3.097 | 23.64 | **0.660x** |
| DDTree 예산8 β=-0.3 | 78.70 | 3.024 | 29.98 | 0.520x |

**같은 예산(7)에서 -34% 다.** 두 가지가 겹친다.

  1. **스텝 단가 +47%** (43.34 → 63.55 ms). 폭이 같은데도 이만큼 난다 — GDN 트리
     커널이다 (`gdn_tree` 호출 53,376회). §27 이 4B +44% / 27B +49% 로 잰 것과
     같은 값이고, **구조적**이다. 계층수·은닉차원에 비례하므로 모델을 키우면
     오히려 커진다.
  2. **수용도 진다** (3.097 대 3.200). DFlash2 의 1위가 이미 충분히 맞아서
     분기가 헛돈다 — §34 의 easy 표본과 같은 현상이다. hard 표본으로도 안 뒤집힌다.

예산을 8로 늘리면 단가가 +82% 로 뛰고 수용은 오히려 떨어져 -48% 다.

## 42-3. 8B 에서의 +1.0% 는 다른 모델·다른 드래프터였다

§39-5 의 "DDTree 가 +1.0% 빠르다" 는 **Qwen3-8B(순수 어텐션) + EAGLE3** 다.
운영 모델은 **27B GDN 하이브리드 + DFlash2** 이고, 거기서는 -34% 다.

  - GDN 하이브리드: 트리 GDN 커널이 구조적으로 비싸다. 피할 방법이 없다.
  - DFlash2: 1위가 정확해 분기가 값을 못 한다.

**운영 모델에 DDTree 를 넣을 근거는 없다.**

## 42-4. TP=2 정합성 — 정적 점검

운영이 GPU1 을 쓰고 있어 실측은 보류했다. 코드 경로는 확인했다.

`sample()` 에는 두 갈래가 있다.

  - **기본 (batch_sharder 없음)**: `self.model.compute_logits(...)` 가 all-gather
    해서 **모든 랭크가 전체 요청의 전체 vocab logits** 을 받는다. 그러면 랭크마다
    같은 트리·같은 수용 경로가 나오고, KV 압축은 각자 자기 헤드 샤드에 같은 슬롯
    인덱스로 적용된다. 마스크·RoPE 도 전역 배치 기준이라 랭크 간 동일하다.
    → **TP>1 에서 동작해야 한다.**
  - **`enable_batch_sharded_sampling=True`**: 이 시점의 `input_batch` 는 **이
    랭크가 맡은 요청만** 담는데, DDTree 상태(`step`/`all_req_ids`/`q_start`/
    `groups`)는 훅 2 에서 **전역 배치**로 채워졌다. 인덱스가 어긋나 압축이 남의
    슬롯을 건드리고 수용이 다른 요청에 매핑된다. **조용히 깨진다.**

기본값이 `None`(= False)이고 운영도 안 켜므로 TP=2 만으로는 문제가 없다. 다만
누가 켜면 침묵 오류라 훅 4 에 **명시적 가드**를 넣었다.

실측(운영 정지 필요)은 남았다. `--disable-custom-all-reduce` 가 필수다 — P2P 가
VMware passthrough IOMMU fault 로 VM 을 죽인다.

# 43. TP=2 실측 — 동작하고, 판정은 안 바뀐다 (2026-09-01)

## 43-1. 절차

운영 트래픽 실측(실행중 0 / 대기 0, 마지막 요청 10:21 UTC 로 8시간 무트래픽) 후
운영을 내리고 GPU 두 장으로 시험한 뒤 원상 복구했다.

  - 정지 `18:41:35Z` → 복구 명령 `18:51:35Z` → healthy `18:55:14Z`
  - **다운타임 13분 39초**
  - 복구: `VLLM_TP=1 VLLM_GPU_COUNT=1 docker compose -f docker-compose.yml
    -f docker-compose.gpu1.yml up -d` (compose 라벨에서 확인한 원래 기동 방식)
  - 복구 검증: health=healthy, GPU=[1], TP=1, `--disable-custom-all-reduce`,
    max-num-seqs 64, fp8 KV — 전부 원래대로. 실제 추론도 확인(2+2 → "4").

🔴 `--disable-custom-all-reduce` 는 compose 에 이미 있었다. P2P 는 VMware
passthrough IOMMU fault 로 VM 을 죽인다. 시험 내내 VM uptime 10일이 유지됐다.

## 43-2. 결과 — DDTree 는 TP=2 에서 정상 동작한다

27B + DFlash2, 배치 1, hard 6프롬프트 × 64토큰:

| | TP=1 | TP=2 | TP=2 이득 |
|---|---|---|---|
| base (스펙 없음) | 11.07s | 7.41s | 1.49x |
| DFlash2 사슬 예산7 | **5.16s** | **4.30s** | 1.20x |
| DDTree 예산7 | 8.23s | 6.93s | 1.19x |

DDTree 는 크래시 없이 돌고 카운터도 정상이다 (TP=1 기준 트리스텝 262/283,
마스크 1145, dropped 0, GDN 압축 오류 없음, compact_unsafe 0).

**출력 정합성** — 여기가 핵심이다:

| 비교 | 일치 |
|---|---|
| base: TP=1 대 TP=2 | **3/6** |
| DFlash2 사슬: TP=1 대 TP=2 | 5/6 |
| DDTree: TP=1 대 TP=2 | **3/6** |

**스펙 디코딩이 없는 base 조차 TP 간 3/6 만 일치한다.** TP 는 모든 matmul 과
all-reduce 의 리덕션 순서를 바꾸므로 그리디 출력도 갈린다 — 그게 이 축의 잡음
바닥이다. DDTree 의 3/6 은 **base 와 정확히 같은 수준**이고, 즉 **TP 고유의
추가 발산이 없다.** §42-4 의 정적 점검("모든 랭크가 전체 logits 을 받아 같은
트리를 만든다")이 실측으로 확인됐다.

## 43-3. 그러나 판정은 안 바뀐다

사슬 대비 비율: **TP=1 에서 0.627x, TP=2 에서 0.620x** — 사실상 동일하다.

TP=2 가 세 팔을 모두 1.2~1.5배 빠르게 하지만 **상대 순위는 그대로**다. TP 로
예산이 남아도 격차가 안 메워지는 이유는 격차의 정체가 **GDN 트리 커널의 스텝
단가**이고, 그건 TP 로 다른 모든 것과 똑같이 나뉘기 때문이다.

## 43-4. 정리

  - **TP=2 정합성: 통과.** DDTree 가 TP>1 에서 정상 동작하고, TP 고유 발산은 없다.
  - **`enable_batch_sharded_sampling` 은 여전히 금지**다 (§42-4). 훅 4 에 가드가
    들어 있다. 기본값 False 라 TP=2 만으로는 안 켜진다.
  - **성능 판정은 TP 와 무관하다.** 운영 모델에서 DDTree 는 TP=1/2 모두 사슬의
    약 0.62x 다.

# 44. 🔴 §39-5 의 "+1.0% 승" 철회 — 최저값 비교의 함정 (2026-09-01)

## 44-1. 내 규칙을 내가 어겼다

§40-2 에서 "n=3 으로는 0.5% 를 못 가른다" 고 적어놓고, 정작 유일한 양(+)의
결과인 §39-5 의 "+1.0%" 는 **n=3 의 최저값**으로 낸 것이었다. n=10 으로 다시
쟀다 (8B + EAGLE3, 배치 1, hard, 192토큰, 5회×2라운드 짝지어):

| 구성 | n | 최저 | 중앙 | 평균 | sd | 토큰/스텝 |
|---|---|---|---|---|---|---|
| 사슬 예산3 | 10 | 23.10 | **23.29** | 23.26 | 0.097 | 1.498 |
| DDTree d3/b16 | 10 | 22.91 | **23.28** | 23.22 | 0.154 | 1.761 |

  - 최저 기준 **+0.82%** (n=3 의 +1.0% 와 일치한다)
  - **중앙 기준 +0.03%** — 사실상 0
  - DDTree 가 빠른 라운드 **7/10**, 부호검정 양측 **p ≈ 0.344** — 유의하지 않다

## 44-2. 왜 최저값이 속였나

**DDTree 의 분산이 1.6배 크다** (sd 0.154 대 0.097). 최소값 통계량은 분산이 큰
쪽에 체계적으로 유리하다 — 표본이 많을수록 꼬리를 더 깊이 파기 때문이다.
n=3 이면 그 편향이 그대로 '승리' 로 보인다.

분산이 큰 이유도 짐작이 간다. 트리는 스텝마다 모양이 달라(수용 경로 길이가
0~3) 압축 유무·마스크 크기가 요동친다. 사슬은 매 스텝이 같다.

## 44-3. 수정된 결론

**DDTree 가 이기는 조합은 하나도 없다.** 8B 순수 어텐션 + EAGLE3 + 배치 1 은
**동률**이고(수용 +17.6% 를 기계값과 폭 값이 정확히 상쇄), 나머지는 전부 진다.

문서 상단 요약의 `1.01x` 를 `1.00x (동률)` 로 고쳤다.

## 44-4. 측정 규칙 6번

**최저값으로 팔을 비교하지 말 것.** 분산이 다른 두 팔에서 최소값 비교는 편향
추정량이다. 중앙값(또는 평균)과 부호검정을 쓸 것. 최저값은 '이 구성이 낼 수 있는
가장 좋은 수' 를 보는 용도이지 비교 통계가 아니다.

이 세션의 여러 표가 최저값 기준이다. 격차가 5% 이상인 것들(27B 0.62x, 배치 16
0.96x, GDN 하이브리드 0.62~0.68x)은 잡음(sd 0.6%)보다 훨씬 커서 결론이 안
바뀌지만, **1~2% 급 주장은 전부 이 기준으로 다시 봐야 한다.**
