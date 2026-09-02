# DDTree on vLLM — Findings

An independent implementation of DDTree (Diffusion Draft Tree, arXiv:2604.12989)
on vLLM, taken far enough to answer "does this pay off in production?"

**Answer: no, in every configuration we tested.** This document records what we
measured, what we got wrong on the way, and which parts are reusable regardless
of DDTree.

Reference implementation: <https://github.com/liranringel/ddtree> (MIT).
Full experiment log: [`DDTREE-SCOPING.md`](DDTREE-SCOPING.md) — ~4,900 lines,
**written in Korean**. This document is the English summary. Section numbers
below (§N) point into it.

Hardware: 2× RTX A6000 (sm_86), x86_64. vLLM `0.26.1rc1.dev1177+ga9a17e709`
(image `vllm/vllm-openai:nightly`). All numbers are with CUDA graphs enabled
and no profiling hooks unless stated.

---

## 1. Verdict

DDTree vs. **the same drafter running a flat chain at the same budget** — not
vs. no speculation.

| Configuration | batch | DDTree vs chain |
|---|---|---|
| Qwen3.8-27B-AWQ (GDN hybrid) + DFlash2, T=0 | 1 | **0.78x** |
| Qwen3.8-27B-AWQ + DFlash2, production sampling¹ | 1 | **0.69x** |
| Qwen3-8B (pure attention) + EAGLE3, T=0 | 1 | **1.00x** (tie) |
| Qwen3-8B + EAGLE3 | 2 / 4 / 16 | 0.98x / 0.97x / 0.96x |
| Qwen3.5-4B (GDN hybrid) + DFlash | 1 | 0.68x² |

¹ `temperature=1.0, top_k=20, top_p=0.95` — vLLM's default when
`--generation-config` is unset, taken from the model's `generation_config.json`.
Any client that doesn't send a temperature gets this.
² Measured before the fixes in §5; not re-measured.

The best case we found — pure attention, a drafter that is frequently wrong,
batch 1 — is a **tie**. Production is the opposite of all three.

---

## 2. The single most important methodological point

**Do not benchmark tree speculative decoding under `enforce_eager=True`.**

Eager execution is launch-bound, so it has GPU idle time that silently absorbs
the tree's extra GPU work. Turning CUDA graphs on removes that slack and the
overhead appears. Same config (Qwen3.5-4B GDN hybrid + DFlash, batch 1, greedy):

| | baseline (flat chain) | DDTree |
|---|---|---|
| `enforce_eager=True` | 70.9 tok/s (53.2 ms/step) | 70.7 (54.0) — no difference |
| CUDA graph | **144.9** tok/s (26.3 ms/step) | **99.1** (37.9) — **−31.6%** |

Both arms took **the same number of steps (34)**: the tree bought no extra
acceptance here, only cost.

Per-layer profiling hooks act in the same direction (~10%): the same eager
baseline reads 63.7 tok/s with hooks and 70.9 without.

We published a **+1.2%** win for this configuration and had to retract it — it
was measured under eager *with* hooks. See §22–§23 of the log.

**Always count `execute_model` calls directly.** Without separating acceptance
gain from per-step cost, causal claims are guesswork.

---

## 3. Nine measurement rules, learned expensively

1. **Instrumentation inflates 3–5×.** Same op, GPU time: isolated microbench
   `logsumexp 0.062 / topk 0.077 ms`; in production with per-call CUDA events
   `0.295 / 0.141`; with per-section `synchronize` `0.327 / 0.164`. Trust only
   **end-to-end A/B** and **isolated microbenchmarks**. (§39-2)
2. **n=3 cannot resolve 0.5%.** Noise sd is ~0.6%. Use n≥10 and a sign test. (§40-2)
3. **The wrong condition hides the effect at any n.** Our whole harness ran at
   batch 1 while production is `max-num-seqs 64`. (§40-3)
4. **Section-disabling ablations are invalid here.** Losslessness protection
   cascades: disable the mask and `accept()` rejects the tree, which also
   disables RoPE and compaction — so the "mask cost" you measure is the entire
   pipeline. (§37-1)
5. **Marginal values measured from forced shapes don't transfer to mixed trees.**
   Forcing depth-1 or topk-1 removes selection bias but the numbers you get
   don't predict the blended tree. (§34-5)
6. **Never compare arms by minimum.** With unequal variance the minimum is a
   biased estimator — DDTree had 1.6× the sd of the chain and "won" on minimum
   only. Use median + sign test. (§44)
7. **Noise depends on the *other* GPU's temperature.** Same A/B design, sd went
   0.15% → 0.5% because the neighbouring card was at 92 °C under production
   load and dragged our clocks. Check `nvidia-smi` before fine measurement. (§49-7)
8. **Distinguish "no effect" from "underpowered."** Report the noise sd next to
   the effect size. A section-level saving of 0.06 ms/step is simply invisible
   end-to-end at 0.5% sd.
9. **Counterfactuals from a single trace are position-biased.** Computing "what
   if we had only used the spine" from a tree run understates the chain by ~10%,
   because the tree stops exactly where continuation is hard. Our
   counterfactual said +15%; end-to-end step counts said **+3.4%**. Use traces
   for conditional probabilities only; use step counts for tokens/step. (§50-4)

---

## 4. Where the time actually goes

27B + DFlash2, batch 1, budget 7. The gap decomposes cleanly once you can force
the tree into chain *shape* (`VLLM_DDTREE_TOPK=1`) and skip the mask
independently — a chain-shaped tree's visibility is identical to causal, so the
mask can be dropped without breaking losslessness.

| | ms/step | tokens/step |
|---|---|---|
| A: DFlash2 flat chain | 43.34 | 3.200 |
| B: DDTree, chain shape, mask omitted | 61.32 | 3.200 |
| C: + `custom_mask` | 63.04 | 3.200 |
| D: + real tree shape | 63.55 | 3.097 |

**B − A = 17.99 ms is pure integration overhead** — same acceptance, same shape.
`custom_mask` is 1.71 ms and the tree shape itself is 0.51 ms. Almost all of the
loss was our own machinery, not the algorithm.

Final accounting of that 17.99 ms:

| item | ms/step | status |
|---|---|---|
| `gdn_info()` rebuilding tensors per layer | **9.77** | fixed (§5) |
| GPU idle from the CPU tree walk | **~4.90** | structural |
| GDN tree path residual | 2.11 | unresolved |
| D2H transfers | 0.32 | structural |
| depth RoPE | 0.21 | |
| KV compaction | 0.05–0.56 | mostly fixed |
| mask array construction | 0.07 | fixed |
| other GPU work | ~0.35 | |

### The structural half: GPU idle, not GPU work

A torch profiler run over both arms (24 tokens, ~7.5 steps):

| | total CUDA | total CPU | wall | GPU busy |
|---|---|---|---|---|
| chain | 329.3 ms | 289.2 ms | 325 ms | **101%** |
| minimal DDTree | 331.9 ms | 319.6 ms | 369 ms | **90%** |

**GPU work differs by only +2.6 ms; CPU by +30.4 ms.** Chain speculative
decoding finishes verification entirely on the GPU (the rejection sampler is a
kernel, the CPU never intervenes), so it pipelines perfectly. DDTree comes down
to the CPU every step to walk the tree (`follow_tree` is a Python traversal of
`child_maps`). That round trip is a **4.9 ms/step bubble** at 27B.

The largest profiler entry was `cudaMemcpyAsync` (25.7 ms / 27 calls). That is
**not transfer cost** — replacing the host buffers with pinned memory changed
nothing. It is the CPU waiting on the GPU stream.

---

## 5. Optimizations that worked

| Change | Effect | Output |
|---|---|---|
| Memoize `gdn_info()` per step instead of per GDN layer | **−9.77 ms/step**, 0.660x → 0.780x | byte-identical |
| Degenerate fast path: skip KV compaction when the accepted path is `[0,1,2,…]` (`src == dst`) | −0.39 ms/step; hits 59% of steps | byte-identical |
| Omit `custom_mask` for chain-shaped trees (causal is equivalent) | −0.70 ms/step | byte-identical |
| Collapse the top-k path to two dispatches (`log_softmax` + `topk`) and merge two D2H copies into one | section −46%, −0.13 ms/step e2e | byte-identical |
| Remove per-request tensor creation in `accept` / `kv_compact` | −0.59% at batch 6; **0 at batch 1** | byte-identical |
| Remove the compaction safety-check D2H by making the Triton kernel order-independent (read all, then write all) | −0.32% (n=12, 6/6, p=0.031) | byte-identical |
| Memoize the KV cache pointer tensor | call 59.4 → 26.7 µs; not resolvable end-to-end | byte-identical |

Two of these are worth stealing regardless of DDTree:

**`gdn_info()` per-layer rebuild.** 27B has 48 GDN layers and the result is
identical across all of them — same step, same tree. We were doing 96 H2D
copies and 48 Python tree traversals per step. The same file's `tolist_cached`
had already solved exactly this for `state_indices`, comment and all. Worth
auditing any per-layer provider for invariants.

**Chain-shaped drafts don't need a custom mask.** Any tree implementation that
degenerates to chain shape is paying for a mask kernel it doesn't need, and the
output is bit-identical without it.

### The top-k dispatch finding

We assumed the top-k → CPU path was dominated by the D2H. It wasn't:

| | ms/step | share |
|---|---|---|
| GPU kernels (isolated) | 0.147 | 19% |
| D2H | 0.056 | 7% |
| Python heap (tree build) | 0.098 | 13% |
| **CPU dispatch** | **0.429** | **57%** |

**Operation *count* was the cost.** In eager PyTorch each dispatch is 30–40 µs,
and `float() → topk → logsumexp (5 internal ops) → subtract → 2× D2H → 2× numpy`
was throwing more than ten. `log_softmax` is by definition
`logits − logsumexp(logits)`, and its `dtype` argument absorbs the cast, so the
whole thing collapses to two dispatches.

This is also why **"move the tree build to the GPU to eliminate the D2H" is the
wrong target** — the D2H is 0.056 ms.

---

## 6. Optimizations that failed

Recorded because they look attractive.

| Attempt | Result |
|---|---|
| **Split attention** into a paged context part (no mask) + a ragged local part (q×q mask), merged with `merge_state` | Correct, but **loses at every context length** (0/700/1500/3000 tokens), and the gap *widens* with context. The merge runs **per attention layer** — 36 layers at 8B means the fixed per-layer cost dominates, while the mask savings were ~0. |
| **Pinned host buffers** for the D2H copies | No effect (−0.33 / +0.30 ms in two runs). Reverted. This is the evidence that the `cudaMemcpyAsync` time is waiting, not transferring. |
| **Adaptive width** — calibrate best-first weights with per-depth (β) and per-rank (δ) bonuses | Theory and average-distribution arithmetic predicted +8.6%; measured **+1.4%**. Rule 5. |
| **Move the tree walk to the GPU** | Ceiling is ~49 ms/step = **0.88x** — still a loss. Would require moving `follow_tree`, tree construction *and* mask generation, all-or-nothing. |
| **Wider trees** (budget 12 / 16 / 32) | Zero additional acceptance. See §8. |
| **Temperature > 0** | Makes it *worse*: 0.78x → 0.69x. |

---

## 7. Things we got wrong (retracted)

Kept visible because the wrong versions are the intuitive ones.

- **"The tree GDN kernel is structurally expensive (+44–49% per step)."**
  This was inference, not measurement. Measured directly at our operating point
  (T=8): the tree kernel costs **1.2 µs/layer** more than the stock fused
  kernel — 0.06 ms/step across 48 layers, **0.3% of the gap**. The real cause
  was §4. (§46)
- **"Mask cost scales with context length."** Based on a `NOMASK` ablation that
  was invalid by rule 4 — disabling the mask disabled the whole tree pipeline.
  Real cost: 0.70 ms/step, context-independent. (§37-1)
- **"DDTree wins by 22% on pure attention + ngram."** Not reproducible; the
  baseline had improved underneath us. (§24)
- **"DDTree is 1.0% faster at batch 1 (8B)."** Minimum-comparison bias; it's a
  tie. (§44)
- **"Each sync point costs ~0.95 ms."** Came from dividing a profiler total by
  call count and applying the average everywhere. Measured: 0.20 ms. (§49-5)
- **"The budget saturates, so spend the surplus on branches."** Saturation was
  not loss of confidence — the drafter's rank-0 was simply already correct. (§33)
- **"T>0 requires tree rejection sampling."** It doesn't; see §9.

---

## 8. The acceptance ceiling — measured, not estimated

The deepest question: *how much acceptance is available from branching at all?*

**Method.** A node's sampled token is "what the target picked given that node's
ancestor context" — which stays valid in **any sub-tree containing that node**.
So run one wide tree (spine of 7 + siblings, budget 16 and 32) with tracing,
then compute acceptance for every sub-shape offline from the same trace. We also
record the drafter's **top-16 candidates per depth**, so sibling hit rates are
measured without selection bias (no forced shapes needed).

27B + DFlash2, 6 open-ended prompts × 192 tokens, batch 1, greedy:

| Configuration | tokens/step | steps | ms/step | vs chain |
|---|---|---|---|---|
| **DFlash2 chain, budget 7** | 3.200 | 360 | 43.35 | 1.00x |
| spine + 1 sibling (budget 8) | 3.200 | 360 | 60.77 | 0.71x |
| spine + 2 siblings (budget 9) | **3.329** | **346** | 62.04 | 0.73x |
| spine + 5 siblings (budget 12) | 3.273 | 352 | 61.98 | 0.72x |
| spine + 9 siblings (budget 16) | 3.310 | 348 | 63.23 | 0.71x |
| spine + 25 siblings (budget 32) | 3.282 | 351 | 72.53 | 0.61x |

**Acceptance available from branching is +3.5–4%, and it saturates at two
siblings.** Budgets 12/16/32 buy zero additional steps.

Why, from the trace:

- At the first mismatch, the target's token **is** the rank-1 sibling **37%** of
  the time (top-3: 58%). Sibling slots genuinely exist.
- The drafter *underestimates* siblings: at (depth 2, rank 1) the measured hit
  rate is 0.473 against a stated 0.218.
- **But the sibling's continuation doesn't follow.** After a sibling is
  accepted, the next depth's marginal is correct only **22%** of the time,
  versus **65–84%** on the spine (15% at budget 32).

A block-diffusion drafter emits *marginals* for k positions from one forward
pass, and those marginals are effectively conditioned on the argmax path. So a
sibling is worth exactly one leaf, and the sibling's children are mostly noise.
That is a property of the drafter family, not of the tree implementation.

Note this also means calibration constants are **per-checkpoint and can flip
sign**: 4B DFlash *under*estimates the deep spine, 27B DFlash2 *over*estimates
it (stated 0.776 vs. measured 0.691 at depth 1).

### Ceiling arithmetic

| Lever | Measured | Remaining ceiling |
|---|---|---|
| Narrow superset tree (spine + 2 siblings) | +3.5–4% acceptance | — |
| Remove machine overhead (GPU idle 4.9 + GDN residual 2.1) | not implemented | 0.88x |
| + widen the GDN tree kernel past T=8 (§10) | not implemented | **0.94x** |
| Conditional re-draft to fix sibling continuation | trace says +0.5/step | costs another drafter forward (~8–10 ms) → **net loss** |
| Tree rejection sampling | no-op (§9) | 0 |

**Even with every implementation lever pulled, 0.94x.** Beating DFlash2 requires
a different drafter — one that emits conditional distributions after a sibling,
or one retrained to stop underestimating siblings.

---

## 9. T>0 works, and tree rejection sampling is unnecessary

We implemented distribution-preserving verification at T>0 without tree
rejection sampling: each node's logits are computed under its ancestor context
(that is what the tree mask means), so **sample at every node and walk the tree
with the sampled token**. Every emitted token is then a legitimate draw from `p`
at its position. Greedy is the special case, and output is byte-identical to the
previous code at T=0.

**Sequential tree rejection sampling would buy nothing here.** For greedy draft
candidates the vLLM rejection sampler takes the `NO_DRAFT_PROBS` branch
(`draft_prob = 1`), i.e. accept with probability `p(x)`. Chaining candidates
x₁, x₂ gives `p₁ + (1−p₁)·p₂/(1−p₁) = p₁ + p₂` — identical to sample-and-walk.
A difference requires using the drafter's `q` (probabilistic drafting), which
moved the *chain* baseline by +2.5% (n=1, within text noise).

---

## 10. A kernel cliff worth knowing about

Step cost as a function of query tokens per step (27B):

| tokens/step | 8 | 9 | 10 | 13 | 17 | 33 |
|---|---|---|---|---|---|---|
| ms/step | 53.8 | 60.8 | 62.0 | 62.0 | 63.2 | 72.5 |

Flat from 9 to 17, with a **+7 ms cliff at 8→9**. It is **not** CUDA-graph
padding — adding an exact capture size of 10 changes nothing (62.07 vs 62.04).

Profiling budget 7 vs 9 shows why: at T≤8 a single fused CUDA kernel runs per
GDN layer; above 8 it falls back to a non-fused Triton path plus packing,
normalization and copies. Per step: CUDA +3.4 ms, **CPU +16.5 ms**, and
`aten::copy_` call count 1760 → 3919. With 48 layers that is roughly **500 extra
kernel launches per step** — launch-bound.

The stock fused GDN MTP kernel has a `state_indices [N, S]`, `S ≤ 8` limit, and
our tree kernel inherited it. A chain never hits this because the DFlash2
drafter's horizon is 8.

---

## 11. Correctness findings

### 11.1 `num_computed_tokens_np` is an optimistic upper bound

Depth-based RoPE needs each tree node's absolute position. We used
`depth + num_computed[CPU mirror]`. In the V2 model runner that CPU mirror is an
**optimistic upper bound** (`vllm/v1/worker/gpu/states.py:61` says so). V1
rewinds the CPU value at the end of the step too, so the bug was invisible
there; V2 corrects only on the GPU. **94% of tree steps were off by +4 to +10.**

Fix: read the base from `positions[s]`, the authoritative value, which is also a
GPU scalar so it costs no synchronization. Byte-identical no-op on V1.

Anything that derives absolute positions from the CPU mirror mid-step has this
bug.

### 11.2 "Bit-identical to non-speculative greedy" is the wrong bar

Tree verification diverged from greedy — but so does vLLM's *own* chain
speculative decoding, on **the same prompts at the same positions**. Once a
draft is attached the query width is > 1 and the prefill kernel path is taken,
which flips bf16 near-ties.

The correct criterion is **"identical to chain speculative decoding under the
same conditions"**, and DDTree meets it. Judge divergences by the top-1/top-2
logit gap, not by count: a gap in the bottom 5% of the distribution is a
numerical tie, not a bug. We misjudged this in *both* directions before adopting
the gap test.

### 11.3 GDN hybrids: the two kernels differ by exactly 1 bf16 ULP

Re-implementing the kernel maths in fp64 against real dumped inputs:

```
CUDA   state  vs fp64           0.000000
Triton state  vs fp64           0.000000
CUDA   output vs fp64           0.003560
Triton output (kept fp32)       0.000803
Triton → bf16 stored            0.003560   ← same as CUDA
fp64 merely stored as bf16      0.003560   ← the storage floor itself
```

The recursive state is exact in both; output error equals the bf16 storage
floor. The kernels cannot be more accurate. But GDN is recurrent with no
renormalization, so a 1-ULP difference compounds across 24 GDN layers × dozens
of steps and eventually flips a decision in deep trees. Not a logic error —
amplification of arithmetic order.

---

## 12. Side findings useful to vLLM independent of DDTree

1. **`flashinfer.merge_state` is consistently faster than vLLM's
   `merge_attn_states`** (measured +0.67 vs +1.00 ms at context 700; +1.30 vs
   +1.52 at 3000). The FlashInfer wrapper's LSE can be consumed **in its native
   layout and log2 domain**, which removes a per-layer transpose → contiguous
   copy → log2-to-ln conversion (about 6 kernels per layer). Verified: against a
   log2 reference the error is 0.0008; against natural log, 0.34 — log2 is
   correct. This should apply to the DCP path, which has the same structure.
2. **Chain-shaped speculative drafts do not need `custom_mask`.** −0.70 ms/step,
   byte-identical.
3. **Per-layer providers should be memoized per step.** See §5.
4. Production default sampling is easy to get wrong when benchmarking: with
   `--generation-config` unset, vLLM takes `temperature/top_k/top_p` from the
   model's `generation_config.json`. Benchmarks written against `temperature=0`
   do not represent that traffic.

---

## 13. What would actually be needed

Not "optimize the implementation further" — that path ends at 0.94x.

- **A drafter whose distribution after a sibling is conditional**, not marginal.
  Two-stage denoising, or a drafter retrained so that rank-1 continuations
  survive. This is outside vLLM.
- Failing that, tree drafting pays off only where the drafter is **frequently
  wrong** and **batch size is 1**. Our best measured case under those conditions
  is a tie.

The tree machinery itself is sound: tree construction matches the reference
implementation (12/12), branch-node logits match a standalone run of the same
ancestor path (48/48), and the GDN tree CUDA kernel is exact against fp64 for
arbitrary slots, accepted lengths 1–33, T=96 and branch depth 11.
