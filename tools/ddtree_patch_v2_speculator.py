"""자기회귀 speculator 의 드래프팅 깊이를 트리 예산에서 분리한다.

🔴 V2 는 num_speculative_steps 하나가 (a) 드래프터 루프 횟수와
   (b) 스케줄되는 드래프트 폭을 겸한다. DDTree 는 둘이 달라야 한다 —
   예산(폭)은 크고 지평(깊이)은 작다. §29 에서 EAGLE3 는 깊이 5 에서
   수용이 포화하는데, 예산 16 을 주면 드래프터가 16번 돌아 11번이 낭비된다.

   VLLM_DDTREE_DEPTH 가 있으면 루프만 거기서 끊는다. draft_logits 버퍼의
   나머지 열은 채워지지 않은 채로 남고, DDTree 가 앞쪽 depth 열만 읽는다.

  사용: python3 patch_v2_speculator.py <원본> <출력>
"""
import io, sys

src, dst = sys.argv[1], sys.argv[2]
s = io.open(src, encoding="utf-8").read()
old = "        for step in range(1, self.num_speculative_steps):"
if s.count(old) != 2:
    raise SystemExit(f"🔴 루프 앵커가 2곳이 아니다: {s.count(old)}곳")
new = """        # --- DDTree: 드래프터 지평을 트리 예산에서 분리한다 ---
        import os as _os
        _dep = int(_os.environ.get("VLLM_DDTREE_DEPTH", "0")) or self.num_speculative_steps
        _steps = min(self.num_speculative_steps, max(1, _dep))
        for step in range(1, _steps):"""
s = s.replace(old, new)
io.open(dst, "w", encoding="utf-8").write(s)
compile(s, dst, "exec")
print(f"  드래프팅 루프 2곳에 깊이 상한 적용 → {dst}")
