# Post-hoc thử: v3 nhưng bậc 1 bỏ qua series *_latency. Patch rank_scored rồi gọi eval_ranking.
import sys, runpy
import baro.root_cause_analysis as rca
orig = rca.rank_scored
def patched(scored, steps, *a, **k):
    return orig(scored, {n: s for n, s in steps.items() if not n.endswith("_latency")}, *a, **k)
rca.rank_scored = patched
sys.argv = ["eval_ranking.py"] + sys.argv[1:]
runpy.run_path("scripts/eval_ranking.py", run_name="__main__")
