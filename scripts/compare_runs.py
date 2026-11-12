"""So sánh các lần chạy bench (per-case JSONL của `rcaeval-agent evaluate --per-case-out`).
Cần scipy (`pip install scipy`).

    R=findings/2026-11-11-re2ob-f1/runs
    python scripts/compare_runs.py tk1_r1=$R/percase_llm_baro_tk1_r1.jsonl \
        tk2_r1=$R/percase_llm_baro_tk2_r1.jsonl ... --pairs tk1_r1:tk2_r1

In ra: AC@1/Avg@5 (service, indicator), indicator AC@1 theo loại fault, chi phí
(lấy từ summary_*.json cùng tên), và với mỗi cặp: số case lệch nhau + McNemar exact.
Với --groups/--group-pairs: gộp các lần lặp của cùng cấu hình (điểm mỗi case = trung bình
qua các rep), so sánh paired theo case bằng sign-flip permutation test + bootstrap CI.
"""
import argparse
import json
import random
from os.path import basename, dirname, exists, join

from scipy.stats import binomtest

FAULTS = ["cpu", "mem", "disk", "socket", "delay", "loss"]


def load(path):
    with open(path) as f:
        return {r["case_id"]: r for r in map(json.loads, f)}


def avg5(rank):
    return 0.0 if rank is None or rank > 5 else (6 - rank) / 5


def usage(path):
    summary = join(dirname(path), basename(path).replace("percase_", "summary_").replace(".jsonl", ".json"))
    if not exists(summary):
        return None
    u = json.load(open(summary)).get("usage", {})
    return {k: u[k]["mean"] for k in ("cost_usd", "tool_calls", "input_tokens") if k in u}


def mcnemar(a, b, ids, level):
    key = f"{level}_rank"
    only_a = sum(a[i][key] == 1 and b[i][key] != 1 for i in ids)
    only_b = sum(b[i][key] == 1 and a[i][key] != 1 for i in ids)
    p = binomtest(only_b, only_a + only_b).pvalue if only_a + only_b else 1.0
    return only_a, only_b, p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+", help="name=percase.jsonl")
    ap.add_argument("--pairs", nargs="*", default=[], help="nameA:nameB")
    ap.add_argument("--groups", nargs="*", default=[], help="nhóm=run1,run2 (các lần lặp của cùng cấu hình)")
    ap.add_argument("--group-pairs", nargs="*", default=[], help="nhómA:nhómB")
    args = ap.parse_args()
    runs = {}
    paths = {}
    for spec in args.runs:
        name, path = spec.split("=", 1)
        runs[name], paths[name] = load(path), path
    ids = sorted(set.intersection(*(set(r) for r in runs.values())))
    print(f"{len(ids)} case chung\n")

    print(f"{'run':16s} {'svc@1':>6s} {'svcAvg5':>7s} {'ind@1':>6s} {'ind@3':>6s} {'indAvg5':>7s} | "
          + " ".join(f"{f:>6s}" for f in FAULTS) + " |  $/case  calls/case")
    for name, r in runs.items():
        n = len(ids)
        s1 = sum(r[i]["service_rank"] == 1 for i in ids)
        i1 = sum(r[i]["indicator_rank"] == 1 for i in ids)
        i3 = sum((r[i]["indicator_rank"] or 99) <= 3 for i in ids)
        sa = sum(avg5(r[i]["service_rank"]) for i in ids) / n
        ia = sum(avg5(r[i]["indicator_rank"]) for i in ids) / n
        per = []
        for f in FAULTS:
            fi = [i for i in ids if r[i]["fault"] == f]
            per.append(f"{sum(r[i]['indicator_rank'] == 1 for i in fi):2d}/{len(fi):<3d}")
        u = usage(paths[name]) or {}
        cost = f"{u['cost_usd']:.4f}" if u.get("cost_usd") is not None else "   -  "
        calls = f"{u['tool_calls']:.1f}" if "tool_calls" in u else "-"
        print(f"{name:16s} {s1:6d} {sa:7.3f} {i1:6d} {i3:6d} {ia:7.3f} | " + " ".join(per) + f" |  {cost}  {calls}")

    if args.pairs:
        print("\nCặp (A:B)            level      chỉ A đúng  chỉ B đúng  McNemar p")
    for pair in args.pairs:
        a, b = pair.split(":")
        for level in ("service", "indicator"):
            oa, ob, p = mcnemar(runs[a], runs[b], ids, level)
            print(f"{pair:20s} {level:10s} {oa:10d} {ob:11d}  {p:9.4f}")

    if args.groups:
        groups = dict(g.split("=", 1) for g in args.groups)
        groups = {g: members.split(",") for g, members in groups.items()}
        score = {}  # group -> metric -> case -> mean over the group's runs
        for g, members in groups.items():
            score[g] = {
                "ind@1": {i: sum(runs[m][i]["indicator_rank"] == 1 for m in members) / len(members) for i in ids},
                "indAvg5": {i: sum(avg5(runs[m][i]["indicator_rank"]) for m in members) / len(members) for i in ids},
            }
        print("\nGộp các lần lặp (điểm mỗi case = trung bình qua các run trong nhóm)")
        print(f"{'nhóm':10s} {'runs':>4s} {'ind@1':>7s} {'indAvg5':>8s} | " + " ".join(f"{f:>6s}" for f in FAULTS))
        for g, members in groups.items():
            per = []
            for f in FAULTS:
                fi = [i for i in ids if runs[members[0]][i]["fault"] == f]
                per.append(f"{sum(score[g]['ind@1'][i] for i in fi):6.1f}")
            print(f"{g:10s} {len(members):4d} {sum(score[g]['ind@1'].values()) / len(ids):7.3f} "
                  f"{sum(score[g]['indAvg5'].values()) / len(ids):8.3f} | " + " ".join(per))
        print(f"\n{'A:B':16s} {'metric':8s} {'B - A':>7s} {'95% CI (bootstrap theo case)':>30s} {'perm p':>7s}")
        for pair in args.group_pairs:
            a, b = pair.split(":")
            for metric in ("ind@1", "indAvg5"):
                d = [score[b][metric][i] - score[a][metric][i] for i in ids]
                lo, hi = bootstrap_ci(d)
                print(f"{pair:16s} {metric:8s} {sum(d) / len(d):+7.3f} {f'[{lo:+.3f}, {hi:+.3f}]':>30s} {sign_flip_p(d):7.4f}")


def bootstrap_ci(d, n=10000, seed=0):
    rng = random.Random(seed)
    means = sorted(sum(rng.choice(d) for _ in d) / len(d) for _ in range(n))
    return means[int(0.025 * n)], means[int(0.975 * n)]


def sign_flip_p(d, n=20000, seed=0):
    """Paired permutation test: flip the sign of each case's difference at random."""
    rng = random.Random(seed)
    observed = abs(sum(d))
    nonzero = [x for x in d if x]
    if not nonzero:
        return 1.0
    hits = sum(abs(sum(x if rng.random() < 0.5 else -x for x in nonzero)) >= observed - 1e-12 for _ in range(n))
    return (hits + 1) / (n + 1)


if __name__ == "__main__":
    main()
