"""Đánh giá offline (không LLM) các phiên bản RobustScorer trên một task view của RCAEval.

Đo thứ hạng của metric đúng `<root>_<indicator>` và của service đúng trong ranking, theo
từng loại fault. Cần RCAEval đã cài (đọc case qua CaseToolkit, đúng cửa sổ agent thấy) và
baro trên PYTHONPATH:

    PYTHONPATH=. python scripts/eval_ranking.py --tasks ../agent_tasks/re2-ob \
        --variants v1 v2 v3 v3:none "v3:quiet_rank=score" "v3:tier:min_rise=0.5"

Variant: v<version>[:<step_rank>][:<key>=<value>...], key là tham số của robust_scorer
(step_rank, quiet_rank, min_persistence, min_rise). --json-out lưu kết quả từng case.
"""
import argparse
import json
import warnings
from collections import defaultdict
from os.path import join

from baro.root_cause_analysis import robust_scorer

FAULT_INDICATOR = {"cpu": "cpu", "mem": "mem", "disk": "diskio", "socket": "socket", "delay": "latency",
                   "loss": "latency", "f1": "latency", "f2": "latency", "f3": "latency", "f4": "latency",
                   "f5": "latency"}
KS = (1, 3, 5, 20)


def parse_variant(spec):
    parts = spec.split(":")
    kwargs = {"version": parts[0].lstrip("v")}
    for p in parts[1:]:
        if "=" in p:
            k, v = p.split("=", 1)
            kwargs[k] = v if k in ("step_rank", "quiet_rank", "baseline") else float(v)
        else:
            kwargs["step_rank"] = p
    return kwargs


def service_of(name):
    from RCAEval.benchmark.scoring import to_service
    return to_service(name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", required=True, help="task view của RCAEval, vd. ../agent_tasks/re2-ob")
    ap.add_argument("--answers", default=None, help="mặc định <tasks>.answers.jsonl")
    ap.add_argument("--variants", nargs="+", default=["v1", "v2"])
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args()
    warnings.filterwarnings("ignore")
    from RCAEval.agent.toolkit import CaseToolkit

    answers = [json.loads(l) for l in open(args.answers or args.tasks.rstrip("/") + ".answers.jsonl")]
    variants = {v: parse_variant(v) for v in args.variants}
    per_case = []
    for a in answers:
        tk = CaseToolkit(join(args.tasks, "cases", a["case_id"]))
        data = tk.standard_metrics()
        target = f"{a['service']}_{a.get('indicator') or FAULT_INDICATOR.get(a['fault'], a['fault'])}"
        row = {"case_id": a["case_id"], "fault": a["fault"], "target": target}
        for name, kw in variants.items():
            r = robust_scorer(data, inject_time=tk.inject_time, dataset=tk.task["dataset"], **kw)
            ranks = r["ranks"]
            services = list(dict.fromkeys(service_of(n) for n in ranks))
            row[name] = {
                "ind": ranks.index(target) + 1 if target in ranks else None,
                "svc": services.index(a["service"]) + 1 if a["service"] in services else None,
                "top1": ranks[0] if ranks else None,
                "flagged": sorted(r.get("steps", {})),
            }
        per_case.append(row)

    faults = sorted({r["fault"] for r in per_case})
    n = len(per_case)
    print(f"{n} case, target = <root>_<indicator>; ind@k / svc@k là số case có hạng <= k\n")
    print(f"{'variant':34s} " + " ".join(f"ind@{k:<2d}" for k in KS) + "  svc@1 svc@3 | ind@1 theo fault: "
          + " ".join(faults))
    for name in variants:
        ind = [r[name]["ind"] or 999 for r in per_case]
        svc = [r[name]["svc"] or 999 for r in per_case]
        by_fault = " ".join(f"{sum(r[name]['ind'] == 1 for r in per_case if r['fault'] == f)}/"
                            f"{sum(r['fault'] == f for r in per_case)}" for f in faults)
        print(f"{name:34s} " + " ".join(f"{sum(x <= k for x in ind):6d}" for k in KS)
              + f"  {sum(x <= 1 for x in svc):5d} {sum(x <= 3 for x in svc):5d} | {by_fault}")
        if variants[name]["version"] == "3":
            flagged = [r[name]["flagged"] for r in per_case]
            tgt = sum(r["target"] in f for r, f in zip(per_case, flagged))
            on_root = sum(sum(service_of(x) == r["target"].split("_")[0] for x in f if x != r["target"])
                          for r, f in zip(per_case, flagged))
            other = sum(len(f) for f in flagged) - tgt - on_root
            print(f"{'':34s}   cờ sustained step: target {tgt}, khác trên service gốc {on_root}, "
                  f"service khác {other}")
    if args.json_out:
        with open(args.json_out, "w") as f:
            for r in per_case:
                f.write(json.dumps(r) + "\n")


if __name__ == "__main__":
    main()
