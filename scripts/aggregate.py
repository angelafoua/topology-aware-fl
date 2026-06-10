"""Aggregate runs/*/result.json into Table I / Figure 2 style summaries."""
from __future__ import annotations

import argparse
import glob
import json
import os
from collections import defaultdict
from statistics import mean, pstdev


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="runs")
    ap.add_argument("--out", default="runs/summary.csv")
    args = ap.parse_args()

    rows = []
    for path in glob.glob(os.path.join(args.root, "*", "result.json")):
        with open(path) as f:
            r = json.load(f)
        c = r["config"]
        rows.append({
            "task": c["task"], "method": c["method"], "p": c["p"], "T": c["T"],
            "seed": c["seed"], "lr": c["lr"],
            "acc": r["final_mean_acc"],
        })

    # group by (task, method, p, T) and compute mean +/- std over seeds
    groups = defaultdict(list)
    for r in rows:
        groups[(r["task"], r["method"], r["p"], r["T"])].append(r["acc"])

    with open(args.out, "w") as f:
        f.write("task,method,p,T,n_seeds,mean_acc,std_acc\n")
        for (task, method, p, T), accs in sorted(groups.items()):
            m = mean(accs)
            s = pstdev(accs) if len(accs) > 1 else 0.0
            f.write(f"{task},{method},{p},{T},{len(accs)},{m:.4f},{s:.4f}\n")
    print(f"wrote {args.out}  ({len(groups)} groups, {len(rows)} runs)")


if __name__ == "__main__":
    main()
