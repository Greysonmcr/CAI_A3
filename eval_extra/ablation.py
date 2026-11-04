#!/usr/bin/env python3
"""
R4 ablation — one axis: which retrievers are fused.

  chunk          FTS5 BM25 over chunks (the kit's baseline)
  paper          BM25 over title + abstract
  chunk+paper    RRF fusion of the two
  full           chunk + paper + alias matcher (submitted default)

Metric: paper-level hit@k (cite exactly the top-k papers), k = 1..5, over every
turn of the dev set (20 turns) and the held-out set (77 turns). Runs on the mock
endpoint, so LLM query rewriting is OFF for all variants (rewrites from the mock are
rejected) — this isolates the retriever. 95% CIs by bootstrap over turns.

    python eval_extra/ablation.py            # writes eval_extra/ablation.csv + ablation.png
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from mock_endpoint import start_mock_endpoint  # noqa: E402
from ta_bot import TABot  # noqa: E402

VARIANTS = {
    "chunk BM25 (baseline)": ["chunk"],
    "paper BM25": ["paper"],
    "chunk + paper": ["chunk", "paper"],
    "chunk + paper + alias (ours)": ["chunk", "paper", "alias"],
}
KS = [1, 2, 3, 4, 5]


def dialogues():
    """Every evaluated turn as (split, dialogue turns, gold per turn)."""
    out = []
    for f, split in [(ROOT / "dev_eval/single_turn.jsonl", "dev"), (ROOT / "eval_extra/single_turn_heldout.jsonl", "heldout")]:
        for l in open(f):
            d = json.loads(l)
            out.append((split, [d["question"]], [d["gold_paper_ids"]]))
    for f, split in [(ROOT / "dev_eval/multi_turn.jsonl", "dev"), (ROOT / "eval_extra/multi_turn_heldout.jsonl", "heldout")]:
        for l in open(f):
            d = json.loads(l)
            out.append((split, [t["content"] for t in d["turns"]], [t["gold_paper_ids"] for t in d["turns"]]))
    return out


def main():
    ep, _ = start_mock_endpoint()
    data = dialogues()
    rng = np.random.default_rng(0)
    rows, curves = [], {}
    for name, rankers in VARIANTS.items():
        curves[name] = {}
        for k in KS:
            bot = TABot({"endpoint": ep, "model": "mock", "corpus_path": str(ROOT / "corpus"), "temperature": 0.0,
                         "seed": 42, "rankers": rankers, "max_citations": k, "cite_ratio": 0.0, "rewrite": "off",
                         "llm_memory": False})
            hits = {"dev": [], "heldout": []}
            for di, (split, turns, golds) in enumerate(data):
                for t, g in zip(turns, golds):
                    c = bot.chat(f"abl-{di}", "abl", t)["citations"]
                    hits[split].append(sum(p in c for p in set(g)) / len(set(g)))
            allh = np.array(hits["dev"] + hits["heldout"])
            boots = [rng.choice(allh, len(allh)).mean() for _ in range(2000)]
            lo, hi = np.percentile(boots, [2.5, 97.5])
            curves[name][k] = (allh.mean(), lo, hi)
            rows.append({"variant": name, "k": k, "dev": np.mean(hits["dev"]), "heldout": np.mean(hits["heldout"]),
                         "all": allh.mean(), "ci_lo": lo, "ci_hi": hi, "n": len(allh)})
            print(f"{name:<30} k={k}  dev={np.mean(hits['dev']):.3f}  heldout={np.mean(hits['heldout']):.3f}  "
                  f"all={allh.mean():.3f} [{lo:.3f}, {hi:.3f}]  n={len(allh)}")
    with open(ROOT / "eval_extra/ablation.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6, 3.8))
    for (name, c), marker in zip(curves.items(), "osD^"):
        m = [c[k][0] for k in KS]
        ax.plot(KS, m, marker=marker, label=name)
        ax.fill_between(KS, [c[k][1] for k in KS], [c[k][2] for k in KS], alpha=0.12)
    ax.axhline(0.70, color="gray", ls="--", lw=1)
    ax.text(5, 0.705, "pass threshold", ha="right", va="bottom", fontsize=8, color="gray")
    ax.set_xlabel("papers cited (k)")
    ax.set_ylabel("paper-level hit@k")
    ax.set_xticks(KS)
    ax.set_ylim(0.4, 1.01)
    ax.set_title(f"Retriever ablation (dev + held-out, n={rows[0]['n']} turns, 95% bootstrap CI)", fontsize=9)
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(ROOT / "eval_extra/ablation.png", dpi=200)
    print("wrote eval_extra/ablation.csv, eval_extra/ablation.png")


if __name__ == "__main__":
    main()
