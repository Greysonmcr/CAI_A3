#!/usr/bin/env python3
"""
Held-out self-check (written for this submission, NOT used for tuning).

Scores every turn (not just the last), and adds negative / isolation memory probes.

    python eval_extra/run_heldout.py --mock
    python eval_extra/run_heldout.py --endpoint http://HOST:8000/v1 --model <name>
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def load(name):
    with open(ROOT / "eval_extra" / name) as f:
        return [json.loads(l) for l in f if l.strip()]


def hit(cited, gold):
    return sum(g in cited for g in set(gold)) / len(set(gold))


ANSWERS = []  # every (question, answer, citations), saved for manual review of answer quality


def ask(bot, sid, uid, q):
    r = bot.chat(sid, uid, q)
    ANSWERS.append({"session": sid, "question": q, "answer": r["answer"], "citations": r["citations"]})
    return r


def run_retrieval(bot, verbose=True):
    rows = []
    for it in load("single_turn_heldout.jsonl"):
        c = ask(bot, f"ho-{it['id']}", "ho_user", it["question"])["citations"]
        rows.append(("single-named" if it["named"] else "single-unnamed", hit(c, it["gold_paper_ids"]), c and c[0] in it["gold_paper_ids"]))
        if verbose and rows[-1][1] < 1:
            print(f"  MISS {it['id']}: cited={c} gold={it['gold_paper_ids']}  {it['question'][:70]}")
    for d in load("multi_turn_heldout.jsonl"):
        for i, t in enumerate(d["turns"]):
            c = ask(bot, f"ho-{d['id']}", "ho_user", t["content"])["citations"]
            rows.append(("multi-first" if i == 0 else "multi-followup", hit(c, t["gold_paper_ids"]), c and c[0] in t["gold_paper_ids"]))
            if verbose and rows[-1][1] < 1:
                print(f"  MISS {d['id']} t{i}: cited={c} gold={t['gold_paper_ids']}  {t['content'][:70]}")
    return rows


def run_memory(bot, verbose=True):
    passed = 0
    probes = load("memory_probes_heldout.jsonl")
    for p in probes:
        u = f"user-{p['id']}"
        bot.reset_memory(u)
        for m in p["plant"]:
            bot.chat(f"{p['id']}-s1", u, m)
        tu = f"other-{p['id']}" if p.get("other_user") else u
        ans = ""
        for m in p["test"]:
            ans = ask(bot, f"{p['id']}-s2", tu, m)["answer"]
        mem = " ".join(bot.get_memory(tu)).lower()
        e = p["expected"].lower()
        ok = (e in mem and e in ans.lower()) if p["type"] == "positive" else (e not in mem and e not in ans.lower())
        passed += ok
        if verbose:
            print(f"  {p['id']:<14} {'PASS' if ok else 'FAIL'}  memory={bot.get_memory(u)}")
    return passed, len(probes)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mock", action="store_true")
    ap.add_argument("--endpoint")
    ap.add_argument("--model", default="mock-model")
    ap.add_argument("--config", default="{}", help="extra TABot config as JSON")
    ap.add_argument("--answers", default=None, help="write every answer to this JSONL file")
    a = ap.parse_args()
    if a.mock:
        from mock_endpoint import start_mock_endpoint
        ep, _ = start_mock_endpoint()
    else:
        ep = a.endpoint
    from ta_bot import TABot
    cfg = {"endpoint": ep, "model": a.model, "corpus_path": str(ROOT / "corpus"), "temperature": 0.0, "seed": 42}
    cfg.update(json.loads(a.config))
    bot = TABot(cfg)
    print("RETRIEVAL (held-out)")
    rows = run_retrieval(bot)
    for g in ["single-named", "single-unnamed", "multi-first", "multi-followup"]:
        r = [x for x in rows if x[0] == g]
        print(f"  {g:<16} hit={sum(x[1] for x in r)/len(r):.2f}  top1={sum(bool(x[2]) for x in r)/len(r):.2f}  n={len(r)}")
    print(f"  {'ALL':<16} hit={sum(x[1] for x in rows)/len(rows):.2f}  n={len(rows)}")
    print("MEMORY (held-out)")
    p, n = run_memory(bot)
    print(f"  {p}/{n} passed")
    if a.answers:
        Path(a.answers).parent.mkdir(parents=True, exist_ok=True)
        with open(a.answers, "w") as f:
            for row in ANSWERS:
                f.write(json.dumps(row) + "\n")
        print(f"  answers written to {a.answers}")


if __name__ == "__main__":
    main()
