"""
build_supplementary.py — Supplementary report on processing time (Experiment E6).

Purpose : Keep the processing-time results out of Chapter 4, since speed is not one of
          the study's objectives, while preserving them, with the same APA formatting
          and the same computed-not-typed numbers, in a separate document.
Function : Builds outputs/evaluation/supplementary/Supplementary_Processing_Time.docx
          from results.json: the per-slice time by tumor class (Table S.1) and the
          time per pipeline stage (Table S.2), each with its discussion. The per-slice
          workbook is written by experiments/appendix.py (write_supplementary).
Notes   : CLI: python -m experiments.chapter4.build_supplementary [--dir outputs/evaluation]
"""
import argparse
import json
import os

import pandas as pd

from experiments.chapter4 import tables as T
from experiments.chapter4.build_chapter4 import REFERENCES, ROOT, Doc, cited, render
from experiments.chapter4.discussion import texts
from experiments.chapter4.tables import f, nz, pct
from experiments.stats import fmt_p


def build(d):
    R = json.load(open(os.path.join(d, "results.json"), encoding="utf-8"))
    E6 = R.get("e6")
    if not E6:
        raise SystemExit("results.json has no timing results; run: python experiments/evaluate.py --timing")
    D = Doc()
    D.tabs = T.build_supplementary(R)
    tb = {k: v["num"] for k, v in D.tabs.items()}
    # The shared discussion module also writes the chapter's paragraphs, which cite
    # chapter table numbers; only the two time paragraphs are used here.
    ch_tb = {k: v["num"] for k, v in T.build_chapter(R).items()}
    disc = texts(R, pd.read_csv(os.path.join(d, "raw_results.csv"), low_memory=False), {**ch_tb, "summary": "", **tb})
    t = E6["all"]
    p = fmt_p(t["p"])

    D.h1("Supplementary Report")
    D.h1("PROCESSING TIME OF THE BASELINE AND ENHANCED ALGORITHM")
    D.p("Processing time is not one of the objectives of the study, so it is reported here rather than in Chapter 4. "
        "It was measured in a separate sequential pass (Experiment E6) under automatic seeding, on "
        f"{t['N']} slices ({t['N'] // 9} per stratum), in one process with nothing else running, after a discarded "
        "warm-up run, with the order of the two algorithms alternated per slice. The time runs from loading the slice "
        "to the final mask and excludes the computation of the evaluation metrics.")
    D.table("time")
    D.p(f"Table {tb['time']} shows the processing time per slice. The median time is {f(t['srg']['median'], 2)} s for "
        f"SRG and {f(t['esrg']['median'], 2)} s for ESRG (Wilcoxon signed-rank test, "
        f"*p* {p if p.startswith('<') else '= ' + p}, *r* = {nz(t['r'])}), a "
        f"{pct(abs(t['reduction_pct_median']))} {'reduction' if t['reduction_pct_median'] > 0 else 'increase'}; ESRG "
        f"was faster on {pct(t['faster_share'])} of slices. " + disc.get("TIME_DISCUSSION", ""))
    D.table("stages")
    D.p(disc.get("STAGE_DISCUSSION", ""))
    D.h2("References")
    body = " ".join(b.get("text", "") + " " + b.get("note", "") for b in D.blocks)
    for r in cited(REFERENCES, body):
        D.ref(r)
    return D


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=os.path.join(ROOT, "outputs", "evaluation"))
    args = ap.parse_args()
    out_dir = os.path.join(args.dir, "supplementary")
    os.makedirs(out_dir, exist_ok=True)
    D = build(args.dir)
    spec = os.path.join(out_dir, "supplementary_spec.json")
    json.dump({"title": "Supplementary Report: Processing Time", "blocks": D.blocks},
              open(spec, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    out = os.path.join(out_dir, "Supplementary_Processing_Time.docx")
    render(spec, out)
    print("wrote", out)
