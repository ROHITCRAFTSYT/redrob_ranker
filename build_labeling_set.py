"""
build_labeling_set.py — Export a candidate sample for you to hand-label.

Builds the gold set evaluate.py needs. This is YOUR judgment work — it is the
single highest-leverage thing only you can do, and it's what you defend in the
interview. Don't outsource it.

Strategy: don't label 50 random candidates (you'd get ~49 obvious non-fits and
learn nothing). Instead pull a STRATIFIED sample that forces you to make the
hard calls the hidden ground truth cares about:
  - your ranker's current top 40 (are they actually good? are any wrong?)
  - a batch of keyword-stuffers (non-eng title + AI skills) — should be 0
  - a batch of honeypots — should be 0
  - a batch of "Tier 5" hidden gems (adjacent eng titles, plain language) — the
    JD says these CAN be strong fits; this is where ranking quality is decided
  - some random profiles for calibration

USAGE:
    python build_labeling_set.py --candidates candidates.jsonl --submission submission.csv

Produces label_these.csv. Open it, read each profile (use show_candidate.py for
detail), and fill the `tier` column with 3/2/1/0. Then convert to gold_labels.json:
    python build_labeling_set.py --to-gold label_these.csv > gold_labels.json
"""
import argparse
import csv
import json
import random

from honeypot import is_honeypot

NONENG = ["hr manager", "sales", "marketing", "accountant", "content writer",
          "graphic designer", "customer support", "operations manager",
          "business analyst", "civil engineer", "mechanical engineer"]
ADJACENT = ["software engineer", "backend", "data engineer", "full stack",
            "cloud engineer", "devops", "platform"]


def load(path):
    out = []
    with open(path) as f:
        for line in f:
            if line.strip():
                out.append(json.loads(line))
    return out


def to_gold(label_csv):
    gold = {}
    with open(label_csv) as f:
        for row in csv.DictReader(f):
            t = row.get("tier", "").strip()
            if t != "":
                gold[row["candidate_id"]] = int(t)
    print(json.dumps(gold, indent=2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", default="candidates.jsonl")
    ap.add_argument("--submission", default="submission.csv")
    ap.add_argument("--to-gold", help="convert a filled label CSV to gold JSON on stdout")
    ap.add_argument("--out", default="label_these.csv")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    if args.to_gold:
        to_gold(args.to_gold)
        return

    random.seed(args.seed)
    cands = load(args.candidates)
    by_id = {c["candidate_id"]: c for c in cands}

    picks = {}  # id -> bucket

    # 1) your current top 40
    top_ids = []
    with open(args.submission) as f:
        for row in csv.DictReader(f):
            top_ids.append(row["candidate_id"])
    for cid in top_ids[:40]:
        picks[cid] = "your_top40"

    stuffers, honeypots, gems, rnd = [], [], [], []
    for c in cands:
        title = c["profile"]["current_title"].lower()
        skills = " ".join(s.get("name", "").lower() for s in c.get("skills", []))
        ai_skilled = any(k in skills for k in ["ml", "ai", "machine learning",
                                               "deep learning", "nlp", "pytorch"])
        if is_honeypot(c):
            honeypots.append(c["candidate_id"])
        elif any(t in title for t in NONENG) and ai_skilled:
            stuffers.append(c["candidate_id"])
        elif any(t in title for t in ADJACENT):
            gems.append(c["candidate_id"])
        else:
            rnd.append(c["candidate_id"])

    for cid in random.sample(stuffers, min(8, len(stuffers))):
        picks.setdefault(cid, "keyword_stuffer")
    for cid in random.sample(honeypots, min(5, len(honeypots))):
        picks.setdefault(cid, "honeypot")
    for cid in random.sample(gems, min(12, len(gems))):
        picks.setdefault(cid, "tier5_gem")
    for cid in random.sample(rnd, min(10, len(rnd))):
        picks.setdefault(cid, "random")

    with open(args.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["candidate_id", "bucket", "current_title", "years",
                    "headline", "tier"])
        for cid, bucket in picks.items():
            c = by_id[cid]
            p = c["profile"]
            w.writerow([cid, bucket, p["current_title"],
                        f"{p['years_of_experience']:.1f}",
                        p["headline"][:60], ""])
    print(f"Wrote {args.out} with {len(picks)} candidates to label.")
    print("Fill the `tier` column (3=strong,2=decent,1=weak,0=no/honeypot),")
    print("then: python build_labeling_set.py --to-gold label_these.csv > gold_labels.json")


if __name__ == "__main__":
    main()
