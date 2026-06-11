"""
evaluate.py — Offline evaluation harness.

The real competition score is computed ONCE against a HIDDEN ground truth you
will never see. There is no leaderboard and no feedback. So the only way to
improve before you submit is to evaluate against a gold set YOU build by hand.

This computes the exact composite the organizers use:
    composite = 0.50*NDCG@10 + 0.30*NDCG@50 + 0.15*MAP + 0.05*P@10
plus P@5 (the first tiebreak) and your top-100 honeypot rate (the DQ gate).

USAGE:
    python evaluate.py --submission submission.csv --gold gold_labels.json

gold_labels.json format (YOU create this with build_labeling_set.py + judgment):
    { "CAND_0046525": 3, "CAND_0011687": 3, "CAND_0000123": 0, ... }
    relevance tiers: 3 = strong fit, 2 = decent, 1 = weak, 0 = not a fit/honeypot.
    Candidates you didn't label are treated as relevance 0.

WHY THIS MATTERS FOR STAGE 5: "I tuned my weights against a 40-candidate gold
set I labeled by reading profiles, and watched NDCG@10 move" is a sentence that
wins the interview. "I picked the weights that felt right" loses it.
"""
import argparse
import csv
import json
import math


def dcg(rels):
    return sum((2 ** r - 1) / math.log2(i + 2) for i, r in enumerate(rels))


def ndcg_at_k(ranked_rels, k):
    ideal = sorted(ranked_rels, reverse=True)
    idcg = dcg(ideal[:k])
    return dcg(ranked_rels[:k]) / idcg if idcg > 0 else 0.0


def average_precision(ranked_rels, positive_threshold=1):
    """AP treating relevance >= threshold as 'relevant'."""
    hits = 0
    summ = 0.0
    n_rel = sum(1 for r in ranked_rels if r >= positive_threshold)
    if n_rel == 0:
        return 0.0
    for i, r in enumerate(ranked_rels):
        if r >= positive_threshold:
            hits += 1
            summ += hits / (i + 1)
    return summ / n_rel


def precision_at_k(ranked_rels, k, positive_threshold=1):
    top = ranked_rels[:k]
    return sum(1 for r in top if r >= positive_threshold) / k if k else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--submission", default="submission.csv")
    ap.add_argument("--gold", required=True, help="JSON dict candidate_id -> relevance tier")
    ap.add_argument("--candidates", default="candidates.jsonl",
                    help="used only for the honeypot-rate check (optional)")
    args = ap.parse_args()

    gold = json.load(open(args.gold))

    ranked_ids = []
    with open(args.submission) as f:
        for row in csv.DictReader(f):
            ranked_ids.append(row["candidate_id"])
    ranked_rels = [int(gold.get(cid, 0)) for cid in ranked_ids]

    n10 = ndcg_at_k(ranked_rels, 10)
    n50 = ndcg_at_k(ranked_rels, 50)
    mapv = average_precision(ranked_rels)
    p10 = precision_at_k(ranked_rels, 10)
    p5 = precision_at_k(ranked_rels, 5)
    composite = 0.50 * n10 + 0.30 * n50 + 0.15 * mapv + 0.05 * p10

    print(f"  NDCG@10 : {n10:.4f}")
    print(f"  NDCG@50 : {n50:.4f}")
    print(f"  MAP     : {mapv:.4f}")
    print(f"  P@10    : {p10:.4f}")
    print(f"  P@5     : {p5:.4f}  (first tiebreak)")
    print(f"  -------------------------------------")
    print(f"  COMPOSITE: {composite:.4f}")
    print(f"  (= 0.50*NDCG@10 + 0.30*NDCG@50 + 0.15*MAP + 0.05*P@10)")

    n_labeled = sum(1 for cid in ranked_ids if cid in gold)
    print(f"\n  labeled coverage in top 100: {n_labeled}/100"
          f"  (unlabeled treated as 0 — label more for a sharper signal)")

    # honeypot-rate gate
    try:
        from honeypot import is_honeypot
        idset = set(ranked_ids)
        hp = 0
        with open(args.candidates) as f:
            for line in f:
                if line.strip():
                    c = json.loads(line)
                    if c["candidate_id"] in idset and is_honeypot(c):
                        hp += 1
        flag = "  <-- DISQUALIFIED" if hp > 10 else "  OK"
        print(f"  honeypots in top 100: {hp}/100 (DQ if >10){flag}")
    except FileNotFoundError:
        pass


if __name__ == "__main__":
    main()
