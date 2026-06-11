"""
honeypot.py — Impossible-profile (honeypot) detection.

The challenge dataset plants ~80 honeypot candidates with *internally
inconsistent* profiles. The spec forces them to relevance tier 0, and a
submission with >10% honeypots in its top 100 is DISQUALIFIED.

DESIGN PRINCIPLE: a honeypot is caught by *consistency checks*, never by
"this person looks unqualified". We only flag profiles that are physically
impossible, not merely weak. Weak candidates are handled by the ranker;
honeypots are removed outright.

>>> THIS IS YOURS TO EXTEND. <<<
I found two signature families below and they catch ~46/100k. The spec says
~80 exist, so there are more signatures (the organizers hint at several).
After you hand-label honeypots from sample_candidates.json, add a check for
every new impossibility you find. Each `is_*` function should stay a pure,
explainable boolean so you can defend it in the Stage 5 interview.
"""
from datetime import date
from typing import Dict, List


def _year(d):
    if not d:
        return None
    try:
        return int(str(d)[:4])
    except (ValueError, TypeError):
        return None


def claims_more_experience_than_career_supports(c: Dict) -> bool:
    """years_of_experience far exceeds the span since their first job started.
    Example from spec: '8 years of experience at a company founded 3 years ago'."""
    yoe = c["profile"].get("years_of_experience", 0)
    starts = [_year(h.get("start_date")) for h in c.get("career_history", [])]
    starts = [s for s in starts if s]
    if not starts:
        return False
    span = date.today().year - min(starts)
    return yoe > span + 3  # 3-yr grace for rounding / pre-listed roles


def expert_skill_never_used(c: Dict) -> bool:
    """'expert' proficiency in a skill with 0 months of usage.
    Example from spec: 'expert proficiency in 10 skills with 0 years used'."""
    return any(
        s.get("proficiency") == "expert" and s.get("duration_months", 0) == 0
        for s in c.get("skills", [])
    )


def tenure_exceeds_time_since_start(c: Dict) -> bool:
    """A single role's duration_months is longer than the time that has
    elapsed since it began — impossible."""
    for h in c.get("career_history", []):
        sy = _year(h.get("start_date"))
        if sy is None:
            continue
        months_since_start = (date.today().year - sy) * 12 + 12  # generous
        if h.get("duration_months", 0) > months_since_start + 12:
            return True
    return False


def career_duration_exceeds_experience(c: Dict) -> bool:
    """Sum of (non-overlapping) role durations wildly exceeds stated YoE.
    Generous threshold so we only catch the impossible, not job-overlap."""
    yoe = c["profile"].get("years_of_experience", 0)
    total_months = sum(h.get("duration_months", 0) for h in c.get("career_history", []))
    return total_months > (yoe * 12) + 60  # 5-yr grace for overlap/rounding


def education_after_career_impossible(c: Dict) -> bool:
    """end_year before start_year in education, or degree finishing before it began."""
    for e in c.get("education", []):
        sy, ey = e.get("start_year"), e.get("end_year")
        if sy and ey and ey < sy:
            return True
    return False


# Register every signature here. Add yours as you find them.
SIGNATURES = [
    ("exp_gt_career_span", claims_more_experience_than_career_supports),
    ("expert_zero_duration", expert_skill_never_used),
    ("tenure_gt_elapsed", tenure_exceeds_time_since_start),
    ("career_gt_experience", career_duration_exceeds_experience),
    ("education_reversed", education_after_career_impossible),
]


def honeypot_reasons(c: Dict) -> List[str]:
    """Return the list of signature names that fired (empty = not a honeypot)."""
    return [name for name, fn in SIGNATURES if fn(c)]


def is_honeypot(c: Dict) -> bool:
    return len(honeypot_reasons(c)) > 0


if __name__ == "__main__":
    import json, sys
    from collections import Counter
    path = sys.argv[1] if len(sys.argv) > 1 else "candidates.jsonl"
    counts = Counter()
    total = flagged = 0
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            c = json.loads(line)
            total += 1
            reasons = honeypot_reasons(c)
            if reasons:
                flagged += 1
                for r in reasons:
                    counts[r] += 1
    print(f"Scanned {total} candidates; flagged {flagged} as honeypots.")
    for name, ct in counts.most_common():
        print(f"  {ct:4d}  {name}")
