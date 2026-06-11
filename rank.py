"""
rank.py — Intelligent Candidate Ranker for the Redrob "Senior AI Engineer" JD.

Reproduce command (must finish < 5 min, CPU-only, no network):
    python rank.py --candidates ./candidates.jsonl --out ./submission.csv

ARCHITECTURE (why it's built this way — defend this in Stage 5):

  The JD is engineered to defeat keyword matching. It explicitly states the
  wrong answer is "most AI keywords" and lists, in plain English, exactly what
  it does and does NOT want. So the core of this ranker is not a model — it's a
  faithful *encoding of the JD's stated preferences*, with lexical similarity as
  a supporting signal, not the driver. Components per candidate:

    relevance = w_lex   * lexical_fit        (TF-IDF cosine JD vs profile text)
              + w_title * title_fit          (is this an ML/AI/IR engineer? — the
                                              decisive anti-keyword-stuffer signal)
              + w_career* career_fit         (retrieval/ranking/recsys/NLP built at
                                              a PRODUCT company; 5-9 yr band)
              + w_skill * skill_trust        (JD-relevant skills weighted by
                                              endorsements + duration — kills
                                              0-duration keyword stuffing)
              + w_loc   * location_fit       (India Tier-1 or relocate; JD: no visa)
              - penalties                    (JD's explicit disqualifiers)

    final = relevance * behavioral_modifier  (availability: a perfect-on-paper
                                              candidate dormant 6 months with 5%
                                              response rate is NOT hireable)

  Honeypots (impossible profiles) are removed before ranking — see honeypot.py.

  >>> THE WEIGHTS BELOW ARE STARTING VALUES, NOT GOSPEL. <<<
  Tune them against YOUR hand-labeled gold set with evaluate.py. You must be able
  to explain every weight in the interview. If you can't justify a number, change
  it until you can.

  No LLM is called. No GPU. TF-IDF over 100k short docs fits the CPU/5-min budget
  with room to spare and reproduces deterministically in a clean Docker container.
"""
import argparse
import csv
import json
import re
from datetime import date

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

from honeypot import is_honeypot

# ============================================================================
# CONFIG — YOURS TO OWN. Every value here is a design decision you must defend.
# ============================================================================
WEIGHTS = {
    "lex": 0.20,     # lexical JD overlap — supporting signal, deliberately low
    "title": 0.30,   # title/role fit — the decisive anti-stuffer signal
    "career": 0.25,  # what they actually BUILT, at what kind of company
    "skill": 0.15,   # trusted (endorsed + used) JD-relevant skills
    "loc": 0.10,     # location / relocate
}

# The JD, distilled to a query string for lexical matching. (Paraphrase of the
# released JD — keep it in YOUR words; do not paste the JD verbatim into the repo.)
JD_QUERY = (
    "senior ai ml engineer embeddings retrieval ranking recommendation search "
    "vector database hybrid search nlp information retrieval learning to rank "
    "evaluation ndcg mrr production deployment applied machine learning python "
    "product company real users scale"
)

# Title fit lookup (lowercased substring match against current + recent titles).
TITLE_STRONG = ["ml engineer", "machine learning engineer", "ai engineer",
                "applied ml", "applied scientist", "nlp engineer", "data scientist",
                "research engineer", "software engineer (ml)", "ai specialist",
                "ai research"]
TITLE_ADJACENT = ["software engineer", "backend engineer", "data engineer",
                  "full stack", "cloud engineer", "devops", "platform engineer"]
# Non-engineering titles: a keyword-stuffer here is the classic trap → ~0 title fit.
TITLE_NONENG = ["hr manager", "sales", "marketing", "accountant", "content writer",
                "graphic designer", "customer support", "operations manager",
                "business analyst", "civil engineer", "mechanical engineer",
                "project manager"]

# JD-relevant skills the ranker trusts (only when endorsed AND used).
RELEVANT_SKILLS = {"python", "pytorch", "tensorflow", "embeddings", "transformers",
                   "nlp", "information retrieval", "elasticsearch", "faiss",
                   "pinecone", "weaviate", "qdrant", "milvus", "bm25", "ranking",
                   "recommendation", "recsys", "vector search", "machine learning",
                   "deep learning", "xgboost", "learning to rank", "semantic search",
                   "sentence transformers", "bge", "retrieval"}

# JD-named retrieval/ranking evidence to look for in career descriptions.
CAREER_EVIDENCE = ["retrieval", "ranking", "recommendation", "recsys", "search",
                   "embedding", "vector", "nlp", "information retrieval",
                   "semantic", "learning to rank", "personalization", "relevance",
                   "matching", "recommender"]

# JD explicit disqualifiers → penalties (subtracted from relevance, pre-modifier).
CONSULTING_FIRMS = ["tcs", "infosys", "wipro", "accenture", "cognizant",
                    "capgemini", "hcl", "tech mahindra", "mindtree", "ltimindtree",
                    "mphasis", "ibm services"]
PENALTY = {
    "consulting_only": 0.35,   # career entirely at services firms (JD: "bad fit")
    "title_chaser": 0.20,      # avg tenure < 18mo across many jobs (JD: not a fit)
    "pure_research": 0.25,     # research-only, no production-deployment language
    "cv_speech_only": 0.20,    # CV/speech/robotics with no NLP/IR exposure
    "langchain_only": 0.15,    # only recent LLM-wrapper work, no pre-LLM ML
}

IDEAL_EXP = (5, 9)            # JD's stated band ("a range, not a requirement")
INDIA_TIER1 = ["pune", "noida", "bangalore", "bengaluru", "hyderabad", "mumbai",
               "delhi", "gurgaon", "gurugram", "ncr", "chennai"]


# ============================================================================
# Feature extraction
# ============================================================================
def profile_text(c):
    p = c["profile"]
    parts = [p.get("headline", ""), p.get("summary", ""), p.get("current_title", "")]
    parts += [h.get("title", "") + " " + h.get("description", "")
              for h in c.get("career_history", [])]
    parts += [s.get("name", "") for s in c.get("skills", [])]
    return " ".join(parts).lower()


def title_fit(c):
    titles = [c["profile"].get("current_title", "")]
    titles += [h.get("title", "") for h in c.get("career_history", [])[:3]]
    blob = " ".join(titles).lower()
    if any(t in blob for t in TITLE_STRONG):
        return 1.0
    if any(t in blob for t in TITLE_ADJACENT):
        return 0.55          # adjacent eng — could be a "Tier 5" hidden gem
    if any(t in blob for t in TITLE_NONENG):
        return 0.05          # keyword-stuffer trap: non-eng title, ignore skills
    return 0.25


def career_fit(c):
    descs = " ".join(h.get("description", "") for h in c.get("career_history", [])).lower()
    evidence = sum(1 for kw in CAREER_EVIDENCE if kw in descs)
    evidence_score = min(evidence / 4.0, 1.0)         # cap at 4 distinct hits
    # product vs services company signal
    companies = " ".join(h.get("company", "") for h in c.get("career_history", [])).lower()
    services = sum(1 for f in CONSULTING_FIRMS if f in companies)
    n_jobs = max(len(c.get("career_history", [])), 1)
    product_score = 1.0 - min(services / n_jobs, 1.0)
    # experience band fit (soft, triangular around 5-9)
    yoe = c["profile"].get("years_of_experience", 0)
    if IDEAL_EXP[0] <= yoe <= IDEAL_EXP[1]:
        band = 1.0
    elif yoe < IDEAL_EXP[0]:
        band = max(0.0, yoe / IDEAL_EXP[0])
    else:
        band = max(0.0, 1.0 - (yoe - IDEAL_EXP[1]) / 8.0)
    return 0.5 * evidence_score + 0.3 * product_score + 0.2 * band


def skill_trust(c):
    """JD-relevant skills, but only TRUSTED when endorsed and actually used.
    This is what defeats the 'expert in 10 AI skills, 0 endorsements' stuffer."""
    score = 0.0
    for s in c.get("skills", []):
        name = s.get("name", "").lower()
        if not any(rs in name or name in rs for rs in RELEVANT_SKILLS):
            continue
        prof = {"beginner": 0.3, "intermediate": 0.6,
                "advanced": 0.85, "expert": 1.0}.get(s.get("proficiency"), 0.3)
        endorsed = min(s.get("endorsements", 0) / 20.0, 1.0)
        used = min(s.get("duration_months", 0) / 24.0, 1.0)
        trust = (endorsed + used) / 2.0       # 0 if never endorsed AND never used
        score += prof * trust
    return min(score / 4.0, 1.0)              # ~4 trusted relevant skills = full


def location_fit(c):
    loc = (c["profile"].get("location", "") + " " + c["profile"].get("country", "")).lower()
    sig = c.get("redrob_signals", {})
    if any(city in loc for city in INDIA_TIER1):
        return 1.0
    if "india" in loc:
        return 0.8
    if sig.get("willing_to_relocate"):
        return 0.6
    return 0.2                                # JD: outside India case-by-case, no visa


def penalties(c):
    total = 0.0
    fired = []
    hist = c.get("career_history", [])
    companies = " ".join(h.get("company", "") for h in hist).lower()
    n_jobs = max(len(hist), 1)
    services = sum(1 for f in CONSULTING_FIRMS if f in companies)
    if n_jobs >= 2 and services >= n_jobs:
        total += PENALTY["consulting_only"]; fired.append("consulting_only")
    # title chaser: many jobs, short average tenure
    durations = [h.get("duration_months", 0) for h in hist]
    if len(durations) >= 4 and (sum(durations) / len(durations)) < 18:
        total += PENALTY["title_chaser"]; fired.append("title_chaser")
    text = profile_text(c)
    if ("research" in text and "phd" in text
            and not any(w in text for w in ["production", "deployed", "shipped", "users"])):
        total += PENALTY["pure_research"]; fired.append("pure_research")
    if (any(w in text for w in ["computer vision", "speech recognition", "robotics"])
            and not any(w in text for w in ["nlp", "retrieval", "information retrieval", "ranking"])):
        total += PENALTY["cv_speech_only"]; fired.append("cv_speech_only")
    if ("langchain" in text
            and not any(w in text for w in ["pre-llm", "deep learning", "xgboost",
                                            "embeddings", "retrieval", "production ml"])):
        total += PENALTY["langchain_only"]; fired.append("langchain_only")
    return total, fired


def behavioral_modifier(c):
    """Multiplier in ~[0.45, 1.10] capturing whether the candidate is actually
    reachable & hireable. JD/signals doc: down-weight the dormant & unresponsive."""
    s = c.get("redrob_signals", {})
    resp = s.get("recruiter_response_rate", 0.0)              # 0-1
    open_flag = 1.0 if s.get("open_to_work_flag") else 0.0
    interview = s.get("interview_completion_rate", 0.0)       # 0-1
    # recency of last activity
    try:
        last = date.fromisoformat(s.get("last_active_date"))
        days = (date.today() - last).days
        recency = max(0.0, 1.0 - days / 180.0)               # 0 if dormant 6mo+
    except (TypeError, ValueError):
        recency = 0.3
    saved = min(s.get("saved_by_recruiters_30d", 0) / 10.0, 1.0)
    avail = (0.35 * resp + 0.25 * recency + 0.20 * open_flag
             + 0.12 * interview + 0.08 * saved)
    return 0.45 + 0.65 * avail                                # floor 0.45, ceiling ~1.10


# ============================================================================
# Reasoning — specific, honest, from real fields only (Stage 4 manual review).
# ============================================================================
def make_reasoning(c, comps):
    p = c["profile"]
    yoe = p.get("years_of_experience", 0)
    title = p.get("current_title", "candidate")
    bits = [f"{title}, {yoe:.1f} yrs"]
    descs = " ".join(h.get("description", "") for h in c.get("career_history", [])).lower()
    hits = [kw for kw in ["retrieval", "ranking", "recommendation", "search",
                          "nlp", "embedding", "vector"] if kw in descs]
    if hits:
        bits.append("built " + "/".join(hits[:2]) + " systems")
    if comps["penalty_fired"]:
        bits.append("concern: " + ", ".join(comps["penalty_fired"]))
    resp = c.get("redrob_signals", {}).get("recruiter_response_rate", 0)
    if comps["behavioral"] < 0.7:
        bits.append(f"low availability (response {resp:.0%})")
    elif resp >= 0.5:
        bits.append(f"responsive ({resp:.0%})")
    return "; ".join(bits) + "."


# ============================================================================
# Main
# ============================================================================
def load(path):
    out = []
    with open(path) as f:
        for line in f:
            if line.strip():
                out.append(json.loads(line))
    return out


def rank(candidates):
    # 1) drop honeypots entirely
    pool = [c for c in candidates if not is_honeypot(c)]

    # 2) lexical fit via TF-IDF (JD as one doc, all profiles as the corpus)
    texts = [profile_text(c) for c in pool]
    vec = TfidfVectorizer(max_features=20000, ngram_range=(1, 2),
                          stop_words="english", sublinear_tf=True)
    mat = vec.fit_transform(texts + [JD_QUERY])
    jd_vec = mat[-1]
    lex = linear_kernel(jd_vec, mat[:-1]).ravel()
    lex = lex / (lex.max() + 1e-9)                  # normalize to 0-1

    # 3) per-candidate structured scoring
    scored = []
    for i, c in enumerate(pool):
        t = title_fit(c); ca = career_fit(c); sk = skill_trust(c); lo = location_fit(c)
        pen, fired = penalties(c)
        relevance = (WEIGHTS["lex"] * lex[i] + WEIGHTS["title"] * t
                     + WEIGHTS["career"] * ca + WEIGHTS["skill"] * sk
                     + WEIGHTS["loc"] * lo) - pen
        relevance = max(relevance, 0.0)
        beh = behavioral_modifier(c)
        final = relevance * beh
        comps = {"lex": lex[i], "title": t, "career": ca, "skill": sk,
                 "loc": lo, "penalty": pen, "penalty_fired": fired, "behavioral": beh}
        scored.append((final, c, comps))

    # 4) sort: score desc, then candidate_id asc (validator tie-break rule)
    scored.sort(key=lambda x: (-x[0], x[1]["candidate_id"]))
    return scored[:100]


def write_csv(top, out_path):
    # rescale scores into a clean non-increasing 0-1 band for readability
    raw = [s for s, _, _ in top]
    hi, lo = max(raw), min(raw)
    rng = (hi - lo) or 1.0
    items = []
    for score, c, comps in top:
        disp = round(0.30 + 0.69 * (score - lo) / rng, 4)   # map into [0.30, 0.99]
        items.append((disp, c["candidate_id"], make_reasoning(c, comps)))
    # Validator rule: score non-increasing AND equal scores ordered by id ascending.
    # Sorting by (-display, candidate_id) guarantees both simultaneously.
    items.sort(key=lambda x: (-x[0], x[1]))
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["candidate_id", "rank", "score", "reasoning"])
        for rank_i, (disp, cid, reason) in enumerate(items, start=1):
            w.writerow([cid, rank_i, f"{disp:.4f}", reason])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", default="candidates.jsonl")
    ap.add_argument("--out", default="submission.csv")
    args = ap.parse_args()
    import time
    t0 = time.time()
    cands = load(args.candidates)
    top = rank(cands)
    write_csv(top, args.out)
    print(f"Ranked {len(cands)} candidates -> top 100 in {time.time()-t0:.1f}s")
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
