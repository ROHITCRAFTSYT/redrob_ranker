# Redrob Intelligent Candidate Ranker

Ranks the top 100 candidates from a 100k pool for the released
"Senior AI Engineer — Founding Team" JD. CPU-only, no network, < 5 min.

## Reproduce the submission
```bash
pip install -r requirements.txt
python rank.py --candidates ./candidates.jsonl --out ./submission.csv
python validate_submission.py submission.csv   # must print "Submission is valid."
```
Runtime: ~50–70s for 100k candidates on a CPU laptop. No GPU, no API calls.

## What's here
| File | Purpose |
|---|---|
| `rank.py` | The ranker. Encodes the JD's stated preferences + lexical fit + behavioral availability; excludes honeypots. |
| `honeypot.py` | Impossible-profile detection (consistency checks). The >10%-in-top-100 DQ gate. |
| `evaluate.py` | Offline NDCG@10/@50 / MAP / P@10 / P@5 + composite against YOUR gold set. |
| `build_labeling_set.py` | Builds a stratified sample for you to hand-label. |
| `show_candidate.py` | Pretty-prints a full profile for labeling. |

## How to actually improve it (do this in order)
1. `python build_labeling_set.py` -> open `label_these.csv` -> label `tier` (3/2/1/0)
   by reading profiles with `show_candidate.py`. Label INDEPENDENTLY of the
   ranker's order.
2. `python build_labeling_set.py --to-gold label_these.csv > gold_labels.json`
3. `python evaluate.py --submission submission.csv --gold gold_labels.json`
4. Change ONE weight in `rank.py` (the `WEIGHTS` / `PENALTY` dicts), re-run,
   re-evaluate. Keep changes that raise the composite. This is your whole loop.
5. Extend `honeypot.py` with any new impossibility you find while labeling.

## Approach (1 paragraph — REWRITE IN YOUR OWN WORDS before submitting)
The JD is built to defeat keyword matching, so the ranker is primarily a faithful
encoding of the JD's explicit preferences and disqualifiers, with TF-IDF lexical
similarity as a supporting signal. Per candidate: a relevance score combines
lexical fit, title/role fit (the decisive anti-keyword-stuffer signal), career
evidence of retrieval/ranking/recsys at product companies, trust-weighted skills
(endorsed + used), and location; minus penalties for the JD's named disqualifiers
(services-only careers, title-chasing, pure research, etc.). That relevance is
multiplied by a behavioral-availability modifier so dormant/unresponsive
candidates are down-weighted. Honeypots are removed via consistency checks.
