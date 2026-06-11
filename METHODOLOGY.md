# Methodology (deck source — convert to PDF for submission)

> Turn this into your 6–10 slide PDF. Every claim here is something you must be
> able to defend live in the Stage 5 interview. If you can't defend a line, cut
> or change it until you can.

## 1. The problem as I read it
- 100k pool, ONE JD (Senior AI Engineer, founding team, 5–9 yrs).
- The JD explicitly says "most AI keywords" is the WRONG answer. The dataset
  plants keyword-stuffers (non-eng titles + AI skills), Tier-5 plain-language
  gems, behavioral twins, and ~80 honeypots.
- Scoring is top-heavy: 0.50*NDCG@10 + 0.30*NDCG@50 + 0.15*MAP + 0.05*P@10.
  => getting the top ~15 right is most of the score.
- Hard constraints: CPU-only, no network during ranking, < 5 min, no LLM calls.

## 2. Why NOT an LLM re-ranker / big embeddings
- Banned by the compute budget; an LLM-per-candidate cannot make 100k in 5 min CPU.
- A transparent feature ranker is also more DEFENSIBLE and reproduces exactly in
  the Stage 3 Docker sandbox.  [<- this is a strength, say it out loud]

## 3. Architecture
relevance = w_lex*lexical + w_title*title_fit + w_career*career_fit
          + w_skill*skill_trust + w_loc*location_fit - penalties
final     = relevance * behavioral_modifier
- title_fit is the anti-keyword-stuffer lever (non-eng title -> ~0).
- skill_trust requires endorsements AND usage -> kills 0-duration stuffing.
- penalties encode the JD's literal disqualifier list.
- behavioral_modifier down-weights the dormant/unresponsive (not actually hireable).
- honeypots removed by consistency checks (impossible, not merely weak).

## 4. How I validated WITHOUT a leaderboard
- Hand-labeled a stratified gold set of N candidates (independently of the ranker).
- Tuned weights to maximize composite on that gold set.
- [Fill in: your gold-set size, what moved when you changed weights, your final
  weights and WHY.]  <- judges care about this more than the model.

## 5. Honest limitations
- Honeypot detector catches ~65/~80; [did you extend it? to what?].
- Lexical signal is TF-IDF, not dense embeddings (deliberate, for the budget).
- [Anything else you'd fix with more time.]
