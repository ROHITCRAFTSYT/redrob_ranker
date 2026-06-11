# Stage 5 interview prep — questions they WILL ask

You will be asked to defend design choices. Have a real answer for each:

1. "Why no LLM / no embeddings?" -> compute budget + reproducibility + the JD
   problem is preference-encoding, not semantic similarity. (You believe this.)
2. "Your title_fit gives a non-eng candidate ~0 even with great AI skills. Why?"
   -> the JD explicitly calls keyword-stuffing the trap; show the sample_submission
   ranks an HR Manager #1 as the wrong answer.
3. "Why these exact weights?" -> because on my N-candidate gold set, this combo
   maximized composite. Walk them through one weight change and its effect.
   (If you skipped the gold set, you have no answer here. Don't skip it.)
4. "How do you know you're not overfitting to your own labels?" -> stratified
   sample incl. stuffers/honeypots/gems I did NOT cherry-pick; labeled before
   seeing the ranker's order.
5. "Show me a candidate your ranker got wrong." -> have one ready. Knowing your
   system's failure modes reads as competence, not weakness.
6. "Walk me through honeypot detection." -> consistency checks, name two, explain
   why they're impossibilities not just weak profiles.

If your honest answer to #3 or #4 is weak, that's not an interview problem — it's
a "you haven't done the work yet" problem. Fix it before you submit, not the night
before the interview.
