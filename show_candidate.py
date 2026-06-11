"""show_candidate.py — Pretty-print a candidate's full profile for hand-labeling.
USAGE: python show_candidate.py CAND_0046525 [CAND_...]"""
import json, sys

def load(path="candidates.jsonl"):
    d = {}
    with open(path) as f:
        for line in f:
            if line.strip():
                c = json.loads(line); d[c["candidate_id"]] = c
    return d

def show(c):
    p, s = c["profile"], c["redrob_signals"]
    print("="*70)
    print(f"{c['candidate_id']}  |  {p['current_title']}  |  {p['years_of_experience']}y")
    print(f"{p['location']}, {p['country']}  |  {p['current_company']} ({p['current_industry']})")
    print(f"HEADLINE: {p['headline']}")
    print(f"SUMMARY: {p['summary']}")
    print("-- CAREER --")
    for h in c["career_history"]:
        print(f"  {h['title']} @ {h['company']} ({h['company_size']}, {h['industry']}) "
              f"{h['start_date']}->{h['end_date']} [{h['duration_months']}mo]")
        print(f"     {h['description'][:200]}")
    print("-- SKILLS --")
    print("  " + ", ".join(f"{x['name']}({x['proficiency']},e{x['endorsements']},"
                           f"{x.get('duration_months',0)}mo)" for x in c.get("skills", [])))
    print("-- SIGNALS --")
    print(f"  response_rate={s['recruiter_response_rate']} last_active={s['last_active_date']} "
          f"open_to_work={s['open_to_work_flag']} relocate={s['willing_to_relocate']} "
          f"interview_completion={s['interview_completion_rate']} github={s['github_activity_score']}")

if __name__ == "__main__":
    db = load()
    for cid in sys.argv[1:]:
        if cid in db: show(db[cid])
        else: print(f"{cid} not found")
