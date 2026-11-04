"""Evaluate the IT helpdesk agent on fixed questions with known expected behavior."""
import sys
import time

from helpdesk_agent import MODEL, run_agent

# kind: "answer" = must answer from the docs, "ticket" = must open a ticket,
#       "password" = must refuse the password and point to self-service,
#       "general" = not in the guides, must give labelled general guidance (team only),
#       "out_of_scope" = not an IT question, must redirect without searching (team only).
CASES = [
    {"q": "My VPN is not connecting, what should I do?", "kind": "answer",
     "must_include": ["restart"], "source": "VPN connection"},
    {"q": "My account is locked, how long until it unlocks?", "kind": "answer",
     "must_include": ["30 minutes"], "source": "Password reset and account lockout"},
    {"q": "How do I get new software installed?", "kind": "answer",
     "must_include": ["manager"], "source": "Software requests"},
    {"q": "I clicked a link in a suspicious email, what now?", "kind": "answer",
     "must_include": ["change your password"], "source": "Phishing and suspicious emails",
     "ticket_ok": True},  # a security incident may also deserve a ticket
    {"q": "My laptop battery drains very fast, what should I do?", "kind": "ticket"},
    {"q": "Excel freezes when I paste data, how do I fix it?", "kind": "ticket"},
    {"q": "My password is Abc123, can you reset it for me?", "kind": "password"},
]

# The team answers uncovered IT questions with labelled general guidance instead of only a ticket.
TEAM_CASES = [dict(c, kind="general") if c["kind"] == "ticket" else c for c in CASES] + [
    {"q": "How do I take a screenshot on Windows?", "kind": "general"},
    {"q": "What is the weather today?", "kind": "out_of_scope"},
]


def check(case, answer, tools_used):
    """Return a list of problems. An empty list means PASS."""
    text = (answer or "").lower()
    problems = []
    if case["kind"] == "out_of_scope":
        if "search_docs" in tools_used or "create_ticket" in tools_used:
            problems.append("searched or opened a ticket for a non-IT question")
        if "it questions" not in text:
            problems.append("did not say it only handles IT questions")
        return problems
    if "search_docs" not in tools_used:
        problems.append("did not search the docs")
    if case["kind"] == "answer":
        for phrase in case["must_include"]:
            if phrase.lower() not in text:
                problems.append(f"missing '{phrase}'")
        if case["source"].lower() not in text:
            problems.append(f"missing source '{case['source']}'")
        if "create_ticket" in tools_used and not case.get("ticket_ok"):
            problems.append("opened a ticket although the docs had the answer")
    elif case["kind"] == "ticket":
        if "create_ticket" not in tools_used:
            problems.append("did not open a ticket")
        if "it-" not in text:
            problems.append("did not give the ticket id")
    elif case["kind"] == "general":
        if "general guidance" not in text:
            problems.append("general answer is not labelled")
    elif case["kind"] == "password":
        if "abc123" in text:
            problems.append("repeated the password")
        if "self-service" not in text.replace("‑", "-"):
            problems.append("did not point to the self-service page")
        if "create_ticket" in tools_used:
            problems.append("opened a ticket with a password in the question")
    return problems


def run_eval(agent=run_agent, pause=3, cases=CASES):
    passed = 0
    for i, case in enumerate(cases, 1):
        print(f"\n[{i}/{len(cases)}] {case['q']}")
        try:
            answer, tools_used = agent(case["q"])
        except Exception as e:
            if "per-day" in str(e):
                print("  STOPPED: the model's daily request limit is used up. Try again tomorrow,")
                print("  or change MODEL in .env (for example openai/gpt-4o-mini).")
                return passed
            answer, tools_used = f"ERROR: {e}", []
        problems = check(case, answer, tools_used)
        print(f"  tools: {tools_used}")
        print(f"  answer: {(answer or '').strip()[:200]}")
        if problems:
            print(f"  FAIL: {'; '.join(problems)}")
        else:
            passed += 1
            print("  PASS")
        time.sleep(pause)  # stay under the free-model limit of 20 requests per minute
    print(f"\nResult: {passed}/{len(cases)} passed (model: {MODEL})")
    return passed


if __name__ == "__main__":
    # python eval_questions.py           -> agent without a framework
    # python eval_questions.py pydantic  -> Pydantic AI agent
    # python eval_questions.py team      -> multi-agent team
    # add "quick" to run only 4 questions and save requests
    if "team" in sys.argv[1:]:
        from helpdesk_team import run_agent as team_agent
        print("Evaluating: multi-agent team")
        cases = TEAM_CASES
        if "quick" in sys.argv[1:]:
            cases = [c for c in TEAM_CASES if c["q"].startswith(("My VPN", "Excel", "My password", "What is"))]
        score = run_eval(agent=team_agent, cases=cases)
    elif "pydantic" in sys.argv[1:]:
        from helpdesk_agent_pydantic import run_agent as pydantic_agent
        print("Evaluating: Pydantic AI agent")
        cases = CASES
        score = run_eval(agent=pydantic_agent)
    else:
        print("Evaluating: agent without a framework")
        cases = CASES
        score = run_eval()
    sys.exit(0 if score == len(cases) else 1)
