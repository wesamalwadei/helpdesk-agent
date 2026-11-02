"""Evaluate the IT helpdesk agent on fixed questions with known expected behavior."""
import sys
import time

from helpdesk_agent import MODEL, run_agent

# kind: "answer" = must answer from the docs, "ticket" = must open a ticket,
#       "password" = must refuse the password and point to self-service.
CASES = [
    {"q": "My VPN is not connecting, what should I do?", "kind": "answer",
     "must_include": ["restart"], "source": "VPN connection"},
    {"q": "My account is locked, how long until it unlocks?", "kind": "answer",
     "must_include": ["30 minutes"], "source": "Password reset and account lockout"},
    {"q": "How do I get new software installed?", "kind": "answer",
     "must_include": ["manager"], "source": "Software requests"},
    {"q": "I clicked a link in a suspicious email, what now?", "kind": "answer",
     "must_include": ["change your password"], "source": "Phishing and suspicious emails"},
    {"q": "How do I restore deleted files from the shared folder?", "kind": "ticket"},
    {"q": "My laptop shows a blue screen after an update, how do I fix it?", "kind": "ticket"},
    {"q": "My password is Abc123, can you reset it for me?", "kind": "password"},
]


def check(case, answer, tools_used):
    """Return a list of problems. An empty list means PASS."""
    text = (answer or "").lower()
    problems = []
    if "search_docs" not in tools_used:
        problems.append("did not search the docs")
    if case["kind"] == "answer":
        for phrase in case["must_include"]:
            if phrase.lower() not in text:
                problems.append(f"missing '{phrase}'")
        if case["source"].lower() not in text:
            problems.append(f"missing source '{case['source']}'")
        if "create_ticket" in tools_used:
            problems.append("opened a ticket although the docs had the answer")
    elif case["kind"] == "ticket":
        if "create_ticket" not in tools_used:
            problems.append("did not open a ticket")
        if "it-" not in text:
            problems.append("did not give the ticket id")
    elif case["kind"] == "password":
        if "abc123" in text:
            problems.append("repeated the password")
        if "self-service" not in text.replace("‑", "-"):
            problems.append("did not point to the self-service page")
        if "create_ticket" in tools_used:
            problems.append("opened a ticket with a password in the question")
    return problems


def run_eval(agent=run_agent, pause=3):
    passed = 0
    for i, case in enumerate(CASES, 1):
        print(f"\n[{i}/{len(CASES)}] {case['q']}")
        try:
            answer, tools_used = agent(case["q"])
        except Exception as e:
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
    print(f"\nResult: {passed}/{len(CASES)} passed (model: {MODEL})")
    return passed


if __name__ == "__main__":
    sys.exit(0 if run_eval() == len(CASES) else 1)
