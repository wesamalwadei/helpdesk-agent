"""Compare keyword and semantic search on sample questions, and suggest MIN_SIMILARITY.

Run: python calibrate_search.py
Cost: embeds the 25 guide sections once (saved to embeddings.json) plus one short request
per question, with a small embedding model. No chat model is used.
"""
from dotenv import load_dotenv

load_dotenv()
import tools  # noqa: E402  (after load_dotenv so the API key is available)

# (question, the section that should come first, or None if the guides don't cover it)
QUESTIONS = [
    ("My VPN is not connecting", "VPN connection"),
    ("I can't reach company systems when I work from home", "VPN connection"),
    ("my laptop is super laggy", "Slow computer"),
    ("I typed my password wrong too many times", "Password reset and account lockout"),
    ("people can't hear me in meetings", "Microsoft Teams, camera and microphone"),
    ("I accidentally deleted a document from the team folder", "Restoring deleted files"),
    ("my second screen stays black", "Monitor and docking station"),
    ("I left my work phone in a taxi", "Lost or stolen device"),
    ("someone sent me a weird link asking for my login", "Phishing and suspicious emails"),
    ("my inbox says it is full", "Mailbox full"),
    ("My laptop battery drains very fast", None),
    ("Excel freezes when I paste data", None),
    ("How do I take a screenshot on Windows?", None),
]


def main():
    good, bad, hits_sem, hits_kw = [], [], 0, 0
    for question, expected in QUESTIONS:
        sem = tools.semantic_search(question, top_k=1, min_similarity=0)  # top match, no cut-off
        kw = tools.search_docs(question, top_k=1)
        top, score = (sem[0]["section"], sem[0]["score"]) if sem else ("-", 0)
        kw_top = kw[0]["section"] if kw else "(nothing)"
        if expected:
            good.append(score)
            hits_sem += top == expected
            hits_kw += kw_top == expected
            mark = "OK  " if top == expected else "MISS"
        else:
            bad.append(score)
            mark = "none"
        print(f"{mark} {score:.3f}  {question}\n        semantic: {top}\n        keyword:  {kw_top}")

    covered = sum(1 for _, e in QUESTIONS if e)
    print(f"\nRight section first: semantic {hits_sem}/{covered}, keyword {hits_kw}/{covered}")
    print(f"Scores of covered questions:   lowest {min(good):.3f}")
    print(f"Scores of uncovered questions: highest {max(bad):.3f}")
    if min(good) > max(bad):
        print(f"Suggested MIN_SIMILARITY = {(min(good) + max(bad)) / 2:.2f}")
    else:
        print("Scores overlap: pick a value near the middle; the resolver still checks each answer.")


if __name__ == "__main__":
    main()
