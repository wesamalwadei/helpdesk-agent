# IT Helpdesk Agent

An AI helpdesk for the IT department. It answers employees' IT questions from the company
IT guides and cites the source, gives clearly labelled general guidance when the guides
don't cover a question, and opens a ticket with a priority when a technician is needed.

> Demo project. The IT guides in `it_docs.md` (25 topics) are sample content, not real company procedures.

## The problem

Answers to recurring IT questions (VPN, passwords, printers, software requests) are spread
across guides that are hard to find. Employees lose time searching, and the IT helpdesk
answers the same questions every day.

## Three versions

| | 1. Without framework | 2. Pydantic AI | 3. Multi-agent team |
|---|---|---|---|
| File | `helpdesk_agent.py` | `helpdesk_agent_pydantic.py` | `helpdesk_team.py` |
| Agents | 1 | 1 | 3 (triage, resolver, ticket writer) |
| Who searches the guides | The model (tool call) | The model (tool call) | The code, once per question |
| Not in the guides | Opens a ticket | Opens a ticket | Labelled general guidance, plus a ticket if a technician is needed |
| Ticket fields | Summary, details | Summary, details | Summary, details, **priority, category** |
| Non-IT questions | — | — | Politely refused |
| Follow-ups ("can you open a ticket?") | — | — | Conversation memory (last 6 messages) |
| Invented sources | — | — | Rejected by an output validator |

Versions 1 and 2 show the same agent built by hand and with a framework.
Version 3 is the most complete one and is the default in the web interface.

## How the multi-agent team works

```
Employee question
   │
   ▼
triage_agent ── not IT? ──► polite refusal (1 request)
   │  category, priority, password shared?
   ▼
search_docs (code, once)
   │
   ▼
resolver_agent ──► "guide" answer + source   (must be a real section: checked by an output validator)
   │           └─► "general" answer, labelled "General guidance (not from the company IT guides)"
   ▼
needs a technician? ──► ticket_agent writes the ticket ──► create_ticket (priority + category)
```

Safety rules:

- A "guide" answer must name a real section of the guides; an invented source is rejected
  and the model must fix it (Pydantic AI `output_validator` + `ModelRetry`).
- General answers never suggest risky steps (registry, disabling antivirus or firewall, BIOS, admin commands).
- A shared password is never repeated, the employee is always warned not to share it,
  and the message never reaches the ticket writer.
- Guide answers open a ticket only for high-priority cases (for example phishing).
- The employee can ask for a ticket at any time ("can you open a ticket?"): triage spots it and
  the ticket is written from the conversation. A message that contained a password is never
  passed on from the conversation memory.

Saving requests: the code searches the guides once and passes the results to the resolver,
and each agent has a `max_tokens` limit. A question costs 1 to 3 model requests.

## Project files

| File | Purpose |
|---|---|
| `it_docs.md` | Sample IT guides (25 topics) |
| `tools.py` | `search_docs` (keyword search) and `create_ticket` |
| `helpdesk_agent.py` | Version 1: agent without a framework |
| `helpdesk_agent_pydantic.py` | Version 2: agent built with Pydantic AI |
| `helpdesk_team.py` | Version 3: multi-agent team |
| `app.py` | Streamlit web interface (all three versions) |
| `eval_questions.py` | Evaluation on fixed questions |
| `test_tools.py`, `test_pydantic_agent.py`, `test_team.py` | Tests that need no API key and no credit |

## Setup

1. Install the libraries:
   ```
   pip install openai python-dotenv pytest "pydantic-ai-slim[openai]" streamlit
   ```
2. Create a `.env` file (it is in `.gitignore` and must never be uploaded):
   ```
   OPENROUTER_API_KEY=your-key
   MODEL=openai/gpt-4o-mini
   PYDANTIC_AI_NO_BANNER=1
   ```
   The model must support tool calling. Free models (`:free`) on OpenRouter work too,
   but they have a daily limit (50 requests without credit) and can be busy (error 429).

## Run

```
python -m pytest -q                       # tests, no credit used
python helpdesk_team.py                   # chat in the terminal (multi-agent team)
python eval_questions.py team             # evaluate the team (9 questions)
python eval_questions.py team quick       # quick evaluation (4 questions)
python eval_questions.py                  # evaluate version 1
python eval_questions.py pydantic         # evaluate version 2
streamlit run app.py                      # web interface at http://localhost:8501
```

## Evaluation

| Version | Model | Questions | Result |
|---|---|---|---|
| 1. Without framework | nvidia/nemotron-3-ultra-550b-a55b:free | 7 | 7/7 |
| 2. Pydantic AI | nvidia/nemotron-3-ultra-550b-a55b:free | 7 | 7/7 |
| 3. Multi-agent team | openai/gpt-4o-mini | 9 | 9/9 |

The team's 9 questions: 4 answered by the guides, 3 IT questions the guides don't cover
(must get labelled general guidance), 1 shared password (must refuse and point to
self-service), and 1 non-IT question (must refuse without searching).

The evaluation also found real problems that were then fixed: the model once **invented a
source** that does not exist in the guides (now rejected by the validator), it classified a
simple how-to question as non-IT (triage instructions fixed), and it searched up to 16 times
for one question (now the code searches once).

What these results mean, and what they do not:

- One run per version, on a small set of questions. Model output can change between runs.
- The checks look for key phrases (for example "30 minutes" and the source name).
- They show the agents behave correctly on these questions, not that they are always accurate.

## Limitations

- Keyword search: it matches words, not meaning.
- General answers come from the model's own knowledge, so they can be wrong. They are
  clearly labelled, and harder problems become tickets.
- Tickets are saved to a local file, not to a real ticketing system.
- Memory lasts for one chat only, and there is no sign-in, so tickets don't record who asked.

## Next steps

- Replace the sample guides with real IT procedures.
- Connect `create_ticket` to a real ticketing system (only that function changes),
  starting with a test (sandbox) environment.
- Search by meaning (embeddings) instead of keywords.
- Sign-in, so each ticket records the employee and the agent can check permissions.
