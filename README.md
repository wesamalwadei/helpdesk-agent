# IT Helpdesk Agent

An AI agent that answers employees' IT questions from the IT guides, cites the source,
and opens a support ticket when it cannot find an answer.

> Demo project. The IT guides in `it_docs.md` are sample content, not real company procedures.

## The problem

Answers to recurring IT questions (VPN, passwords, printers, software requests) are spread
across guides that are hard to find. Employees lose time searching, and the IT helpdesk
answers the same questions every day.

## What the agent does

1. The employee asks an IT question.
2. The agent decides to call `search_docs` to search the IT guides.
3. If it finds the answer, it replies from the guides only and cites the section.
4. If not, it says it doesn't know and calls `create_ticket` to open a ticket (saved in `tickets.json`).
5. It never asks for, repeats, or stores a password.

The model decides which tool to call and when. This is what makes it an AI agent rather than a chatbot.

## Two versions of the same agent

| | Without framework | Pydantic AI |
|---|---|---|
| File | `helpdesk_agent.py` | `helpdesk_agent_pydantic.py` |
| Tool-calling loop | Written by hand | Run by the framework |
| Tool descriptions | Written by hand (JSON schema) | Generated from the function and its docstring |
| Bad tool arguments | Handled by our code | Detected by the framework, model asked to fix them |
| Lines of code (no blanks/comments) | 110 | 62 |

Both versions use the same tools (`tools.py`), the same rules (system prompt) and the same model.

## Project files

| File | Purpose |
|---|---|
| `it_docs.md` | Sample IT guides (8 topics) |
| `tools.py` | `search_docs` (keyword search) and `create_ticket` |
| `helpdesk_agent.py` | Agent without a framework |
| `helpdesk_agent_pydantic.py` | Agent built with Pydantic AI |
| `app.py` | Streamlit web interface |
| `eval_questions.py` | Evaluation on 7 fixed questions |
| `test_tools.py`, `test_pydantic_agent.py` | Tests that need no API key and no credit |

## Setup

1. Install the libraries:
   ```
   pip install openai python-dotenv pytest "pydantic-ai-slim[openai]" streamlit
   ```
2. Create a `.env` file (it is in `.gitignore` and must never be uploaded):
   ```
   OPENROUTER_API_KEY=your-key
   MODEL=nvidia/nemotron-3-ultra-550b-a55b:free
   PYDANTIC_AI_NO_BANNER=1
   ```
   The model must support tool calling. Free models (`:free`) on OpenRouter have daily
   and shared rate limits; `openai/gpt-4o-mini` is a cheap, more stable option.

## Run

```
python -m pytest -q                    # tests, no credit used
python helpdesk_agent.py               # chat in the terminal (no framework)
python helpdesk_agent_pydantic.py      # chat in the terminal (Pydantic AI)
python eval_questions.py               # evaluate the version without a framework
python eval_questions.py pydantic      # evaluate the Pydantic AI version
streamlit run app.py                   # web interface at http://localhost:8501
```

## Evaluation

7 fixed questions: 4 answered by the guides, 2 not covered (must open a ticket),
and 1 where the employee shares a password (must refuse and point to self-service).

| Version | Model | Result |
|---|---|---|
| Without framework | nvidia/nemotron-3-ultra-550b-a55b:free | 7/7 |
| Pydantic AI | nvidia/nemotron-3-ultra-550b-a55b:free | 7/7 |

What this result means, and what it does not:

- One run per version, on 7 questions. Model output can change between runs.
- The checks look for key phrases (for example "30 minutes" and the source name), so a correct
  answer with different wording could fail, and the reverse is possible.
- It shows the agent behaves correctly on these questions, not that it is always accurate.

## Limitations

- Keyword search: it finds sections that share words with the question, not meaning.
  A question worded very differently may miss the right section.
- 8 sample topics only. Anything else becomes a ticket.
- Tickets are saved to a local file, not to a real ticketing system.
- No memory between questions and no user permissions.
- Free models can be busy (error 429). The agent retries 3 times, then shows a clear error.

## Next steps

- Replace the sample guides with real IT procedures.
- Connect `create_ticket` to a real ticketing system (only that function changes),
  starting with a test (sandbox) environment.
- Search by meaning (embeddings) instead of keywords.
- Grow into a multi-agent system: one agent classifies the issue, one searches,
  one follows up on tickets.
