"""Multi-agent IT helpdesk built with Pydantic AI.

Three agents, each with one job:
  1. triage_agent   - classifies the question: IT or not, category, priority, password shared?
  2. resolver_agent - answers from the IT guides (searched once by the code); if they don't cover it,
                      gives safe general IT guidance, clearly labelled
  3. ticket_agent   - writes a clear ticket when a technician is needed
Python code (run_team) passes the work from one agent to the next.
"""
import time
from typing import Literal

from pydantic import BaseModel, Field
from pydantic_ai import Agent, ModelRetry

import tools
from helpdesk_agent import MODEL

Category = Literal["network", "accounts", "software", "hardware", "email", "security", "other"]
Priority = Literal["low", "medium", "high"]
GENERAL_LABEL = "General guidance (not from the company IT guides):"


# ---------- 1. Triage ----------
class Triage(BaseModel):
    is_it_question: bool = Field(description="True if this is about computers, devices, accounts, software, network or IT security")
    category: Category
    priority: Priority
    shares_password: bool = Field(description="True if the employee wrote a password in the new message")
    wants_ticket: bool = Field(default=False,
                               description="True if the employee asks to open a ticket or to talk to a technician")


triage_agent = Agent(
    output_type=Triage,
    model_settings={"max_tokens": 150},
    instructions="""You classify helpdesk messages. Do not answer them.
is_it_question: true for anything about computers, phones, devices, accounts, passwords,
email, internet, software and apps (Windows, Office, Teams...), including simple how-to
questions such as taking a screenshot or using a shortcut. When in doubt, choose true.
false only for clearly non-technical topics (weather, pay, holidays, food...).
Priority:
- high: a security risk (clicked a phishing link, shared a password) or the employee cannot work at all
- medium: one tool or device is not working
- low: a how-to question or a request
category "security" is for phishing, suspicious emails and shared passwords.
If the new message refers to earlier messages (for example "can you open a ticket" or
"that didn't work"), classify the IT issue being discussed; that is an IT question.""",
)


# ---------- 2. Resolver ----------
class Resolution(BaseModel):
    answer_type: Literal["guide", "general"] = Field(
        description="guide = answered from the search results; general = the guides don't cover it")
    answer: str = Field(description="The answer for the employee")
    source: str = Field(description="The guide section used. Empty for general answers.")
    needs_technician: bool = Field(description="True if the employee probably cannot fix this alone")


# To save requests, the code searches the guides ONCE and gives the results to the resolver,
# instead of letting the model call a search tool (each tool call costs one more model request).
resolver_agent = Agent(
    output_type=Resolution,
    retries=2,
    model_settings={"max_tokens": 500},
    instructions=f"""You answer IT questions for company employees.
You get the question and the matching sections of the company IT guides.
1. If the sections answer the question: answer_type "guide", answer ONLY from them,
   and set source to the exact section name.
2. If they don't: answer_type "general". Start the answer with exactly
   "{GENERAL_LABEL}" and give short, standard, safe steps an employee can do alone.
   Never suggest: editing the registry, disabling antivirus or the firewall, BIOS changes,
   admin-only commands, or installing software that is not approved.
3. needs_technician = true when the fix needs admin rights, a hardware repair, data recovery,
   a security response, or when the steps may well not solve it.
4. Never repeat or ask for a password. If the employee shared one, tell them not to
   share passwords and to use the self-service password page.
5. Keep answers short: at most 5 steps.""",
)


def resolver_prompt(question, sections, shares_password, context=""):
    found = "\n\n".join(f"## {r['section']}\n{r['text']}" for r in sections) or "(no matching sections)"
    note = "\n\n(Note: the employee shared a password. Do not repeat it.)" if shares_password else ""
    return f"{context}Question: {question}{note}\n\nIT guide sections:\n{found}"


# ---------- Conversation memory ----------
def conversation_context(history, max_messages=6):
    """The last messages of the chat as text, so agents understand follow-ups.

    history: list of {"role": "user" | "assistant", "content": str, "has_password": bool}.
    A user message that contained a password is never passed on.
    """
    lines = []
    for m in (history or [])[-max_messages:]:
        if m["role"] == "user":
            text = "[message removed: it contained a password]" if m.get("has_password") else m["content"]
            lines.append(f"Employee: {text}")
        else:
            lines.append(f"Assistant: {m['content'][:300]}")
    return "Earlier in this conversation:\n" + "\n".join(lines) + "\n\n" if lines else ""


def last_user_message(history):
    for m in reversed(history or []):
        if m["role"] == "user" and not m.get("has_password"):
            return m["content"]
    return ""


SECTION_TITLES = {title.lower() for title, _ in tools.load_sections()}


def is_real_section(name):
    return name.strip().lower() in SECTION_TITLES


@resolver_agent.output_validator
def check_source(res: Resolution) -> Resolution:
    """Stop invented sources: a "guide" answer must name a real section of the IT guides."""
    if res.answer_type == "guide" and not is_real_section(res.source):
        raise ModelRetry(
            f"'{res.source}' is not a section of the company IT guides. "
            "Use the exact section name from the search results, "
            "or set answer_type to 'general' if the results do not answer the question."
        )
    return res


# ---------- 3. Ticket writer ----------
class TicketDraft(BaseModel):
    summary: str = Field(description="Short title of the problem, max 8 words")
    details: str = Field(description="What the employee needs, for the IT technician. Never include a password.")


ticket_agent = Agent(
    output_type=TicketDraft,
    model_settings={"max_tokens": 200},
    instructions="You write IT support tickets for technicians. Be short and specific. Never include a password.",
)


# ---------- Orchestration ----------
def make_model(name=MODEL):
    from pydantic_ai.models.openrouter import OpenRouterModel
    return OpenRouterModel(name)


def run_with_retry(agent, prompt, model, retries=3, wait=15):
    """Run one agent. If the model is busy (error 429), wait and try again."""
    for attempt in range(1, retries + 1):
        try:
            return agent.run_sync(prompt, model=model)
        except Exception as e:
            # A daily limit won't clear in seconds, so don't wait on it.
            if getattr(e, "status_code", None) != 429 or attempt == retries or "per-day" in str(e):
                raise
            print(f"  [busy] Model is rate-limited. Waiting {wait}s (try {attempt}/{retries})...")
            time.sleep(wait)


def open_ticket(question, triage, context, models, steps, verbose):
    # Never send the new message to the ticket writer if it holds a password.
    new_message = "The employee shared a password; it was removed." if triage.shares_password else question
    draft = run_with_retry(ticket_agent, f"{context}New message: {new_message}", models["ticket"]).output
    ticket = tools.create_ticket(draft.summary, draft.details, path=tools.TICKETS_PATH,
                                 category=triage.category, priority=triage.priority)
    steps += ["ticket_agent", "create_ticket"]
    if verbose:
        print(f"  [ticket] {ticket['id']}: {ticket['summary']}")
    return ticket


def run_team(question, models=None, verbose=True, history=None):
    """Run the agents. Returns a dict with the answer and every step taken.

    history: earlier chat messages (see conversation_context), so follow-ups work.
    """
    if models is None:
        shared = make_model()
        models = {"triage": shared, "resolver": shared, "ticket": shared}
    steps = []

    context = conversation_context(history)

    # 1. Triage
    triage_input = f"{context}New message: {question}" if context else question
    triage = run_with_retry(triage_agent, triage_input, models["triage"]).output
    steps.append(f"triage_agent ({triage.category}, {triage.priority})")
    if verbose:
        print(f"  [triage] it={triage.is_it_question} category={triage.category} "
              f"priority={triage.priority} password={triage.shares_password} ticket={triage.wants_ticket}")

    # The employee asked for a ticket: open it straight away.
    if triage.wants_ticket:
        ticket = open_ticket(question, triage, context, models, steps, verbose)
        answer = (f"Done. I opened ticket **{ticket['id']}**: {ticket['summary']} "
                  f"(priority: {triage.priority}, category: {triage.category}). "
                  "An IT technician will follow up with you.")
        return {"answer": answer, "steps": steps, "triage": triage, "ticket": ticket}

    if not triage.is_it_question:
        answer = ("I'm the IT helpdesk assistant, so I can only help with IT questions "
                  "(devices, accounts, software, network and security). "
                  "For other topics, please contact the right department.")
        return {"answer": answer, "steps": steps, "triage": triage, "ticket": None}

    # 2. Search once in code, then resolve
    # A short follow-up ("that didn't work") is searched together with the earlier question.
    query = f"{last_user_message(history)} {question}" if len(question.split()) <= 5 else question
    sections = tools.search_docs(query.strip())
    run = run_with_retry(resolver_agent, resolver_prompt(question, sections, triage.shares_password, context),
                         models["resolver"])
    res = run.output
    steps += ["search_docs", "resolver_agent"]
    if verbose:
        print(f"  [search] {[r['section'] for r in sections]}")
        print(f"  [resolver] type={res.answer_type} source={res.source!r} technician={res.needs_technician}")

    answer = res.answer.strip()
    if triage.shares_password and "share" not in answer.lower():
        answer = "Please don't share your password with anyone, including IT.\n\n" + answer
    if res.answer_type == "guide" and not is_real_section(res.source):
        res.answer_type = "general"  # last safety net: never show an invented source
    if res.answer_type == "guide":
        if res.source and res.source.lower() not in answer.lower():
            answer += f"\n\nSource: {res.source}"
    elif not answer.lower().startswith(GENERAL_LABEL.lower()):
        answer = f"{GENERAL_LABEL}\n\n{answer}"

    # 3. Ticket. Guide answers are official steps, so only high-priority cases (e.g. security)
    #    get a ticket. General answers get one when a technician is needed or it's high priority.
    ticket = None
    if res.answer_type == "guide":
        needs_ticket = triage.priority == "high" and res.needs_technician
    else:
        needs_ticket = res.needs_technician or triage.priority == "high"
    if needs_ticket:
        ticket = open_ticket(question, triage, context, models, steps, verbose)
        answer += (f"\n\nI also opened ticket **{ticket['id']}** "
                   f"(priority: {triage.priority}, category: {triage.category}) "
                   "so an IT technician can follow up.")

    return {"answer": answer, "steps": steps, "triage": triage, "ticket": ticket}


def run_agent(question, models=None, verbose=True, history=None):
    """Same shape as the other versions: returns (answer, steps)."""
    result = run_team(question, models=models, verbose=verbose, history=history)
    return result["answer"], result["steps"]


if __name__ == "__main__":
    print(f"IT Helpdesk - Multi-agent team (model: {MODEL}). Type 'exit' to quit.")
    chat = []
    while True:
        question = input("\nYou: ").strip()
        if question.lower() in {"exit", "quit"}:
            break
        if question:
            try:
                result = run_team(question, history=chat)
                print(f"Agent: {result['answer']}")
                chat.append({"role": "user", "content": question,
                             "has_password": result["triage"].shares_password})
                chat.append({"role": "assistant", "content": result["answer"]})
            except Exception as e:
                print(f"Error: {e}\nTry again in a minute, or change MODEL in .env.")
