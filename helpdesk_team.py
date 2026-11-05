"""Multi-agent IT helpdesk built with Pydantic AI.

Three agents, each with one job:
  1. triage_agent   - classifies the message: IT or not, password shared, ticket wanted,
                      and splits it into separate issues (each with category and priority)
  2. resolver_agent - answers from the IT guides (searched by meaning, once, by the code); if they don't cover it,
                      gives safe general IT guidance, clearly labelled
  3. ticket_agent   - writes a clear ticket when the employee asks for one
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
class Issue(BaseModel):
    question: str = Field(description="The IT problem rewritten as a short standalone question")
    category: Category
    priority: Priority


class Triage(BaseModel):
    is_it_question: bool = Field(description="True if this is about computers, devices, accounts, software, network or IT security")
    shares_password: bool = Field(description="True if the employee wrote a password in the new message")
    wants_ticket: bool = Field(default=False,
                               description="True ONLY if the employee explicitly asks to open/create/raise a "
                                           "ticket or to talk to a technician. Describing a problem is NOT a ticket request.")
    ticket_for: Literal["latest", "named", "all"] = Field(
        default="latest",
        description="For a ticket request: latest = the last problem discussed; named = the employee "
                    "names the problem in the new message; all = 'both' / 'all of them'")
    issues: list[Issue] = Field(default_factory=list, description="See the instructions")

    # The first issue sets the category and priority shown in the steps.
    @property
    def category(self):
        return self.issues[0].category if self.issues else "other"

    @property
    def priority(self):
        return self.issues[0].priority if self.issues else "low"


triage_agent = Agent(
    output_type=Triage,
    model_settings={"max_tokens": 300},
    instructions="""You classify helpdesk messages. Do not answer them.
is_it_question: true for anything about computers, phones, devices, accounts, passwords,
email, internet, software and apps (Windows, Office, Teams...), including simple how-to
questions such as taking a screenshot or using a shortcut. When in doubt, choose true.
false only for clearly non-technical topics (weather, pay, holidays, food...).

issues: one entry per DIFFERENT IT problem, each as a short standalone question.
- The new message names several problems ("my VPN and Outlook don't work"): one entry each.
- The new message is a follow-up ("that didn't work", "still broken"): one entry for the
  problem being discussed, using the earlier messages.
- "I have an issue with my email" or "my VPN is broken" is a PROBLEM, not a ticket request:
  wants_ticket = false, and add the problem to issues so it gets solutions.
- Only an explicit request ("open a ticket", "raise a ticket", "I want a technician") sets
  wants_ticket and ticket_for. Then fill issues only when ticket_for is "named" (the problem
  the employee names); otherwise leave issues empty.
Priority per issue:
- high: a security risk (clicked a phishing link, shared a password) or the employee cannot work at all
- medium: one tool or device is not working
- low: a how-to question or a request
category "security" is for phishing, suspicious emails and shared passwords.
Never copy a password into an issue.""",
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
    instructions="""You write IT support tickets for technicians. Be short and specific.
In details, say what the employee reported and what was already suggested in the conversation
(so the technician doesn't repeat it). Never include a password.""",
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


MAX_ISSUES = 3  # each issue costs one more model request


def open_ticket(issue, triage, context, models, steps, verbose):
    """Write and save a ticket for one issue."""
    # The issue text is written by triage; a password is never passed on (see conversation_context).
    problem = issue.question
    if triage.shares_password:  # the issue text might echo the password: don't pass it on
        problem = (f"An IT {issue.category} problem. The employee's message contained a password, "
                   "which was removed. Never include a password.")
    prompt = f"{context}Write a ticket about this problem: {problem}"
    draft = run_with_retry(ticket_agent, prompt, models["ticket"]).output
    ticket = tools.create_ticket(draft.summary, draft.details, path=tools.TICKETS_PATH,
                                 category=issue.category, priority=issue.priority)
    steps += ["ticket_agent", "create_ticket"]
    if verbose:
        print(f"  [ticket] {ticket['id']}: {ticket['summary']}")
    return ticket


def resolve_issue(issue, triage, context, models, steps, verbose, history=None):
    """Search the guides once and answer one issue. Tickets are opened only on request."""
    sections, mode = tools.smart_search(issue.question)  # by meaning; keywords if embeddings fail
    run = run_with_retry(resolver_agent,
                         resolver_prompt(issue.question, sections, triage.shares_password, context),
                         models["resolver"])
    res = run.output
    steps += ["semantic_search" if mode == "semantic" else "search_docs", "resolver_agent"]
    if verbose:
        print(f"  [search:{mode}] {issue.question!r} -> {[(r['section'], r['score']) for r in sections]}")
        print(f"  [resolver] type={res.answer_type} source={res.source!r} technician={res.needs_technician}")

    answer = res.answer.strip()
    if res.answer_type == "guide" and not is_real_section(res.source):
        res.answer_type = "general"  # last safety net: never show an invented source
    if res.answer_type == "guide":
        if res.source and res.source.lower() not in answer.lower():
            answer += f"\n\nSource: {res.source}"
    elif not answer.lower().startswith(GENERAL_LABEL.lower()):
        answer = f"{GENERAL_LABEL}\n\n{answer}"

    # Solutions first. A ticket is opened only when the employee asks for one;
    # here we just offer it (more strongly when a technician is probably needed).
    already = existing_ticket(issue.question, history)
    if already:
        answer += f"\n\nYou already have ticket **{already}** for this; the IT team will follow up."
    elif res.needs_technician or issue.priority == "high":
        answer += ("\n\nThis may need an IT technician. If the steps don't fix it, "
                   "say **open a ticket** and I'll create one for you.")
    else:
        answer += "\n\nStill not working? Say **open a ticket** and I'll pass it to the IT team."
    return answer, None


# ---------- Which problem is a ticket for? ----------
# The code decides from the chat history, not the model, so the ticket is always about the
# right problem. Each assistant message in the history keeps the issues it answered:
#   {"role": "assistant", "content": ..., "issues": [{"question", "category", "priority", "ticket"}]}
def latest_issues(history):
    for m in reversed(history or []):
        if m["role"] == "assistant" and m.get("issues"):
            return m["issues"]
    return []


def all_issues(history):
    seen = {}
    for m in history or []:
        for i in m.get("issues", []) if m["role"] == "assistant" else []:
            seen.setdefault(i["question"].lower(), dict(i))
            if i.get("ticket"):
                seen[i["question"].lower()]["ticket"] = i["ticket"]
    return list(seen.values())


def existing_ticket(issue_question, history):
    """The ticket already opened for this problem in this chat, if any."""
    for i in all_issues(history):
        if i["question"].lower() == issue_question.lower() and i.get("ticket"):
            return i["ticket"]
    return None


def ticket_targets(triage, history):
    if triage.ticket_for == "named" and triage.issues:
        return [i.model_dump() for i in triage.issues[:MAX_ISSUES]]
    if triage.ticket_for == "all":
        return all_issues(history)[:MAX_ISSUES]
    return latest_issues(history)


def handle_ticket_request(question, triage, context, history, models, steps, verbose):
    targets = ticket_targets(triage, history)
    if not targets:
        answer = ("Sure, I can open a ticket. What is the problem? "
                  "Describe it in a sentence and I'll pass it to the IT team.")
        return answer, [], []
    steps[0] = f"triage_agent ({targets[0]['category']}, {targets[0]['priority']})"
    lines, tickets, issues = [], [], []
    for target in targets:
        issue = Issue(question=target["question"], category=target["category"], priority=target["priority"])
        already = existing_ticket(issue.question, history)
        if already:
            lines.append(f"- **{already}** was already opened for: {issue.question}")
            issues.append({**issue.model_dump(), "ticket": already})
            continue
        t = open_ticket(issue, triage, context, models, steps, verbose)
        tickets.append(t)
        issues.append({**issue.model_dump(), "ticket": t["id"]})
        lines.append(f"- **{t['id']}**: {t['summary']} (priority: {t['priority']}, category: {t['category']}), "
                     f"for: {issue.question}")
    answer = ("Here is your ticket status:\n\n" + "\n".join(lines)
              + "\n\nAn IT technician will follow up with you. "
              "If you meant a different problem, just tell me which one.")
    return answer, tickets, issues


def run_team(question, models=None, verbose=True, history=None):
    """Run the agents. Returns a dict with the answer and every step taken.

    history: earlier chat messages (see conversation_context), so follow-ups work.
    """
    if models is None:
        shared = make_model()
        models = {"triage": shared, "resolver": shared, "ticket": shared}
    steps = []
    context = conversation_context(history)

    # 1. Triage: is it IT, which problems, which ticket?
    triage_input = f"{context}New message: {question}" if context else question
    triage = run_with_retry(triage_agent, triage_input, models["triage"]).output
    issues = triage.issues[:MAX_ISSUES] or [Issue(question=question, category="other", priority="medium")]
    steps.append(f"triage_agent ({triage.category}, {triage.priority})")
    if verbose:
        print(f"  [triage] it={triage.is_it_question} password={triage.shares_password} "
              f"ticket={triage.wants_ticket} issues={[(i.question, i.category, i.priority) for i in issues]}")

    # 2a. The employee asked for a ticket: the code picks the problem from the chat history.
    #     (If there is nothing to ticket yet but the message describes a problem, solve it first.)
    if triage.wants_ticket and not ticket_targets(triage, history) and triage.issues:
        triage.wants_ticket = False
    if triage.wants_ticket:
        answer, tickets, done = handle_ticket_request(question, triage, context, history, models, steps, verbose)
        return {"answer": answer, "steps": steps, "triage": triage, "tickets": tickets,
                "ticket": tickets[0] if tickets else None, "issues": done}

    if not triage.is_it_question:
        answer = ("I'm the IT helpdesk assistant, so I can only help with IT questions "
                  "(devices, accounts, software, network and security). "
                  "For other topics, please contact the right department.")
        return {"answer": answer, "steps": steps, "triage": triage, "tickets": [], "ticket": None,
                "issues": []}

    # 2b. Answer each problem separately.
    parts, tickets, done = [], [], []
    for issue in issues:
        part, ticket = resolve_issue(issue, triage, context, models, steps, verbose, history)
        parts.append(part)
        done.append({**issue.model_dump(), "ticket": ticket["id"] if ticket else None})
        if ticket:
            tickets.append(ticket)
    if len(parts) == 1:
        answer = parts[0]
    else:
        answer = "\n\n".join(f"**{n}. {issue.question}**\n\n{part}"
                               for n, (issue, part) in enumerate(zip(issues, parts), 1))
    if triage.shares_password and "share" not in answer.lower():
        answer = "Please don't share your password with anyone, including IT.\n\n" + answer

    return {"answer": answer, "steps": steps, "triage": triage, "tickets": tickets,
            "ticket": tickets[0] if tickets else None, "issues": done}


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
                chat.append({"role": "assistant", "content": result["answer"], "issues": result["issues"]})
            except Exception as e:
                print(f"Error: {e}\nTry again in a minute, or change MODEL in .env.")
