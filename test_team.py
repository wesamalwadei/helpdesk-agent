"""Test the multi-agent team with fake models (FunctionModel). No API key, no credit."""
import json

from pydantic_ai.messages import ModelResponse, ToolCallPart
from pydantic_ai.models.function import FunctionModel

import tools
from helpdesk_team import GENERAL_LABEL, run_team


def output(data):
    """A fake model that returns the agent's final structured output."""
    def model(messages, info):
        return ModelResponse(parts=[ToolCallPart(info.output_tools[0].name, data)])
    return FunctionModel(model)


def search_then_output(query, data):
    """A fake resolver: has no tools (the code already searched) and answers in ONE request."""
    def model(messages, info):
        assert info.function_tools == []
        assert sum(isinstance(m, ModelResponse) for m in messages) == 0  # first and only request
        return ModelResponse(parts=[ToolCallPart(info.output_tools[0].name, data)])
    return FunctionModel(model)


def triage(category="network", priority="medium", password=False, it=True, wants_ticket=False,
           questions=("The question",), ticket_for="latest"):
    issues = [{"question": q, "category": category, "priority": priority} for q in questions]
    return output({"is_it_question": it, "shares_password": password, "wants_ticket": wants_ticket,
                   "ticket_for": ticket_for, "issues": issues})


def ticket_request(ticket_for="latest", questions=()):
    """Triage output for 'can you open a ticket?'."""
    return triage(wants_ticket=True, ticket_for=ticket_for, questions=questions)


def answered(question, category, priority="medium", ticket=None):
    """An assistant message in the history, remembering which issue it answered."""
    return {"role": "assistant", "content": f"Answer about: {question}",
            "issues": [{"question": question, "category": category, "priority": priority, "ticket": ticket}]}


NO_TICKET = output({"summary": "should not run", "details": "x"})


def recording_ticket_model(seen, summary="Ticket", details="Details"):
    def model(messages, info):
        seen.append(str(messages))
        return ModelResponse(parts=[ToolCallPart(info.output_tools[0].name,
                                                 {"summary": summary, "details": details})])
    return FunctionModel(model)


def test_guide_answer_cites_source_and_opens_no_ticket(tmp_path, monkeypatch):
    monkeypatch.setattr(tools, "TICKETS_PATH", tmp_path / "tickets.json")
    models = {
        "triage": triage(questions=["My VPN is not connecting"]),
        "resolver": search_then_output("VPN", {"answer_type": "guide", "answer": "Restart the VPN app.",
                                               "source": "VPN connection", "needs_technician": False}),
        "ticket": NO_TICKET,
    }
    result = run_team("My VPN is not connecting", models=models, verbose=False)
    assert result["steps"] == ["triage_agent (network, medium)", "search_docs", "resolver_agent"]
    assert "Source: VPN connection" in result["answer"]
    assert not (tmp_path / "tickets.json").exists()


def test_general_answer_is_labelled_and_opens_no_ticket(tmp_path, monkeypatch):
    monkeypatch.setattr(tools, "TICKETS_PATH", tmp_path / "tickets.json")
    models = {
        "triage": triage("software", "low"),
        "resolver": search_then_output("screenshot", {"answer_type": "general", "answer": "Press Win+Shift+S.",
                                                      "source": "", "needs_technician": False}),
        "ticket": NO_TICKET,
    }
    result = run_team("How do I take a screenshot?", models=models, verbose=False)
    assert result["answer"].startswith(GENERAL_LABEL)
    assert "create_ticket" not in result["steps"]


def test_general_answer_with_technician_also_opens_ticket(tmp_path, monkeypatch):
    monkeypatch.setattr(tools, "TICKETS_PATH", tmp_path / "tickets.json")
    models = {
        "triage": triage("hardware", "high"),
        "resolver": search_then_output("battery", {"answer_type": "general", "answer": "Lower the brightness.",
                                                   "source": "", "needs_technician": True}),
        "ticket": output({"summary": "Battery drains fast", "details": "Laptop battery drains quickly."}),
    }
    result = run_team("My laptop battery drains fast", models=models, verbose=False)
    assert result["steps"][-2:] == ["ticket_agent", "create_ticket"]
    saved = json.loads((tmp_path / "tickets.json").read_text(encoding="utf-8"))
    assert saved[0]["priority"] == "high" and saved[0]["category"] == "hardware"
    assert "IT-0001" in result["answer"] and GENERAL_LABEL in result["answer"]


def test_non_it_question_is_redirected_without_more_calls(tmp_path, monkeypatch):
    monkeypatch.setattr(tools, "TICKETS_PATH", tmp_path / "tickets.json")
    models = {"triage": output({"is_it_question": False, "shares_password": False, "issues": []}),
              "resolver": NO_TICKET, "ticket": NO_TICKET}
    result = run_team("What is the weather today?", models=models, verbose=False)
    assert result["steps"] == ["triage_agent (other, low)"]
    assert "IT questions" in result["answer"]


def test_invented_source_is_rejected_and_answer_becomes_general(tmp_path, monkeypatch):
    """The validator rejects a source that is not in the guides; the model must fix it."""
    monkeypatch.setattr(tools, "TICKETS_PATH", tmp_path / "tickets.json")
    calls = []

    def resolver(messages, info):
        step = sum(isinstance(m, ModelResponse) for m in messages)
        calls.append(step)
        if step == 0:  # invents a source
            return ModelResponse(parts=[ToolCallPart(info.output_tools[0].name, {
                "answer_type": "guide", "answer": "Disable Live Preview.",
                "source": "Microsoft Office Troubleshooting Guide", "needs_technician": False})])
        return ModelResponse(parts=[ToolCallPart(info.output_tools[0].name, {
            "answer_type": "general", "answer": "Disable Live Preview.", "source": "", "needs_technician": False})])

    models = {"triage": triage("software", "low"), "resolver": FunctionModel(resolver), "ticket": NO_TICKET}
    result = run_team("Excel freezes when I paste data", models=models, verbose=False)
    assert calls == [0, 1]  # it was asked to fix the invented source
    assert result["answer"].startswith(GENERAL_LABEL)
    assert "Microsoft Office Troubleshooting Guide" not in result["answer"]


def test_guide_answer_with_medium_priority_opens_no_ticket(tmp_path, monkeypatch):
    """Official steps are enough for a normal issue, even if the model says 'technician'."""
    monkeypatch.setattr(tools, "TICKETS_PATH", tmp_path / "tickets.json")
    models = {
        "triage": triage("network", "medium"),
        "resolver": search_then_output("VPN", {"answer_type": "guide", "answer": "Restart the VPN app.",
                                               "source": "VPN connection", "needs_technician": True}),
        "ticket": NO_TICKET,
    }
    result = run_team("My VPN is not connecting", models=models, verbose=False)
    assert "create_ticket" not in result["steps"]


def test_resolver_gets_search_results_in_one_request(tmp_path, monkeypatch):
    """The code searches once and puts the guide sections in the prompt: no extra tool requests."""
    monkeypatch.setattr(tools, "TICKETS_PATH", tmp_path / "tickets.json")
    prompts = []

    def resolver(messages, info):
        prompts.append(str(messages))
        return ModelResponse(parts=[ToolCallPart(info.output_tools[0].name, {
            "answer_type": "guide", "answer": "Restart the VPN app.", "source": "VPN connection",
            "needs_technician": False})])

    models = {"triage": triage(questions=["My VPN is not connecting"]),
              "resolver": FunctionModel(resolver), "ticket": NO_TICKET}
    run_team("My VPN is not connecting", models=models, verbose=False)
    assert len(prompts) == 1
    assert "## VPN connection" in prompts[0]


def test_shared_password_always_gets_a_warning(tmp_path, monkeypatch):
    monkeypatch.setattr(tools, "TICKETS_PATH", tmp_path / "tickets.json")
    models = {
        "triage": triage("accounts", "high", password=True, questions=["How do I reset my password?"]),
        "resolver": search_then_output("password", {"answer_type": "guide", "answer": "Use the self-service page.",
                                                    "source": "Password reset and account lockout",
                                                    "needs_technician": False}),
        "ticket": NO_TICKET,
    }
    result = run_team("My password is Abc123, reset it", models=models, verbose=False)
    assert result["answer"].startswith("Please don't share your password")
    assert "Abc123" not in result["answer"]


def test_password_never_reaches_the_ticket(tmp_path, monkeypatch):
    """Even if triage copies the password into the issue, the ticket writer never sees it."""
    monkeypatch.setattr(tools, "TICKETS_PATH", tmp_path / "tickets.json")
    seen = []
    models = {
        "triage": triage("security", "high", password=True, questions=["My password is Abc123, reset it"]),
        "resolver": search_then_output("password", {"answer_type": "general", "answer": "Use self-service.",
                                                    "source": "", "needs_technician": True}),
        "ticket": recording_ticket_model(seen, "Password help", "Reset needed"),
    }
    run_team("My password is Abc123, reset it", models=models, verbose=False)
    assert seen and "Abc123" not in seen[0]
    assert "Abc123" not in (tmp_path / "tickets.json").read_text(encoding="utf-8")


# ---------- Several issues and follow-ups ----------
def test_two_problems_in_one_message_get_two_answers(tmp_path, monkeypatch):
    monkeypatch.setattr(tools, "TICKETS_PATH", tmp_path / "tickets.json")
    queries = []
    original = tools.search_docs
    monkeypatch.setattr(tools, "search_docs", lambda q: queries.append(q) or original(q))
    answers = iter([
        {"answer_type": "guide", "answer": "Restart the VPN app.", "source": "VPN connection",
         "needs_technician": False},
        {"answer_type": "guide", "answer": "Check that Outlook shows Connected.",
         "source": "Outlook not sending or receiving email", "needs_technician": False},
    ])

    def resolver(messages, info):
        return ModelResponse(parts=[ToolCallPart(info.output_tools[0].name, next(answers))])

    models = {
        "triage": triage(questions=["Why is my VPN not connecting?", "Why is Outlook not receiving email?"]),
        "resolver": FunctionModel(resolver),
        "ticket": NO_TICKET,
    }
    result = run_team("my VPN and my Outlook both don't work", models=models, verbose=False)
    assert queries == ["Why is my VPN not connecting?", "Why is Outlook not receiving email?"]
    assert result["steps"].count("resolver_agent") == 2
    assert "**1. Why is my VPN not connecting?**" in result["answer"]
    assert "**2. Why is Outlook not receiving email?**" in result["answer"]
    assert "Source: VPN connection" in result["answer"]
    assert "Source: Outlook not sending or receiving email" in result["answer"]


VPN_THEN_EMAIL = [
    {"role": "user", "content": "I cannot connect to the VPN"},
    answered("Why can't I connect to the VPN?", "network"),
    {"role": "user", "content": "my email is not working"},
    answered("Why is my email not working?", "email"),
]


def test_ticket_is_about_the_latest_problem_chosen_by_code(tmp_path, monkeypatch):
    """After VPN then email, 'can you open ticket' is about EMAIL: the code picks it, not the model."""
    monkeypatch.setattr(tools, "TICKETS_PATH", tmp_path / "tickets.json")
    seen = []
    models = {
        # even if triage wrongly suggests the VPN, "latest" means the code uses the email issue
        "triage": triage("network", "high", wants_ticket=True, questions=["Why can't I connect to the VPN?"]),
        "resolver": NO_TICKET,  # must not run
        "ticket": recording_ticket_model(seen, "Email not working", "Outlook steps failed"),
    }
    result = run_team("can you open ticket", models=models, verbose=False, history=VPN_THEN_EMAIL)
    assert result["steps"] == ["triage_agent (email, medium)", "ticket_agent", "create_ticket"]
    assert "Why is my email not working?" in seen[0]
    saved = json.loads((tmp_path / "tickets.json").read_text(encoding="utf-8"))
    assert len(saved) == 1 and saved[0]["category"] == "email"
    assert result["issues"][0]["ticket"] == "IT-0001"
    assert "If you meant a different problem" in result["answer"]


def test_ticket_for_a_named_problem(tmp_path, monkeypatch):
    """'Open a ticket for the VPN' uses the problem the employee named."""
    monkeypatch.setattr(tools, "TICKETS_PATH", tmp_path / "tickets.json")
    seen = []
    models = {"triage": triage("network", "medium", wants_ticket=True, ticket_for="named",
                               questions=["Why can't I connect to the VPN?"]),
              "resolver": NO_TICKET, "ticket": recording_ticket_model(seen, "VPN not connecting")}
    run_team("open a ticket for the VPN one", models=models, verbose=False, history=VPN_THEN_EMAIL)
    saved = json.loads((tmp_path / "tickets.json").read_text(encoding="utf-8"))
    assert saved[0]["category"] == "network"
    assert "VPN" in seen[0]


def test_ticket_request_for_both_issues_opens_two_tickets(tmp_path, monkeypatch):
    monkeypatch.setattr(tools, "TICKETS_PATH", tmp_path / "tickets.json")
    models = {"triage": ticket_request("all"), "resolver": NO_TICKET, "ticket": recording_ticket_model([])}
    result = run_team("open tickets for both", models=models, verbose=False, history=VPN_THEN_EMAIL)
    assert result["steps"].count("create_ticket") == 2
    saved = json.loads((tmp_path / "tickets.json").read_text(encoding="utf-8"))
    assert [t["category"] for t in saved] == ["network", "email"]


def test_no_duplicate_ticket_for_the_same_problem(tmp_path, monkeypatch):
    monkeypatch.setattr(tools, "TICKETS_PATH", tmp_path / "tickets.json")
    history = [{"role": "user", "content": "my email is not working"},
               answered("Why is my email not working?", "email", ticket="IT-0007")]
    models = {"triage": ticket_request(), "resolver": NO_TICKET, "ticket": NO_TICKET}
    result = run_team("can you open a ticket", models=models, verbose=False, history=history)
    assert "create_ticket" not in result["steps"]
    assert "IT-0007" in result["answer"] and "already" in result["answer"]
    assert not (tmp_path / "tickets.json").exists()


def test_ticket_request_with_no_problem_yet_asks_what_it_is(tmp_path, monkeypatch):
    monkeypatch.setattr(tools, "TICKETS_PATH", tmp_path / "tickets.json")
    models = {"triage": ticket_request(), "resolver": NO_TICKET, "ticket": NO_TICKET}
    result = run_team("I want to open a ticket", models=models, verbose=False)
    assert "What is the problem?" in result["answer"]
    assert not (tmp_path / "tickets.json").exists()


def test_password_in_earlier_message_is_never_passed_on(tmp_path, monkeypatch):
    monkeypatch.setattr(tools, "TICKETS_PATH", tmp_path / "tickets.json")
    seen = []
    history = [{"role": "user", "content": "My password is Abc123", "has_password": True},
               {"role": "assistant", "content": "Please don't share your password."}]
    models = {
        "triage": triage("accounts", "high", wants_ticket=True, ticket_for="named",
                         questions=["I need help with my account"]),
        "resolver": NO_TICKET,
        "ticket": recording_ticket_model(seen, "Account help", "Needs a technician"),
    }
    run_team("open a ticket please", models=models, verbose=False, history=history)
    assert "Abc123" not in seen[0]
    assert "Abc123" not in (tmp_path / "tickets.json").read_text(encoding="utf-8")
