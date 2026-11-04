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


def triage(category="network", priority="medium", password=False, it=True):
    return output({"is_it_question": it, "category": category, "priority": priority, "shares_password": password})


NO_TICKET = output({"summary": "should not run", "details": "x"})


def test_guide_answer_cites_source_and_opens_no_ticket(tmp_path, monkeypatch):
    monkeypatch.setattr(tools, "TICKETS_PATH", tmp_path / "tickets.json")
    models = {
        "triage": triage(),
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
        "resolver": search_then_output("blue screen", {"answer_type": "general", "answer": "Restart the laptop.",
                                                       "source": "", "needs_technician": True}),
        "ticket": output({"summary": "Blue screen after update", "details": "Laptop shows a blue screen."}),
    }
    result = run_team("My laptop shows a blue screen", models=models, verbose=False)
    assert result["steps"][-2:] == ["ticket_agent", "create_ticket"]
    saved = json.loads((tmp_path / "tickets.json").read_text(encoding="utf-8"))
    assert saved[0]["priority"] == "high" and saved[0]["category"] == "hardware"
    assert "IT-0001" in result["answer"] and GENERAL_LABEL in result["answer"]


def test_non_it_question_is_redirected_without_more_calls(tmp_path, monkeypatch):
    monkeypatch.setattr(tools, "TICKETS_PATH", tmp_path / "tickets.json")
    models = {"triage": triage("other", "low", it=False), "resolver": NO_TICKET, "ticket": NO_TICKET}
    result = run_team("What is the weather today?", models=models, verbose=False)
    assert result["steps"] == ["triage_agent (other, low)"]
    assert "IT questions" in result["answer"]


def test_password_never_reaches_the_ticket(tmp_path, monkeypatch):
    monkeypatch.setattr(tools, "TICKETS_PATH", tmp_path / "tickets.json")
    seen = []

    def ticket_model(messages, info):
        seen.append(str(messages))
        return ModelResponse(parts=[ToolCallPart(info.output_tools[0].name,
                                                 {"summary": "Password help", "details": "Reset needed"})])

    models = {
        "triage": triage("security", "high", password=True),
        "resolver": search_then_output("password", {"answer_type": "general", "answer": "Use self-service.",
                                                    "source": "", "needs_technician": True}),
        "ticket": FunctionModel(ticket_model),
    }
    run_team("My password is Abc123, reset it", models=models, verbose=False)
    assert seen and "Abc123" not in seen[0]
    assert "Abc123" not in (tmp_path / "tickets.json").read_text(encoding="utf-8")


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

    models = {"triage": triage(), "resolver": FunctionModel(resolver), "ticket": NO_TICKET}
    run_team("My VPN is not connecting", models=models, verbose=False)
    assert len(prompts) == 1
    assert "## VPN connection" in prompts[0]


def test_shared_password_always_gets_a_warning(tmp_path, monkeypatch):
    monkeypatch.setattr(tools, "TICKETS_PATH", tmp_path / "tickets.json")
    models = {
        "triage": triage("accounts", "high", password=True),
        "resolver": search_then_output("password", {"answer_type": "guide", "answer": "Use the self-service page.",
                                                    "source": "Password reset and account lockout",
                                                    "needs_technician": False}),
        "ticket": NO_TICKET,
    }
    result = run_team("My password is Abc123, reset it", models=models, verbose=False)
    assert result["answer"].startswith("Please don't share your password")
    assert "Abc123" not in result["answer"]
