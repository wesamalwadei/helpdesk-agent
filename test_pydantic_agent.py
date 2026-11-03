"""Test the Pydantic AI agent with a fake model (FunctionModel). No API key, no credit."""
from pydantic_ai.messages import ModelResponse, TextPart, ToolCallPart
from pydantic_ai.models.function import FunctionModel

import tools
from helpdesk_agent_pydantic import run_agent


def scripted(*steps):
    """A fake model that returns the given responses one by one."""
    def model(messages, info):
        assert {t.name for t in info.function_tools} == {"search_docs", "create_ticket"}
        step = sum(isinstance(m, ModelResponse) for m in messages)
        return steps[step]
    return FunctionModel(model)


def test_answer_from_docs():
    model = scripted(
        ModelResponse(parts=[ToolCallPart("search_docs", {"query": "VPN not connecting"})]),
        ModelResponse(parts=[TextPart("Restart the VPN app. Source: VPN connection")]),
    )
    answer, used = run_agent("My VPN is not connecting", model=model, verbose=False)
    assert used == ["search_docs"]
    assert "Source: VPN connection" in answer


def test_no_answer_opens_ticket(tmp_path, monkeypatch):
    monkeypatch.setattr(tools, "TICKETS_PATH", tmp_path / "tickets.json")
    model = scripted(
        ModelResponse(parts=[ToolCallPart("search_docs", {"query": "blue screen"})]),
        ModelResponse(parts=[ToolCallPart("create_ticket", {"summary": "Blue screen", "details": "After update"})]),
        ModelResponse(parts=[TextPart("I opened ticket IT-0001.")]),
    )
    answer, used = run_agent("blue screen", model=model, verbose=False)
    assert used == ["search_docs", "create_ticket"]
    assert (tmp_path / "tickets.json").exists()


def test_bad_arguments_are_retried():
    """Pydantic AI checks the arguments and asks the model to fix them, without crashing."""
    model = scripted(
        ModelResponse(parts=[ToolCallPart("search_docs", {"q": "vpn"})]),
        ModelResponse(parts=[ToolCallPart("search_docs", {"query": "vpn"})]),
        ModelResponse(parts=[TextPart("Source: VPN connection")]),
    )
    answer, used = run_agent("vpn", model=model, verbose=False)
    assert used == ["search_docs", "search_docs"]
    assert "VPN connection" in answer
