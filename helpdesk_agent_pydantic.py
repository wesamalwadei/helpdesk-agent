"""IT helpdesk agent built with the Pydantic AI framework.

Same tools, same rules and same model as helpdesk_agent.py.
The difference: Pydantic AI runs the tool-calling loop for us.
"""
import time

from pydantic_ai import Agent
from pydantic_ai.messages import ModelResponse, ToolCallPart

import tools
from helpdesk_agent import MODEL, SYSTEM_PROMPT  # reuse the same rules and model


def search_docs(query: str) -> list[dict]:
    """Search the IT helpdesk guides. Returns matching sections, or [] if nothing matches.

    Args:
        query: The employee's question.
    """
    return tools.search_docs(query)


def create_ticket(summary: str, details: str) -> dict:
    """Open an IT support ticket when the guides do not answer the question.

    Args:
        summary: Short title of the problem.
        details: What the employee asked, without any password.
    """
    return tools.create_ticket(summary, details, path=tools.TICKETS_PATH)


agent = Agent(instructions=SYSTEM_PROMPT, tools=[search_docs, create_ticket])


def make_model(name=MODEL):
    from pydantic_ai.models.openrouter import OpenRouterModel
    return OpenRouterModel(name)  # reads OPENROUTER_API_KEY from .env


def tools_called(messages):
    """List the tool names the model called, in order."""
    return [
        part.tool_name
        for msg in messages if isinstance(msg, ModelResponse)
        for part in msg.parts if isinstance(part, ToolCallPart)
    ]


def run_agent(question, model=None, verbose=True, retries=3, wait=15):
    """Answer one question. Returns (answer, list of tools used), like helpdesk_agent.run_agent."""
    model = model or make_model()
    for attempt in range(1, retries + 1):
        try:
            result = agent.run_sync(question, model=model)
            break
        except Exception as e:
            if getattr(e, "status_code", None) != 429 or attempt == retries:
                raise
            print(f"  [busy] Model is rate-limited. Waiting {wait}s (try {attempt}/{retries})...")
            time.sleep(wait)
    used = tools_called(result.all_messages())
    if verbose:
        for name in used:
            print(f"  [tool] {name}")
    return result.output, used


if __name__ == "__main__":
    print(f"IT Helpdesk Agent - Pydantic AI (model: {MODEL}). Type 'exit' to quit.")
    while True:
        question = input("\nYou: ").strip()
        if question.lower() in {"exit", "quit"}:
            break
        if question:
            try:
                answer, _ = run_agent(question)
                print(f"Agent: {answer}")
            except Exception as e:
                print(f"Error: {e}\nTry again in a minute, or change MODEL in .env.")
