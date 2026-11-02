"""IT helpdesk agent without a framework: a simple tool-calling loop."""
import json
import os
import time

from dotenv import load_dotenv

from tools import create_ticket, search_docs

load_dotenv()
MODEL = os.getenv("MODEL", "openai/gpt-4o-mini")
MAX_STEPS = 5

SYSTEM_PROMPT = """You are an IT helpdesk assistant for company employees.
Rules:
1. Always call search_docs before answering an IT question.
2. Answer ONLY from the search results. Never use your general knowledge.
3. End every answer with: Source: <section name>
4. If search_docs returns no results, or the results do not answer the question,
   say you don't know, call create_ticket, and give the employee the ticket id.
5. Never ask for, repeat, or store a password. If the employee shares one,
   tell them not to share passwords and to use the self-service password page.
6. Keep answers short and clear."""

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_docs",
            "description": "Search the IT helpdesk guides. Returns matching sections, or [] if nothing matches.",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string", "description": "The employee's question"}},
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "create_ticket",
            "description": "Open an IT support ticket when the guides do not answer the question.",
            "parameters": {
                "type": "object",
                "properties": {
                    "summary": {"type": "string", "description": "Short title of the problem"},
                    "details": {"type": "string", "description": "What the employee asked, without any password"},
                },
                "required": ["summary", "details"],
            },
        },
    },
]


def make_client():
    from openai import OpenAI
    return OpenAI(base_url="https://openrouter.ai/api/v1", api_key=os.getenv("OPENROUTER_API_KEY"))


def run_tool(name, arguments):
    """Run one tool. Bad arguments return an error instead of crashing."""
    try:
        args = json.loads(arguments or "{}")
        if name == "search_docs":
            return search_docs(args["query"])
        if name == "create_ticket":
            return create_ticket(args["summary"], args["details"])
        return {"error": f"Unknown tool: {name}"}
    except (json.JSONDecodeError, KeyError, TypeError) as e:
        return {"error": f"Bad arguments for {name}: {e}"}


def ask_model(client, model, messages, retries=3, wait=15):
    """Call the model. If it is busy (error 429), wait and try again."""
    for attempt in range(1, retries + 1):
        try:
            return client.chat.completions.create(model=model, messages=messages, tools=TOOLS)
        except Exception as e:
            if getattr(e, "status_code", None) != 429 or attempt == retries:
                raise
            print(f"  [busy] Model is rate-limited. Waiting {wait}s (try {attempt}/{retries})...")
            time.sleep(wait)


def run_agent(question, client=None, model=MODEL, verbose=True):
    """Answer one question. Returns (answer, list of tools used)."""
    client = client or make_client()
    messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": question}]
    tools_used = []
    for _ in range(MAX_STEPS):
        response = ask_model(client, model, messages)
        msg = response.choices[0].message
        if not msg.tool_calls:
            return msg.content, tools_used
        messages.append({
            "role": "assistant",
            "content": msg.content or "",
            "tool_calls": [
                {"id": c.id, "type": "function",
                 "function": {"name": c.function.name, "arguments": c.function.arguments}}
                for c in msg.tool_calls
            ],
        })
        for call in msg.tool_calls:
            result = run_tool(call.function.name, call.function.arguments)
            tools_used.append(call.function.name)
            if verbose:
                print(f"  [tool] {call.function.name}({call.function.arguments})")
            messages.append({"role": "tool", "tool_call_id": call.id, "content": json.dumps(result)})
    return "Sorry, I could not finish. Please contact IT.", tools_used


if __name__ == "__main__":
    print(f"IT Helpdesk Agent (model: {MODEL}). Type 'exit' to quit.")
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
