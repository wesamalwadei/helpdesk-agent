"""Web interface for the IT helpdesk agent. Run with: streamlit run app.py"""
import json

import streamlit as st

import tools
from helpdesk_agent import MODEL
from helpdesk_agent import run_agent as run_plain
from helpdesk_agent_pydantic import run_agent as run_pydantic
from helpdesk_team import run_team

TEAM = "Multi-agent team"
AGENTS = {TEAM: None, "Without framework": run_plain, "Pydantic AI": run_pydantic}
DEMO_QUESTIONS = [
    "My VPN is not connecting, what should I do?",
    "My laptop shows a blue screen after an update, how do I fix it?",
    "My password is Abc123, can you reset it for me?",
    "How do I take a screenshot on Windows?",
]

st.set_page_config(page_title="IT Helpdesk Agent")
st.title("IT Helpdesk Agent")
st.caption("Gives solutions from the IT guides with the source. Say \"open a ticket\" any time to reach a technician.")

if "messages" not in st.session_state:
    st.session_state.messages = []


def load_tickets():
    if tools.TICKETS_PATH.exists():
        return json.loads(tools.TICKETS_PATH.read_text(encoding="utf-8"))
    return []


# ----- Sidebar -----
with st.sidebar:
    version = st.radio("Agent version", list(AGENTS))
    st.caption(f"Model: {MODEL}")
    if st.button("Clear chat"):
        st.session_state.messages = []
        st.rerun()

    st.subheader("Demo questions")
    clicked = None
    for q in DEMO_QUESTIONS:
        if st.button(q, use_container_width=True):
            clicked = q

    st.subheader("Open tickets")
    tickets = load_tickets()
    if not tickets:
        st.caption("No tickets yet.")
    for t in reversed(tickets[-5:]):
        extra = f" · {t['priority']} · {t['category']}" if "priority" in t else ""
        st.markdown(f"**{t['id']}**: {t['summary']}{extra}")

# ----- Chat history -----
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
        if m.get("tools"):
            st.caption(f"Steps: {' -> '.join(m['tools'])}  ({m['version']})")

# ----- New question -----
question = st.chat_input("Ask an IT question...") or clicked
if question:
    # Earlier messages, so the team understands follow-ups like "can you open a ticket?"
    history = [m for m in st.session_state.messages if m.get("version", TEAM) == TEAM or m["role"] == "user"]
    user_msg = {"role": "user", "content": question, "has_password": False}
    st.session_state.messages.append(user_msg)
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Searching the IT guides..."):
            try:
                issues = []
                if version == TEAM:
                    result = run_team(question, verbose=False, history=history)
                    answer, used, issues = result["answer"], result["steps"], result["issues"]
                    user_msg["has_password"] = result["triage"].shares_password
                else:
                    answer, used = AGENTS[version](question, verbose=False)
            except Exception as e:
                answer, used, issues = f"Error: {e}\n\nTry again in a minute, or change MODEL in .env.", [], []
        st.markdown(answer)
        if used:
            st.caption(f"Steps: {' -> '.join(used)}  ({version})")

    st.session_state.messages.append(
        # "issues" remembers which problems this answer was about, so a later
        # "can you open a ticket?" is about the right one.
        {"role": "assistant", "content": answer, "tools": used, "version": version, "issues": issues}
    )
    if "create_ticket" in used:
        st.rerun()  # refresh the ticket list in the sidebar
