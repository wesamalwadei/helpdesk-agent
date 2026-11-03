"""Web interface for the IT helpdesk agent. Run with: streamlit run app.py"""
import json

import streamlit as st

import tools
from helpdesk_agent import MODEL
from helpdesk_agent import run_agent as run_plain
from helpdesk_agent_pydantic import run_agent as run_pydantic

AGENTS = {"Without framework": run_plain, "Pydantic AI": run_pydantic}
DEMO_QUESTIONS = [
    "My VPN is not connecting, what should I do?",
    "My laptop shows a blue screen after an update, how do I fix it?",
    "My password is Abc123, can you reset it for me?",
]

st.set_page_config(page_title="IT Helpdesk Agent")
st.title("IT Helpdesk Agent")
st.caption("Answers from the IT guides, cites the source, and opens a ticket when it can't find an answer.")

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
        st.markdown(f"**{t['id']}**: {t['summary']}")

# ----- Chat history -----
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
        if m.get("tools"):
            st.caption(f"Tools used: {' -> '.join(m['tools'])}  ({m['version']})")

# ----- New question -----
question = st.chat_input("Ask an IT question...") or clicked
if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Searching the IT guides..."):
            try:
                answer, used = AGENTS[version](question, verbose=False)
            except Exception as e:
                answer, used = f"Error: {e}\n\nTry again in a minute, or change MODEL in .env.", []
        st.markdown(answer)
        if used:
            st.caption(f"Tools used: {' -> '.join(used)}  ({version})")

    st.session_state.messages.append(
        {"role": "assistant", "content": answer, "tools": used, "version": version}
    )
    if "create_ticket" in used:
        st.rerun()  # refresh the ticket list in the sidebar
