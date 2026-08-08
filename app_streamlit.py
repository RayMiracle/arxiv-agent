"""
ArXiv Research Assistant — Streamlit web UI
=============================================
A browser-based front end for core.py's ResearchAgent and persistence
layer. Shares conversations-history.json with the CLI (arxiv-agent.py) —
a conversation started in one can be resumed in the other.

Run with:  streamlit run app_streamlit.py
"""

import re

import anthropic
import streamlit as st
from datetime import datetime
from pathlib import Path

from core import (
    ResearchAgent,
    load_conversations,
    save_conversations,
    generate_topic_summary,
    generate_resume_summary,
    render_conversation_markdown,
    export_filename,
)

SCRIPT_FOLDER = Path(__file__).parent

st.set_page_config(page_title="ArXiv Research Assistant", page_icon="📚")


def _tame_headings(text: str) -> str:
    """
    Demote Markdown headings so Claude's replies never render as giant page
    titles inside a chat bubble. '#'/'##'/'###' (page/section-title sized)
    become '####'/'#####' (small sub-heading sized); anything already at
    '####'+ is left alone.
    """
    def demote(match: re.Match) -> str:
        hashes = match.group(1)
        return "#" * min(len(hashes) + 3, 6) + " "

    return re.sub(r"^(#{1,3}) ", demote, text, flags=re.MULTILINE)


# ---------------------------------------------------------------------------
# Session state helpers
# ---------------------------------------------------------------------------

def _start_new_conversation(language: str) -> None:
    """Reset session state for a fresh conversation."""
    now = datetime.now()
    st.session_state.agent = ResearchAgent(language=language)
    st.session_state.language = language
    st.session_state.conv_id = now.strftime("%Y%m%d_%H%M%S")
    st.session_state.conv_timestamp = now.isoformat()
    st.session_state.last_papers = None
    st.session_state.resume_summary = None


def _resume_conversation(conv: dict, client: anthropic.Anthropic) -> None:
    """Restore session state from a saved conversation."""
    agent = ResearchAgent(language=conv["language"])
    agent.history = list(conv["history"])
    agent._has_paper_context = any(
        "[ArXiv search results" in msg["content"]
        for msg in agent.history
        if msg["role"] == "user"
    )
    st.session_state.agent = agent
    st.session_state.language = conv["language"]
    st.session_state.conv_id = conv["id"]
    st.session_state.conv_timestamp = conv["timestamp"]
    st.session_state.last_papers = None
    st.session_state.resume_summary = generate_resume_summary(
        client, agent.history, conv["language"]
    )


def _persist_current_conversation(client: anthropic.Anthropic) -> None:
    """Save the current session's history to conversations-history.json."""
    agent = st.session_state.agent
    if not agent.history:
        return
    topic = generate_topic_summary(client, agent.history, st.session_state.language)
    updated_conv = {
        "id": st.session_state.conv_id,
        "timestamp": st.session_state.conv_timestamp,
        "language": st.session_state.language,
        "topic_summary": topic,
        "history": agent.history,
    }
    conversations = load_conversations()
    existing_idx = next(
        (i for i, c in enumerate(conversations) if c["id"] == st.session_state.conv_id),
        None,
    )
    if existing_idx is not None:
        conversations[existing_idx] = updated_conv
    else:
        conversations.append(updated_conv)
    save_conversations(conversations)


# ---------------------------------------------------------------------------
# Initialize session state on first load
# ---------------------------------------------------------------------------

if "agent" not in st.session_state:
    _start_new_conversation(language="en")

client = anthropic.Anthropic()


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.header("Settings")

    # Language is locked once a conversation has messages, matching the CLI
    # (switching languages mid-chat would desync the system prompt already
    # baked into the running agent).
    chat_started = bool(st.session_state.agent.history)
    lang_options = {"English": "en", "Czech (Čeština)": "cs"}
    lang_labels = list(lang_options.keys())
    current_label = "English" if st.session_state.language == "en" else "Czech (Čeština)"
    chosen_label = st.radio(
        "Language / Jazyk",
        lang_labels,
        index=lang_labels.index(current_label),
        disabled=chat_started,
    )
    if not chat_started and lang_options[chosen_label] != st.session_state.language:
        _start_new_conversation(language=lang_options[chosen_label])
        st.rerun()

    if st.button("New conversation", use_container_width=True):
        _start_new_conversation(language=st.session_state.language)
        st.rerun()

    st.divider()
    st.subheader("Saved conversations")

    saved = load_conversations()
    if not saved:
        st.caption("No saved conversations yet.")
    else:
        # Most recent first
        for conv in sorted(saved, key=lambda c: c["timestamp"], reverse=True):
            ts = conv["timestamp"][:16].replace("T", " ")
            with st.expander(f"{ts} — {conv.get('topic_summary', '?')}"):
                if st.button("Resume", key=f"resume_{conv['id']}", use_container_width=True):
                    _resume_conversation(conv, client)
                    st.rerun()
                st.download_button(
                    "Export",
                    data=render_conversation_markdown(conv),
                    file_name=export_filename(conv),
                    mime="text/markdown",
                    key=f"export_{conv['id']}",
                    use_container_width=True,
                )
                if st.button("Delete", key=f"delete_{conv['id']}", use_container_width=True):
                    st.session_state[f"confirm_delete_{conv['id']}"] = True
                if st.session_state.get(f"confirm_delete_{conv['id']}"):
                    st.warning("Delete this conversation?")
                    if st.button("Confirm delete", key=f"confirm_{conv['id']}"):
                        remaining = [c for c in saved if c["id"] != conv["id"]]
                        save_conversations(remaining)
                        del st.session_state[f"confirm_delete_{conv['id']}"]
                        st.rerun()


# ---------------------------------------------------------------------------
# Main area
# ---------------------------------------------------------------------------

st.title("📚 ArXiv Research Assistant")

st.info(
    "You are interacting with an AI system. Answers are generated by an LLM "
    "(Claude) and may be incomplete or inaccurate — verify against the cited "
    "papers before relying on them."
)

if st.session_state.resume_summary:
    st.info(_tame_headings(st.session_state.resume_summary))

# Render chat history (strip the injected ArXiv context block, same as the
# Markdown exporter does, so the raw search payload isn't shown to the user)
for msg in st.session_state.agent.history:
    role = "user" if msg["role"] == "user" else "assistant"
    content = msg["content"]
    if role == "user":
        content = content.split("[ArXiv search results")[0].strip()
    else:
        content = _tame_headings(content)
    with st.chat_message(role):
        st.markdown(content)

# Papers fetched on the most recent SEARCH turn, shown as cards
if st.session_state.last_papers:
    st.subheader("Papers found")
    for paper in st.session_state.last_papers:
        authors = ", ".join(paper["authors"][:3])
        if len(paper["authors"]) > 3:
            authors += " et al."
        with st.expander(paper["title"]):
            st.markdown(f"**Authors:** {authors}")
            st.markdown(f"**Published:** {paper['published']}")
            st.markdown(f"**Link:** {paper['url']}")

# Export current session
if st.session_state.agent.history:
    temp_conv = {
        "id": st.session_state.conv_id,
        "timestamp": st.session_state.conv_timestamp,
        "language": st.session_state.language,
        "topic_summary": "conversation",
        "history": st.session_state.agent.history,
    }
    st.download_button(
        "Export current session to Markdown",
        data=render_conversation_markdown(temp_conv),
        file_name=export_filename(temp_conv),
        mime="text/markdown",
    )

user_input = st.chat_input("Ask me to find or explain research papers...")
if user_input:
    # Show the user's message right away — agent.ask() can take several
    # seconds (classification + possible ArXiv search + Claude reply), and
    # without this the UI looks frozen with no sign the input was received.
    with st.chat_message("user"):
        st.markdown(user_input)
    with st.chat_message("assistant"):
        with st.spinner("Searching ArXiv and thinking…"):
            try:
                reply, papers = st.session_state.agent.ask(user_input)
                st.session_state.last_papers = papers
            except anthropic.APIError as e:
                reply, papers = f"[API error] {e}", None
                st.session_state.last_papers = None
        st.markdown(_tame_headings(reply))
    _persist_current_conversation(client)
    st.rerun()
