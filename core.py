"""
ArXiv Research Assistant — core logic
======================================
Shared module used by both the CLI (arxiv-agent.py) and the Streamlit web
UI (app_streamlit.py).

Architecture:
  - search_arxiv()   : pure data function, no AI involved
  - ResearchAgent    : core logic (Claude + search), no I/O
  - persistence layer: conversations-history.json read/write, Markdown export

Neither this module nor anything it imports performs any CLI or web I/O
(no print/input/streamlit calls) — that keeps it reusable across UIs.
"""

import json
import re
import threading
import arxiv
import anthropic
from datetime import date, datetime
from pathlib import Path
from dotenv import load_dotenv

# Load ANTHROPIC_API_KEY from the .env file
load_dotenv()

_MODEL = "claude-haiku-4-5-20251001"

# claude-haiku-4-5 pricing, USD per token (list rate: $1.00 / $5.00 per 1M tokens)
_PRICE_PER_INPUT_TOKEN = 1.00 / 1_000_000
_PRICE_PER_OUTPUT_TOKEN = 5.00 / 1_000_000


# ---------------------------------------------------------------------------
# 0. LLM usage tracking (cost visibility)
# ---------------------------------------------------------------------------

# usage-log.jsonl lives in the same folder as this module — one JSON line per
# Claude API call, so the log is append-only and safe to tail/parse per-line.
_USAGE_LOG_FILE = Path(__file__).parent / "usage-log.jsonl"

# Running total for the current process (CLI run or Streamlit server process).
# Streamlit reruns the whole script per interaction but keeps the process
# alive, so this persists across reruns within one server session.
_session_usage_lock = threading.Lock()
_session_usage = {
    "call_count": 0,
    "input_tokens": 0,
    "output_tokens": 0,
    "cost_usd": 0.0,
}


def _record_usage(call_type: str, response) -> None:
    """
    Record token usage and cost for one Claude API call.

    Updates the in-process running total (see get_session_usage()) and
    appends a line to usage-log.jsonl for durable, cross-session history.
    """
    input_tokens = response.usage.input_tokens
    output_tokens = response.usage.output_tokens
    cost_usd = (
        input_tokens * _PRICE_PER_INPUT_TOKEN
        + output_tokens * _PRICE_PER_OUTPUT_TOKEN
    )

    with _session_usage_lock:
        _session_usage["call_count"] += 1
        _session_usage["input_tokens"] += input_tokens
        _session_usage["output_tokens"] += output_tokens
        _session_usage["cost_usd"] += cost_usd

    entry = {
        "timestamp": datetime.now().isoformat(),
        "call_type": call_type,       # e.g. "classify", "search_query", "ask", "topic_summary"
        "model": _MODEL,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "cost_usd": round(cost_usd, 6),
    }
    with open(_USAGE_LOG_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def get_session_usage() -> dict:
    """
    Return a snapshot of this process's cumulative usage so far:
    {call_count, input_tokens, output_tokens, cost_usd}.
    """
    with _session_usage_lock:
        return dict(_session_usage)


def load_usage_log() -> list[dict]:
    """
    Load the full usage log from disk (all calls ever made, across sessions).
    Returns an empty list if the file does not exist or is unreadable.
    """
    if not _USAGE_LOG_FILE.exists():
        return []
    entries = []
    with open(_USAGE_LOG_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError:
                continue  # skip a corrupted line rather than failing the whole log
    return entries


# ---------------------------------------------------------------------------
# 1. ArXiv search helper
# ---------------------------------------------------------------------------

def search_arxiv(query: str, max_results: int = 5) -> list[dict]:
    """
    Search ArXiv for papers matching *query* and return structured results.

    Parameters
    ----------
    query       : the search string (e.g. "transformer attention mechanism")
    max_results : how many papers to fetch (default 5)

    Returns
    -------
    A list of dicts, one per paper, each containing:
        title     : str
        authors   : list[str]
        summary   : str  (the abstract)
        published : str  (ISO-format date)
        url       : str  (canonical ArXiv link)
    """
    client = arxiv.Client()

    search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.Relevance,
    )

    papers = []
    for result in client.results(search):
        papers.append({
            "title":     result.title,
            "authors":   [str(a) for a in result.authors],
            "summary":   result.summary,
            "published": result.published.strftime("%Y-%m-%d"),
            "url":       result.entry_id,          # permanent ArXiv URL
        })

    return papers


# ---------------------------------------------------------------------------
# 2. Core agent (no CLI logic lives here)
# ---------------------------------------------------------------------------

def _build_system_prompt(language: str) -> str:
    """
    Build the system prompt for Claude based on the chosen language.

    Parameters
    ----------
    language : 'en' for English, 'cs' for Czech
    """
    base = (
        f"Today's date is {date.today()}.\n\n"
        "You are a helpful research assistant that specialises in academic papers from ArXiv.\n\n"
        "When the user asks about a topic, you will be given relevant paper summaries as "
        "context. Use those papers to give accurate, well-organised answers.\n\n"
        "Guidelines:\n"
        "- Cite papers by title when you reference them.\n"
        "- If the context does not contain enough information to answer, say so clearly.\n"
        "- Keep explanations accessible — avoid unnecessary jargon.\n"
        "- For follow-up questions, use the papers already in the conversation.\n"
        "- Formatting: you may use Markdown, but keep it light. Use bold text or "
        "small '###' sub-headings for structure — never '#' or '##', which render "
        "as oversized page titles in the UI. Prefer short paragraphs and bullet "
        "lists over deep heading hierarchies.\n"
    )

    if language == "cs":
        # Czech-specific instructions appended to the shared base
        base += (
            "\nLanguage instructions:\n"
            "- Always respond in Czech (česky).\n"
            "- Use simple, clear Czech suitable for someone still learning the field.\n"
            "- Keep technical terms in English but add a brief Czech explanation in "
            "brackets immediately after, e.g. 'transformer (síť pro zpracování sekvencí)'.\n"
            "- Paper titles and author names should be left in their original language.\n"
            "- The source abstracts may contain non-English, non-Czech text (e.g. "
            "Cyrillic script). Never copy such fragments verbatim into your answer — "
            "translate the meaning into Czech, except for proper nouns (author names, "
            "paper titles) which stay as-is."
        )

    return base


class ResearchAgent:
    """
    Manages a multi-turn conversation with Claude about ArXiv papers.

    The agent decides on each turn whether the user is:
      (a) asking about a NEW topic  → fetch fresh papers, add them as context
      (b) asking a FOLLOW-UP        → use the papers already in history

    Usage
    -----
        agent = ResearchAgent()
        reply, papers = agent.ask("Find papers about diffusion models")
        reply, papers = agent.ask("Which of those papers is most beginner-friendly?")
    """

    def __init__(self, language: str = "en"):
        """
        Parameters
        ----------
        language : 'en' for English (default), 'cs' for Czech
        """
        # Anthropic client reads ANTHROPIC_API_KEY from the environment
        self._client = anthropic.Anthropic()

        # Build the system prompt once based on the chosen language
        self._system_prompt = _build_system_prompt(language)

        # Full conversation history: list of {"role": ..., "content": ...} dicts
        # This is passed to Claude on every request so it remembers context.
        self.history: list[dict] = []

        # Remember whether we have already loaded papers into the conversation.
        # Once True, follow-up questions skip a new ArXiv search.
        self._has_paper_context: bool = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def ask(self, user_message: str) -> tuple[str, list[dict] | None]:
        """
        Send *user_message* to Claude and return its reply.

        Steps
        -----
        1. Classify the message as SEARCH, FOLLOWUP, or UNCLEAR.
        2. If SEARCH: translate to an English query, call search_arxiv(), and
           prepend the results as context.
           If FOLLOWUP or UNCLEAR: use the message as-is (no search).
        3. Append the (possibly enriched) message to history.
        4. Call Claude with the full history.
        5. Append Claude's reply to history.
        6. Return Claude's reply and (if a search happened) the papers found.

        Returns
        -------
        (reply, papers) — papers is the list of freshly fetched papers when
        this turn triggered a SEARCH, otherwise None.
        """
        # Step 1 — classify the message
        intent = self._classify_message(user_message)

        papers: list[dict] | None = None

        if intent == "SEARCH":
            # Step 2a — get an English query for ArXiv (translates if necessary)
            search_query = self._to_english_query(user_message)
            papers = search_arxiv(search_query, max_results=5)

            if papers:
                # Format papers as a readable context block
                context_block = self._format_papers(papers)
                enriched_message = (
                    f"{user_message}\n\n"
                    f"[ArXiv search results for your reference]\n{context_block}"
                )
                self._has_paper_context = True
            else:
                enriched_message = (
                    f"{user_message}\n\n"
                    "[No ArXiv papers were found for this query.]"
                )
        else:
            # Step 2b — FOLLOWUP or UNCLEAR: pass message straight through.
            # For UNCLEAR, Claude will ask the user for clarification on its own.
            enriched_message = user_message

        # Step 3 — add user turn to history
        self.history.append({"role": "user", "content": enriched_message})

        # Step 4 — call Claude
        response = self._client.messages.create(
            model=_MODEL,
            max_tokens=1024,
            system=self._system_prompt,   # built from the chosen language
            messages=self.history,
        )
        _record_usage("ask", response)

        assistant_reply = response.content[0].text

        # Step 5 — add Claude's reply to history
        self.history.append({"role": "assistant", "content": assistant_reply})

        # Step 6 — return the reply and any freshly fetched papers
        return assistant_reply, papers

    def reset(self):
        """Clear conversation history so you can start a fresh topic."""
        self.history.clear()
        self._has_paper_context = False

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _to_english_query(self, message: str) -> str:
        """
        Extract a short, English-language ArXiv search query from *message*.

        If the message is already in English this typically returns the key
        topic words unchanged.  If it is in another language (e.g. Czech),
        Claude translates and distils it into a concise English query.

        A low max_tokens value keeps this call fast and cheap.
        """
        prompt = (
            "Extract the core academic search topic from the user's message and "
            "express it as a short English query suitable for searching ArXiv "
            "(2–6 words, no punctuation, no filler words).\n"
            "If the message is not in English, translate the topic to English first.\n\n"
            f"User message: \"{message}\"\n\n"
            "Reply with only the search query, nothing else."
        )

        response = self._client.messages.create(
            model=_MODEL,
            max_tokens=30,          # a short query is all we need
            messages=[{"role": "user", "content": prompt}],
        )
        _record_usage("search_query", response)

        return response.content[0].text.strip()

    def _classify_message(self, message: str) -> str:
        """
        Ask Claude to classify the user's message into one of three categories.

        Returns
        -------
        'SEARCH'   — the user is asking about a new topic; do a fresh ArXiv search.
        'FOLLOWUP' — the user is asking a follow-up about papers already in context.
        'UNCLEAR'  — the message is gibberish, too vague, or lacks a real topic;
                     skip the search and let Claude ask for clarification.

        Uses max_tokens=10 to keep this call fast and cheap.

        When no papers are in context yet (first message), only SEARCH and UNCLEAR
        are offered — FOLLOWUP makes no sense before any papers have been found.
        """
        if not self._has_paper_context:
            # First message: no prior papers, so FOLLOWUP is not a valid option.
            classification_prompt = (
                "You are a routing assistant for a research tool that searches ArXiv.\n"
                "This is the user's first message; no papers have been found yet.\n\n"
                f"User message: \"{message}\"\n\n"
                "Classify this message into exactly one of these two categories:\n"
                "  SEARCH  — the user is asking about a topic and wants papers found.\n"
                "  UNCLEAR — the message is gibberish, too vague, or contains no real topic to search for.\n\n"
                "Reply with exactly one word: SEARCH or UNCLEAR."
            )
            valid_verdicts = {"SEARCH", "UNCLEAR"}
            default = "UNCLEAR"
        else:
            # Subsequent messages: all three options are available.
            classification_prompt = (
                "You are a routing assistant for a research tool that searches ArXiv.\n"
                "Papers from a previous search are already available in the conversation.\n\n"
                f"User message: \"{message}\"\n\n"
                "Classify this message into exactly one of these three categories:\n"
                "  SEARCH   — the user is asking about a NEW topic and needs a fresh ArXiv search.\n"
                "  FOLLOWUP — the user is asking a follow-up question about the papers already found.\n"
                "  UNCLEAR  — the message is gibberish, too vague, or contains no real topic to search for.\n\n"
                "Reply with exactly one word: SEARCH, FOLLOWUP, or UNCLEAR."
            )
            valid_verdicts = {"SEARCH", "FOLLOWUP", "UNCLEAR"}
            default = "FOLLOWUP"

        response = self._client.messages.create(
            model=_MODEL,
            max_tokens=10,          # we only need one word
            messages=[{"role": "user", "content": classification_prompt}],
        )
        _record_usage("classify", response)

        verdict = response.content[0].text.strip().upper()
        # Guard against unexpected output by falling back to the safe default.
        return verdict if verdict in valid_verdicts else default

    @staticmethod
    def _format_papers(papers: list[dict]) -> str:
        """
        Convert a list of paper dicts into a human-readable (and LLM-readable)
        text block that Claude can use as context.
        """
        lines = []
        for i, paper in enumerate(papers, start=1):
            authors = ", ".join(paper["authors"][:3])  # show at most 3 authors
            if len(paper["authors"]) > 3:
                authors += " et al."

            lines.append(
                f"Paper {i}: {paper['title']}\n"
                f"  Authors  : {authors}\n"
                f"  Published: {paper['published']}\n"
                f"  URL      : {paper['url']}\n"
                f"  Abstract : {paper['summary'][:400]}...\n"  # trim very long abstracts
            )
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# 3. Conversation persistence
# ---------------------------------------------------------------------------

# conversations.json lives in the same folder as this module
_CONV_FILE = Path(__file__).parent / "conversations-history.json"

# Maximum number of saved conversations to keep on disk
_MAX_CONVERSATIONS = 12


def load_conversations() -> list[dict]:
    """
    Load saved conversations from the JSON file.
    Returns an empty list if the file does not exist or is unreadable.
    """
    if not _CONV_FILE.exists():
        return []
    try:
        with open(_CONV_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        # File is corrupted or unreadable — start fresh rather than crashing
        return []


def save_conversations(conversations: list[dict]) -> None:
    """
    Write conversations to the JSON file.

    If there are more than _MAX_CONVERSATIONS entries, the oldest ones
    (sorted by timestamp) are dropped so only the newest are kept.
    """
    # Sort oldest-first; slicing from the end keeps the most recent ones
    conversations.sort(key=lambda c: c["timestamp"])
    trimmed = conversations[-_MAX_CONVERSATIONS:]
    with open(_CONV_FILE, "w", encoding="utf-8") as f:
        json.dump(trimmed, f, ensure_ascii=False, indent=2)


def generate_topic_summary(
    client: anthropic.Anthropic,
    history: list[dict],
    language: str,
) -> str:
    """
    Ask Claude to produce a short (5–10 word) English label for the conversation.
    This is used as the display name in the menu and as part of the export filename.
    """
    # Using only the first user message keeps the prompt tiny and cheap
    first_user_text = next(
        (m["content"] for m in history if m["role"] == "user"), ""
    )
    # Strip the injected ArXiv block so Claude sees a clean message
    first_user_text = first_user_text.split("[ArXiv search results")[0].strip()

    prompt = (
        "Summarise the topic of this research conversation in 5–10 words "
        "(no punctuation at the end, always write in English).\n\n"
        f"First user message: \"{first_user_text[:300]}\"\n\n"
        "Reply with only the topic summary."
    )
    response = client.messages.create(
        model=_MODEL,
        max_tokens=25,
        messages=[{"role": "user", "content": prompt}],
    )
    _record_usage("topic_summary", response)
    return response.content[0].text.strip()


def generate_resume_summary(
    client: anthropic.Anthropic,
    history: list[dict],
    language: str,
) -> str:
    """
    Ask Claude for a 2–4 sentence summary of a saved conversation.
    Displayed to the user when they choose to resume a previous session.
    The summary is written in the same language as the conversation.
    """
    # Build a short transcript from the first 10 messages to keep the call cheap
    lines = []
    for msg in history[:10]:
        role = "User" if msg["role"] == "user" else "Assistant"
        # Strip the injected ArXiv block — it's noise for a summary
        content = msg["content"].split("[ArXiv search results")[0].strip()
        lines.append(f"{role}: {content[:200]}")
    transcript = "\n".join(lines)

    lang_instruction = (
        "Odpověz v češtině." if language == "cs" else "Reply in English."
    )
    prompt = (
        f"{lang_instruction}\n"
        "Write a 2–4 sentence summary of what was discussed in this research "
        "conversation. Mention the main topic and any key papers or findings.\n\n"
        f"Conversation excerpt:\n{transcript}\n\n"
        "Summary:"
    )
    response = client.messages.create(
        model=_MODEL,
        max_tokens=300,
        messages=[{"role": "user", "content": prompt}],
    )
    _record_usage("resume_summary", response)
    return response.content[0].text.strip()


def export_to_markdown(conv: dict, folder: Path) -> str:
    """
    Write a conversation to a Markdown file in *folder*.

    The filename is derived from the topic summary and conversation timestamp
    so it is human-readable and unique.

    Returns the filename (not the full path) that was written.
    """
    # Build a filesystem-safe filename from the topic and timestamp
    safe_topic = re.sub(r"[^\w\s-]", "", conv.get("topic_summary", "conversation"))
    safe_topic = re.sub(r"\s+", "_", safe_topic).strip("_")[:50]
    ts = conv["timestamp"][:16].replace(":", "-").replace("T", "_")
    filename = f"{safe_topic}_{ts}-export.md"
    filepath = folder / filename

    with open(filepath, "w", encoding="utf-8") as f:
        # Header block
        f.write(f"# {conv.get('topic_summary', 'Research Conversation')}\n\n")
        f.write(f"**Date:** {conv['timestamp'][:10]}  \n")
        f.write(
            f"**Language:** {'Czech' if conv['language'] == 'cs' else 'English'}\n\n"
        )
        f.write("---\n\n")

        # One section per message
        for msg in conv["history"]:
            if msg["role"] == "user":
                # Hide the injected ArXiv block — it clutters the exported doc
                content = msg["content"].split("[ArXiv search results")[0].strip()
                f.write(f"## You\n\n{content}\n\n---\n\n")
            else:
                f.write(f"## Assistant\n\n{msg['content']}\n\n---\n\n")

    return filename


def render_conversation_markdown(conv: dict) -> str:
    """
    Build the same Markdown content that export_to_markdown() writes to disk,
    but return it as a string instead of writing a file.

    Used by the Streamlit UI to power st.download_button, which needs bytes
    in memory rather than a filepath.
    """
    lines = [
        f"# {conv.get('topic_summary', 'Research Conversation')}\n",
        f"**Date:** {conv['timestamp'][:10]}  ",
        f"**Language:** {'Czech' if conv['language'] == 'cs' else 'English'}\n",
        "---\n",
    ]
    for msg in conv["history"]:
        if msg["role"] == "user":
            content = msg["content"].split("[ArXiv search results")[0].strip()
            lines.append(f"## You\n\n{content}\n\n---\n")
        else:
            lines.append(f"## Assistant\n\n{msg['content']}\n\n---\n")
    return "\n".join(lines)


def export_filename(conv: dict) -> str:
    """Compute the same filename export_to_markdown() would use, without writing a file."""
    safe_topic = re.sub(r"[^\w\s-]", "", conv.get("topic_summary", "conversation"))
    safe_topic = re.sub(r"\s+", "_", safe_topic).strip("_")[:50]
    ts = conv["timestamp"][:16].replace(":", "-").replace("T", "_")
    return f"{safe_topic}_{ts}-export.md"
