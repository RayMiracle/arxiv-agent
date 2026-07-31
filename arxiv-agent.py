"""
ArXiv Research Assistant
========================
An interactive CLI tool that uses Claude to help you explore academic papers.

Architecture:
  - search_arxiv()   : pure data function, no AI involved
  - ResearchAgent    : core logic (Claude + search), no I/O
  - main()           : CLI interface only

This separation means you can swap main() for a web UI later without
touching the agent logic.
"""

import json
import re
import arxiv
import anthropic
from datetime import date, datetime
from pathlib import Path
from dotenv import load_dotenv

# Load ANTHROPIC_API_KEY from the .env file
load_dotenv()


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
    )

    if language == "cs":
        # Czech-specific instructions appended to the shared base
        base += (
            "\nLanguage instructions:\n"
            "- Always respond in Czech (česky).\n"
            "- Use simple, clear Czech suitable for someone still learning the field.\n"
            "- Keep technical terms in English but add a brief Czech explanation in "
            "brackets immediately after, e.g. 'transformer (síť pro zpracování sekvencí)'.\n"
            "- Paper titles and author names should be left in their original language."
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
        reply = agent.ask("Find papers about diffusion models")
        reply = agent.ask("Which of those papers is most beginner-friendly?")
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

    def ask(self, user_message: str) -> str:
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
        6. Return Claude's reply.
        """
        # Step 1 — classify the message
        intent = self._classify_message(user_message)

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
            model="claude-haiku-4-5-20251001",
            max_tokens=1024,
            system=self._system_prompt,   # built from the chosen language
            messages=self.history,
        )

        assistant_reply = response.content[0].text

        # Step 5 — add Claude's reply to history
        self.history.append({"role": "assistant", "content": assistant_reply})

        # Step 6 — return the reply
        return assistant_reply

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
            model="claude-haiku-4-5-20251001",
            max_tokens=30,          # a short query is all we need
            messages=[{"role": "user", "content": prompt}],
        )

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
            model="claude-haiku-4-5-20251001",
            max_tokens=10,          # we only need one word
            messages=[{"role": "user", "content": classification_prompt}],
        )

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

# conversations.json lives in the same folder as this script
_CONV_FILE = Path(__file__).parent / "conversations-history.json"

# Maximum number of saved conversations to keep on disk
_MAX_CONVERSATIONS = 12


def _load_conversations() -> list[dict]:
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


def _save_conversations(conversations: list[dict]) -> None:
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


def _generate_topic_summary(
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
        model="claude-haiku-4-5-20251001",
        max_tokens=25,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.content[0].text.strip()


def _generate_resume_summary(
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
        model="claude-haiku-4-5-20251001",
        max_tokens=300,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.content[0].text.strip()


def _export_to_markdown(conv: dict, folder: Path) -> str:
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


def _show_startup_menu(
    language: str,
    conversations: list[dict],
    client: anthropic.Anthropic,
    folder: Path,
) -> tuple:
    """
    Display the saved-conversations menu at startup.

    The user can:
      - Enter a number to resume a conversation
      - Press N to start a new one
      - Press D to delete a saved conversation
      - Press E to export a saved conversation to Markdown

    Returns a tuple (action, conv_or_None) where action is "new" or "resume".
    """
    # Bilingual labels — index 0 = English, index 1 = Czech
    i = 0 if language == "en" else 1

    while True:
        print()
        print(("Saved conversations:", "Uložené konverzace:")[i])
        for idx, conv in enumerate(conversations, start=1):
            # Show a human-friendly timestamp (strip the T separator)
            ts = conv["timestamp"][:16].replace("T", " ")
            print(f"  {idx}. {ts} — {conv.get('topic_summary', '?')}")

        print()
        print(("Options:", "Možnosti:")[i])
        print(("  N — New conversation",          "  N — Nová konverzace")[i])
        print(("  D — Delete a conversation",     "  D — Smazat konverzaci")[i])
        print(("  E — Export to Markdown",         "  E — Exportovat do Markdown")[i])
        print(("  Or enter a number to resume",   "  nebo zadejte číslo pro pokračování")[i])
        print()

        choice = input(("Your choice: ", "Váš výběr: ")[i]).strip().lower()

        # ── New conversation ────────────────────────────────────────────
        if choice == "n":
            return ("new", None)

        # ── Delete ──────────────────────────────────────────────────────
        elif choice == "d":
            raw = input(
                ("Enter number to delete: ", "Zadejte číslo ke smazání: ")[i]
            ).strip()
            if not raw.isdigit() or not (1 <= int(raw) <= len(conversations)):
                print(("Invalid number.", "Neplatné číslo.")[i])
                continue
            num = int(raw) - 1
            confirm = input(
                ("Are you sure? (y/n): ", "Jste si jistý/á? (a/n): ")[i]
            ).strip().lower()
            yes_answers = ({"y"}, {"a", "y"})[i]
            if confirm in yes_answers:
                conversations.pop(num)
                _save_conversations(conversations)
                print(("Deleted.", "Smazáno.")[i])
                if not conversations:
                    # No conversations left — go straight to new
                    return ("new", None)
            else:
                print(("Cancelled.", "Zrušeno.")[i])

        # ── Export ──────────────────────────────────────────────────────
        elif choice == "e":
            raw = input(
                ("Enter number to export: ", "Zadejte číslo pro export: ")[i]
            ).strip()
            if not raw.isdigit() or not (1 <= int(raw) <= len(conversations)):
                print(("Invalid number.", "Neplatné číslo.")[i])
                continue
            num = int(raw) - 1
            conv = conversations[num]
            # Generate topic_summary on-the-fly if it is missing
            if not conv.get("topic_summary"):
                conv["topic_summary"] = _generate_topic_summary(
                    client, conv["history"], conv["language"]
                )
            filename = _export_to_markdown(conv, folder)
            print(("Exported to: ", "Exportováno do: ")[i] + filename)

        # ── Resume by number ────────────────────────────────────────────
        elif choice.isdigit() and 1 <= int(choice) <= len(conversations):
            return ("resume", conversations[int(choice) - 1])

        else:
            print(("Invalid choice, try again.", "Neplatná volba, zkuste znovu.")[i])


# ---------------------------------------------------------------------------
# 4. CLI interface
# ---------------------------------------------------------------------------

def main():
    """
    Run the interactive command-line research assistant.

    The agent and all conversation state live only in this function.
    To add a web UI later, create a ResearchAgent per user session and call
    agent.ask() — no changes to the agent or persistence functions needed.
    """
    script_folder = Path(__file__).parent  # used for JSON file and .md exports

    print("=" * 60)
    print("  ArXiv Research Assistant (powered by Claude)")
    print("=" * 60)
    print(
        "  You are interacting with an AI system. Answers are generated by\n"
        "  an LLM and may be incomplete or inaccurate — verify against the\n"
        "  cited papers before relying on them."
    )
    print("=" * 60)

    # --- Language selection -----------------------------------------------
    print("\nSelect language / Vyberte jazyk:")
    print("  1. English")
    print("  2. Czech (Čeština)")

    language = "en"
    while True:
        lang_choice = input("Your choice (1/2): ").strip()
        if lang_choice == "1":
            language = "en"
            break
        elif lang_choice == "2":
            language = "cs"
            break
        else:
            print("Please enter 1 or 2.")

    # We need a client early because the startup menu may call Claude
    client = anthropic.Anthropic()

    # --- Startup menu (only shown when saved conversations exist) ----------
    conversations = _load_conversations()
    if conversations:
        action, selected_conv = _show_startup_menu(
            language, conversations, client, script_folder
        )
    else:
        action, selected_conv = "new", None

    # --- Set up the agent -------------------------------------------------
    agent = ResearchAgent(language=language)

    if action == "resume" and selected_conv:
        # Restore saved history and paper-context flag
        conv_id        = selected_conv["id"]
        conv_timestamp = selected_conv["timestamp"]
        agent.history  = list(selected_conv["history"])
        agent._has_paper_context = any(
            "[ArXiv search results" in msg["content"]
            for msg in agent.history
            if msg["role"] == "user"
        )
        # Print a short recap so the user remembers where they left off
        print()
        print(
            ("Generating conversation summary…",
             "Načítám shrnutí konverzace…")[0 if language == "en" else 1]
        )
        summary = _generate_resume_summary(client, agent.history, language)
        print()
        print(summary)
        print()
    else:
        # Fresh conversation — assign a new ID based on current time
        now            = datetime.now()
        conv_id        = now.strftime("%Y%m%d_%H%M%S")
        conv_timestamp = now.isoformat()

    # --- Print usage instructions -----------------------------------------
    if language == "cs":
        print("Zeptejte se mě na výzkumné články.")
        print(
            "Příkazy: 'export' (uložit do Markdown), "
            "'exit' / 'quit' / 'konec' / 'ukončit' (pro ukončení programu)"
        )
    else:
        print("Ask me to find or explain research papers.")
        print("Commands: 'export' (save to Markdown),  'exit' / 'quit' (quit)")
    print()

    # Build the exit-command set (Czech extras added when language is Czech)
    exit_commands = {"exit", "quit"}
    if language == "cs":
        exit_commands |= {"konec", "ukončit"}

    # Track whether the user exited explicitly (determines farewell message)
    explicit_exit = False

    # --- Chat loop --------------------------------------------------------
    try:
        while True:
            try:
                user_input = input("> ").strip()
            except (EOFError, KeyboardInterrupt):
                # Ctrl+D or Ctrl+C — no farewell, but we still save below
                print()
                break

            if not user_input:
                continue

            # Explicit exit command
            if user_input.lower() in exit_commands:
                explicit_exit = True
                break

            # In-session export command
            if user_input.lower() == "export":
                if not agent.history:
                    print(
                        ("Nothing to export yet.",
                         "Zatím není co exportovat.")[0 if language == "en" else 1]
                    )
                    continue
                # Build a temporary conv dict to pass to the exporter
                topic = _generate_topic_summary(client, agent.history, language)
                temp_conv = {
                    "id":            conv_id,
                    "timestamp":     conv_timestamp,
                    "language":      language,
                    "topic_summary": topic,
                    "history":       agent.history,
                }
                filename = _export_to_markdown(temp_conv, script_folder)
                label = ("Exported to: ", "Exportováno do: ")[0 if language == "en" else 1]
                print(f"{label}{filename}")
                print()
                continue

            # Normal chat turn
            print()
            try:
                reply = agent.ask(user_input)
                print(reply)
            except anthropic.APIError as e:
                print(f"[API error] {e}")
            except Exception as e:
                print(f"[Unexpected error] {e}")

            print()

    finally:
        # --- Save conversation on every exit path -------------------------
        # Only skip saving if the user never sent a single message
        if agent.history:
            print(
                ("Saving conversation…",
                 "Ukládám konverzaci…")[0 if language == "en" else 1]
            )
            topic = _generate_topic_summary(client, agent.history, language)
            updated_conv = {
                "id":            conv_id,
                "timestamp":     conv_timestamp,
                "language":      language,
                "topic_summary": topic,
                "history":       agent.history,
            }
            # Replace the existing entry when resuming, otherwise append
            existing_idx = next(
                (i for i, c in enumerate(conversations) if c["id"] == conv_id),
                None,
            )
            if existing_idx is not None:
                conversations[existing_idx] = updated_conv
            else:
                conversations.append(updated_conv)
            _save_conversations(conversations)
            print(
                ("Saved as: ", "Uloženo jako: ")[0 if language == "en" else 1]
                + topic
            )

    # Farewell message — only for explicit exit commands, not Ctrl+C/Ctrl+D
    if explicit_exit:
        print("Goodbye!" if language == "en" else "Na shledanou!")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    main()

