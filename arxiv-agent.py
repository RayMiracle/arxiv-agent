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

import arxiv
import anthropic
from datetime import date
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
# 3. CLI interface
# ---------------------------------------------------------------------------

def main():
    """
    Run the interactive command-line research assistant.

    The agent and the conversation live only in this function — if you later
    build a web UI, you would create a ResearchAgent per user session there
    instead, without touching the class itself.
    """
    print("=" * 60)
    print("  ArXiv Research Assistant (powered by Claude)")
    print("=" * 60)

    # --- Language selection ---
    print("\nSelect language / Vyberte jazyk:")
    print("  1. English")
    print("  2. Czech (Čeština)")

    language = "en"  # default
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

    print()
    if language == "cs":
        print("Jazyk nastaven na češtinu.")
        print("Zeptejte se mě na výzkumné články. Napište 'exit', 'quit', 'konec' nebo 'ukončit' pro ukončení.")
    else:
        print("Ask me to find or explain research papers.")
        print("Type 'exit' or 'quit' to leave.")
    print()

    agent = ResearchAgent(language=language)

    while True:
        # Prompt the user for input
        try:
            user_input = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            # Handle Ctrl+D / Ctrl+C gracefully
            print("\nGoodbye!")
            break

        # Skip empty input
        if not user_input:
            continue

        # Check for exit commands (English always accepted; Czech added when language is Czech)
        exit_commands = {"exit", "quit"}
        if language == "cs":
            exit_commands |= {"konec", "ukončit"}
        if user_input.lower() in exit_commands:
            print("Goodbye!" if language == "en" else "Na shledanou!")
            break

        # Send the message to the agent and print the response
        print()  # blank line before the response
        try:
            reply = agent.ask(user_input)
            print(reply)
        except anthropic.APIError as e:
            print(f"[API error] {e}")
        except Exception as e:
            print(f"[Unexpected error] {e}")

        print()  # blank line after the response


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    main()
