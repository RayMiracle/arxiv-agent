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

import os
import arxiv
import anthropic
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

# System prompt that tells Claude how to behave
_SYSTEM_PROMPT = """\
You are a helpful research assistant that specialises in academic papers from ArXiv.

When the user asks about a topic, you will be given relevant paper summaries as
context.  Use those papers to give accurate, well-organised answers.

Guidelines:
- Cite papers by title when you reference them.
- If the context does not contain enough information to answer, say so clearly.
- Keep explanations accessible — avoid unnecessary jargon.
- For follow-up questions, use the papers already in the conversation.
"""


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

    def __init__(self):
        # Anthropic client reads ANTHROPIC_API_KEY from the environment
        self._client = anthropic.Anthropic()

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
        1. Decide if we need a fresh ArXiv search.
        2. If yes, search and prepend the results to the user message.
        3. Append the (possibly enriched) message to history.
        4. Call Claude with the full history.
        5. Append Claude's reply to history.
        6. Return Claude's reply.
        """
        # Step 1 — should we search?
        if self._needs_new_search(user_message):
            # Extract the core topic from the message and search
            papers = search_arxiv(user_message, max_results=5)

            if papers:
                # Step 2 — format papers as a readable context block
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
            # Step 2 (skip) — just use the message as-is
            enriched_message = user_message

        # Step 3 — add user turn to history
        self.history.append({"role": "user", "content": enriched_message})

        # Step 4 — call Claude
        response = self._client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=1024,
            system=_SYSTEM_PROMPT,
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

    def _needs_new_search(self, message: str) -> bool:
        """
        Ask Claude whether the user's message requires a fresh ArXiv search.

        Claude replies with exactly one word — 'SEARCH' or 'FOLLOWUP' — so
        we use max_tokens=10 to keep this call fast and cheap.

        If no papers are in context yet we skip the call entirely and always
        search (there is nothing to follow up on).
        """
        if not self._has_paper_context:
            # No papers loaded yet — first message must always be a search.
            return True

        # Build a minimal prompt that gives Claude just enough context.
        classification_prompt = (
            "You are a routing assistant. "
            "The user is chatting with a research tool that can search ArXiv for papers.\n"
            "Papers from a previous search are already available in the conversation.\n\n"
            f"User message: \"{message}\"\n\n"
            "Does this message ask about a NEW topic that needs a fresh ArXiv search, "
            "or is it a follow-up question about the papers already found?\n"
            "Reply with exactly one word: SEARCH or FOLLOWUP."
        )

        response = self._client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=10,          # we only need one word
            messages=[{"role": "user", "content": classification_prompt}],
        )

        verdict = response.content[0].text.strip().upper()
        return verdict == "SEARCH"

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
    print("Ask me to find or explain research papers.")
    print("Type 'exit' or 'quit' to leave.\n")

    agent = ResearchAgent()

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

        # Check for exit commands
        if user_input.lower() in {"exit", "quit"}:
            print("Goodbye!")
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
