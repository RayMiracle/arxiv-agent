"""
ArXiv Research Assistant — CLI
================================
An interactive CLI tool that uses Claude to help you explore academic papers.

Architecture:
  - core.py    : search_arxiv, ResearchAgent, persistence layer (shared with
                 the Streamlit web UI in app_streamlit.py)
  - main()     : CLI interface only

This separation means the web UI can reuse all agent/persistence logic
without touching this file.
"""

import anthropic
from datetime import datetime
from pathlib import Path

from core import (
    ResearchAgent,
    load_conversations,
    save_conversations,
    generate_topic_summary,
    generate_resume_summary,
    export_to_markdown,
)


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
                save_conversations(conversations)
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
                conv["topic_summary"] = generate_topic_summary(
                    client, conv["history"], conv["language"]
                )
            filename = export_to_markdown(conv, folder)
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
    conversations = load_conversations()
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
        summary = generate_resume_summary(client, agent.history, language)
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
                topic = generate_topic_summary(client, agent.history, language)
                temp_conv = {
                    "id":            conv_id,
                    "timestamp":     conv_timestamp,
                    "language":      language,
                    "topic_summary": topic,
                    "history":       agent.history,
                }
                filename = export_to_markdown(temp_conv, script_folder)
                label = ("Exported to: ", "Exportováno do: ")[0 if language == "en" else 1]
                print(f"{label}{filename}")
                print()
                continue

            # Normal chat turn
            print()
            try:
                reply, _papers = agent.ask(user_input)
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
            topic = generate_topic_summary(client, agent.history, language)
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
            save_conversations(conversations)
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

