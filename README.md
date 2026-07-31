# ArXiv Research Assistant

An interactive CLI tool that uses Claude to help you explore academic papers from ArXiv.

## Features

- Search ArXiv for papers on any topic
- Multi-turn conversation — ask follow-up questions about papers already found
- Smart routing: Claude classifies every message as **SEARCH**, **FOLLOWUP**, or **UNCLEAR**
  - `SEARCH` — fetches fresh ArXiv papers for a new topic
  - `FOLLOWUP` — answers using papers already in the conversation
  - `UNCLEAR` — skips the search and asks the user for clarification
  - On the first message (no prior papers), only `SEARCH` or `UNCLEAR` are offered
- Language selection at startup: **English** or **Czech (Čeština)**
  - Czech mode instructs Claude to respond in Czech, keeping technical terms in English with Czech explanations in brackets
- **Conversation persistence** — conversations are automatically saved to `conversations-history.json` on exit
  - Resume any previous session (with an AI-generated recap)
  - Delete saved conversations
  - Export any conversation to a Markdown file
  - Type `export` at any time during a chat to export the current session
  - Stores up to 12 conversations; oldest are dropped automatically
- Powered by Claude (`claude-haiku-4-5-20251001`)

## AI disclosure (EU AI Act, Article 50)

This is a conversational AI tool. A startup banner discloses that answers are
AI-generated (Claude) and may be incomplete or inaccurate, and recommends verifying
against the cited papers — see the `main()` function in `arxiv-agent.py`.

## What You Can Search For

ArXiv hosts over 2 million open-access academic papers across eight major fields:

| Field | Example topics |
|---|---|
| **Computer Science** | AI, machine learning, NLP, cryptography, computer vision |
| **Physics** | Astrophysics, quantum physics, nuclear physics, optics |
| **Mathematics** | Algebra, geometry, number theory, statistics |
| **Quantitative Biology** | Genomics, neuroscience, evolutionary biology |
| **Statistics** | Machine learning, applied statistics, methodology |
| **Electrical Engineering** | Signal processing, systems and control |
| **Quantitative Finance** | Econometrics, financial risk, portfolio theory |
| **Economics** | Microeconomics, macroeconomics, general economics |

All papers on ArXiv are **open-access** — no subscription required.

## Architecture

```
arxiv-agent.py
├── search_arxiv()        — Pure data function; queries ArXiv, no AI involved
├── _build_system_prompt()— Builds Claude's system prompt (language-aware, date-stamped)
├── ResearchAgent         — Core logic: Claude + search, no I/O
│   ├── ask()             — Main entry point; classify → (translate+search) → respond
│   ├── _classify_message()    — SEARCH / FOLLOWUP / UNCLEAR via Claude
│   └── _to_english_query()    — Translates/extracts ArXiv search term
├── Persistence layer     — conversations-history.json read/write, Markdown export
│   ├── _load_conversations()  
│   ├── _save_conversations()
│   ├── _generate_topic_summary()
│   ├── _generate_resume_summary()
│   ├── _export_to_markdown()
│   └── _show_startup_menu()
└── main()                — CLI interface only; persistence wired here
```

This separation makes it easy to swap the CLI (`main()`) for a web UI without touching the agent or persistence logic.

## Requirements

- Python 3.10+
- An [Anthropic API key](https://console.anthropic.com/)

## Installation

```bash
pip install anthropic arxiv python-dotenv
```

## Configuration

Create a `.env` file in the project root:

```
ANTHROPIC_API_KEY=your_api_key_here
```

## Usage

> **Note:** Make sure the virtual environment is activated first (`venv\Scripts\activate` on Windows), otherwise Python won't find the installed packages.

```bash
python arxiv-agent.py
```

### Example session

**First run (no saved conversations):**

```
============================================================
  ArXiv Research Assistant (powered by Claude)
============================================================

Select language / Vyberte jazyk:
  1. English
  2. Czech (Čeština)
Your choice (1/2): 1

Ask me to find or explain research papers.
Commands: 'export' (save to Markdown),  'exit' / 'quit' (quit)

> Find papers about diffusion models
...
> Which of those papers is most beginner-friendly?
...
> exit
Saving conversation…
Saved as: Diffusion models image generation
Goodbye!
```

**Subsequent runs (saved conversations exist):**

```
Select language / Vyberte jazyk:
  1. English
  2. Czech (Čeština)
Your choice (1/2): 1

Saved conversations:
  1. 2026-06-24 10:30 — Diffusion models image generation
  2. 2026-06-23 15:45 — Transformer attention mechanisms

Options:
  N — New conversation
  D — Delete a conversation
  E — Export to Markdown
  Or enter a number to resume

Your choice: 1

Generating conversation summary…
[2–4 sentence recap of the previous session]

> What about video diffusion?
...
```

In Czech mode, accepted exit commands are: `exit`, `quit`, `konec`, `ukončit`.

### Commands

| Input | Action |
|---|---|
| Any topic or question | Search ArXiv and answer |
| Follow-up question | Answer using papers already in context |
| `export` | Export current session to a Markdown file |
| `exit` / `quit` | Save and exit |
| `konec` / `ukončit` (Czech mode) | Save and exit |
| `Ctrl+C` / `Ctrl+D` | Save and exit (no farewell message) |

## How It Works

**Routing (every message)**
1. A fast classifier call (`max_tokens=10`) asks Claude to label the message **SEARCH**, **FOLLOWUP**, or **UNCLEAR**.
   - On the first message, only **SEARCH** or **UNCLEAR** are valid (nothing to follow up on yet).
2. If **SEARCH**: a second quick call extracts and translates the topic into a concise English ArXiv query, then the top 5 most relevant papers are fetched and injected as context.
3. If **FOLLOWUP**: the message is passed to Claude with the existing conversation history.
4. If **UNCLEAR**: the message is passed to Claude without a search so it can ask for clarification.
5. The full conversation history is sent to Claude on every turn, enabling coherent multi-turn dialogue.
6. The system prompt includes today's date so Claude can reason accurately about paper recency.

**Persistence (every session)**

1. Conversations are saved to `conversations-history.json` automatically on exit — whether via an exit command or Ctrl+C/D.
2. On startup, if saved conversations exist, a menu lets you resume, delete, or export them.
3. Resuming loads the full history and generates a short AI summary of what was previously discussed.
4. Exporting writes a clean Markdown file (ArXiv context blocks stripped) named `{topic}_{timestamp}-export.md`.
5. At most 12 conversations are kept; oldest are dropped when a 13th is saved.
