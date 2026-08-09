# ArXiv Research Assistant

An interactive tool that uses Claude to help you explore academic papers from ArXiv,
available as both a CLI and a Streamlit web UI.

## Why this instead of plain ChatGPT/Claude chat?

A general chat model without search answers **from its training data** — it has no
access to current ArXiv papers and can hallucinate citations or claims, especially
for recent or narrow topics.

This agent instead:

- **Actually searches ArXiv** on every new question (`search_arxiv()`) and fetches the
  5 most relevant, real, current papers
- Feeds their abstracts into Claude as context, so answers are grounded in **real,
  verifiable sources** — with titles, authors, dates, and links — instead of the
  model's memory
- Cites specific paper titles in its answers, so you can immediately look up the
  original and verify the claim

**Concrete benefits for a researcher:**

1. **Currency** — finds a paper uploaded last week; a model without search may not
   know it exists at all
2. **Verifiability** — every claim is backed by a link to a specific paper (also shown
   as cards with authors/date/link in the UI), instead of an unsourced general answer
3. **Fast orientation in a new field** — instead of manually searching ArXiv and
   reading a dozen abstracts, get a synthesis in seconds, with follow-up questions
   ("which of these is best for a beginner?")
4. **Continuity tied to concrete sources** — FOLLOWUP mode keeps the found papers in
   context, so follow-up questions relate to specific articles, not the model's
   general knowledge
5. **Cost transparency** — see exactly what each query cost (see LLM usage tracking
   below)

In short: the difference between *"ask the model what it remembers"* and *"ask the
model what actually exists on ArXiv right now, with evidence"* — for research work,
that difference is what makes an answer trustworthy.

## Scope & limitations

**This agent works from paper abstracts only — it never reads full paper text.**
`search_arxiv()` fetches only ArXiv's search-result metadata (title, authors,
publication date, and abstract), and the abstract is further truncated to 400
characters before being sent to Claude as context. It does not download or parse
PDFs, and there is no full-text retrieval, chunking, or RAG over paper content.

**What this means in practice:**

- You can ask for an overview, the paper's stated contribution, or how it compares to
  other found papers — all answerable from the abstract.
- You **cannot** ask for specific examples, numeric results, method details, or
  quotes from inside the paper — Claude does not have that text and will (correctly)
  say so rather than guess.
- If you need that level of detail, copy the relevant passage from the paper
  yourself and paste it into the chat — Claude can then discuss that text directly.

This is a deliberate scope choice for this version, not a bug: full-text retrieval
would add real latency, cost (a full paper can be 10–50k+ tokens), and complexity
that a lightweight abstract-search assistant doesn't need. See "Possible future
extensions" below if this scope ever needs to grow.

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
- **LLM usage tracking** — every Claude API call is logged to `usage-log.jsonl` with token counts and estimated USD cost
  - The Streamlit sidebar shows the current session's running cost and an all-time total (by call type)
- Powered by Claude (`claude-haiku-4-5-20251001`)

## AI disclosure (EU AI Act, Article 50)

This is a conversational AI tool. Answers are AI-generated (Claude) and may be
incomplete or inaccurate — verify against the cited papers before relying on them.
- CLI: a startup banner in `main()` (`arxiv-agent.py`) discloses this.
- Web UI: an always-visible `st.info` banner at the top of the page (`app_streamlit.py`) discloses this.

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
core.py                   — Shared logic, used by both the CLI and the web UI
├── search_arxiv()        — Pure data function; queries ArXiv, no AI involved
├── _build_system_prompt()— Builds Claude's system prompt (language-aware, date-stamped)
├── ResearchAgent         — Core logic: Claude + search, no I/O
│   ├── ask()             — Main entry point; classify → (translate+search) → respond
│   │                        Returns (reply, papers) — papers is set on SEARCH turns
│   ├── _classify_message()    — SEARCH / FOLLOWUP / UNCLEAR via Claude
│   └── _to_english_query()    — Translates/extracts ArXiv search term
└── Persistence layer     — conversations-history.json read/write, Markdown export
    ├── load_conversations()
    ├── save_conversations()
    ├── generate_topic_summary()
    ├── generate_resume_summary()
    ├── export_to_markdown()          — writes a .md file to disk (used by the CLI)
    └── render_conversation_markdown()— returns .md content as a string (used by the web UI)

arxiv-agent.py             — CLI interface only; imports core.py
└── main(), _show_startup_menu()

app_streamlit.py           — Streamlit web UI; imports core.py
```

Both interfaces share `conversations-history.json` — a conversation started in the
CLI can be resumed in the web UI, and vice versa.

## Requirements

- Python 3.10+
- An [Anthropic API key](https://console.anthropic.com/)

## Installation

```bash
pip install -r requirements.txt
```

## Configuration

Create a `.env` file in the project root:

```
ANTHROPIC_API_KEY=your_api_key_here
```

## Usage

> **Note:** Make sure the virtual environment is activated first (`venv\Scripts\activate` on Windows), otherwise Python won't find the installed packages.

### CLI

```bash
python arxiv-agent.py
```

### Web UI

```bash
streamlit run app_streamlit.py
```

Opens in your browser (default `http://localhost:8501`). Features mirror the CLI:
language selection, chat with SEARCH/FOLLOWUP/UNCLEAR routing, fetched papers shown
as expandable cards, a sidebar to resume/delete/export saved conversations, and a
button to export the current session to Markdown.

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

`conversations-history.json`, any `*export*` Markdown file, and `usage-log.jsonl` (LLM usage log — see Features above) are local, per-user data and are excluded from version control via `.gitignore`.

## Possible future extensions

Ideas for growing beyond the current abstract-only scope (see "Scope & limitations"
above), roughly in order of how self-contained they are:

- **Full-text reading of a single paper** — download the PDF from `paper["url"]`,
  extract text (e.g. `pypdf` or `pdfplumber`), and send the full text (or the
  relevant section) as context instead of just the 400-character abstract snippet.
  Straightforward as a first step for one paper at a time; the main cost is token
  usage on longer papers.
- **RAG over full paper text** — for longer papers or multi-paper questions, chunk
  the extracted text, embed it, and retrieve only the passages relevant to the
  question instead of sending the whole document. Needed once full-text sending
  becomes too expensive or the paper is too long to fit in context.
- **Citing specific sections/pages** — once full text is available, have Claude cite
  the section or page a claim comes from, not just the paper title.
- **Multi-paper comparison at full-text depth** — read several full papers in
  parallel and compare specific findings, not just abstracts.
- **Structured extraction** — pull out things like datasets used, reported metrics,
  or code/data availability into a structured table across multiple papers.

None of this is implemented — this list exists to record the scope trade-off made in
this version and give a starting point if the project is picked up again.
