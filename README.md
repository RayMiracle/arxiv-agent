# ArXiv Research Assistant

An interactive CLI tool that uses Claude to help you explore academic papers from ArXiv.

## Features

- Search ArXiv for papers on any topic
- Multi-turn conversation — ask follow-up questions about papers already found
- Smart routing: automatically detects whether a new search is needed or the question is a follow-up
- Powered by Claude (`claude-haiku-4-5-20251001`)

## Architecture

```
arxiv-agent.py
├── search_arxiv()   — Pure data function; queries ArXiv, no AI involved
├── ResearchAgent    — Core logic: Claude + search, no I/O
└── main()           — CLI interface only
```

This separation makes it easy to swap the CLI (`main()`) for a web UI without touching the agent logic.

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

```bash
python arxiv-agent.py
```

### Example session

```
============================================================
  ArXiv Research Assistant (powered by Claude)
============================================================
Ask me to find or explain research papers.
Type 'exit' or 'quit' to leave.

> Find papers about diffusion models
...
> Which of those papers is most beginner-friendly?
...
> exit
Goodbye!
```

### Commands

| Input | Action |
|---|---|
| Any topic or question | Search ArXiv and answer |
| Follow-up question | Answer using papers already in context |
| `exit` / `quit` | Exit the program |
| `Ctrl+C` / `Ctrl+D` | Exit gracefully |

## How It Works

1. On the first message, the agent always searches ArXiv.
2. For subsequent messages, a fast classifier call (`max_tokens=10`) asks Claude whether the message is a **new topic** (`SEARCH`) or a **follow-up** (`FOLLOWUP`).
3. If a new search is needed, the top 5 most relevant papers are fetched and prepended to the user message as context.
4. The full conversation history is sent to Claude on every turn, enabling coherent multi-turn dialogue.
