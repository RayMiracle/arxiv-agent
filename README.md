# ArXiv Research Assistant

An interactive CLI tool that uses Claude to help you explore academic papers from ArXiv.

## Features

- Search ArXiv for papers on any topic
- Multi-turn conversation — ask follow-up questions about papers already found
- Smart routing: Claude itself decides whether a new search is needed or the question is a follow-up
- Language selection at startup: **English** or **Czech (Čeština)**
  - Czech mode instructs Claude to respond in Czech, keeping technical terms in English with Czech explanations in brackets
- Powered by Claude (`claude-haiku-4-5-20251001`)

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

Select language / Vyberte jazyk:
  1. English
  2. Czech (Čeština)
Your choice (1/2): 1

Ask me to find or explain research papers.
Type 'exit' or 'quit' to leave.

> Find papers about diffusion models
...
> Which of those papers is most beginner-friendly?
...
> exit
Goodbye!
```

In Czech mode, accepted exit commands are: `exit`, `quit`, `konec`, `ukončit`.

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
