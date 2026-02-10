## Agentic Blog Tagger + Summarizer (Ollama • LangChain • LangGraph)

A small, **local-first** agent workflow powered by **Ollama (SmolLM 1.7B)** and orchestrated with **LangChain + LangGraph**.

### What it does
Given a blog **title** + **content**, the system runs a short agent flow:

**Planner → Reviewer → Finalizer**

It outputs:
- **Exactly 3 topical tags**
- **A single-sentence summary (≤ 25 words)**
- Printed as **strict, valid JSON**

### Why it’s interesting
- Runs fully **offline** (privacy-friendly; no hosted LLM required)
- Uses **structured prompting + JSON parsing + validation/repair** to guarantee format correctness
- Demonstrates **agentic orchestration** (multi-step reasoning + iterative refinement)

### Future potential
This project can evolve into a more general **stateful agent graph** for real workflows, for example:
- Supervisor routing + “issue-driven” loops (re-plan until quality criteria are met)
- Tool use (web/search, schema validation, linting, etc.)
- RAG integration for large document sets (auto-tagging and metadata generation at scale)
- Packaging as a lightweight service for **offline summarization/tagging** in internal knowledge bases
