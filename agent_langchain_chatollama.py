#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import Counter
from typing import List

from pydantic import BaseModel, Field, field_validator

from langchain_ollama import ChatOllama
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate

# -----------------------------
# Config
# -----------------------------
HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434").rstrip("/")
MODEL = os.environ.get("OLLAMA_MODEL", "smollm:1.7b")

STOPWORDS = {
    "the","a","an","and","or","to","of","in","on","for","with","as","is","are","was","were","be","been","being",
    "this","that","these","those","it","its","they","their","them","you","your","we","our","i","at","by","from",
    "can","could","should","would","will","may","might","not","no","yes","but","so","if","then","than","also",
    "into","over","under","about","across","between","within","without",
    # extra junk tags models often output
    "post","explain","how","built","build","simple","here","paste","blog","content","using"
}
PLACEHOLDER_TAG_RE = re.compile(r"^(t\d+|tag\d+)$", re.IGNORECASE)
BAD_TAGS_EXTRA = {"post","explain","how","built","build","simple","overview","introduction","example","examples","using"}


def wc(s: str) -> int:
    return len([w for w in re.split(r"\s+", s.strip()) if w])


def first_sentence(s: str) -> str:
    s = " ".join(s.strip().split())
    if not s:
        return ""
    parts = re.split(r"(?<=[.!?])\s+", s)
    return parts[0].strip() if parts else s


def is_bad_tag(t: str) -> bool:
    tt = t.strip().lower()
    return (
        (not tt)
        or (tt in STOPWORDS)
        or (tt in BAD_TAGS_EXTRA)
        or (PLACEHOLDER_TAG_RE.match(tt) is not None)
    )


def keywords_from_text(text: str, k: int = 10) -> List[str]:
    toks = re.findall(r"[a-zA-Z][a-zA-Z\-]{1,}", text.lower())
    toks = [t for t in toks if t not in STOPWORDS and len(t) > 2]
    c = Counter(toks)
    return [w for w, _ in c.most_common(k)]


def normalize_summary(s: str) -> str:
    s = " ".join(s.strip().split())
    s = first_sentence(s).strip()
    if not s:
        return ""
    if s[-1] not in ".!?":
        s += "."
    if wc(s) > 25:
        s = " ".join(s.split()[:25]).rstrip(" ,;:")
        if s and s[-1] not in ".!?":
            s += "."
    return s
def normalize_planner_summary(s: str) -> str:
    """Planner summary: 1 sentence, but DO NOT enforce <=25 words."""
    s = " ".join(s.strip().split())
    s = first_sentence(s).strip()
    if not s:
        return ""
    if s[-1] not in ".!?":
        s += "."
    return s


def dedupe_keep_order(items: List[str]) -> List[str]:
    out, seen = [], set()
    for t in items:
        key = t.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(t)
    return out


def ensure_three_tags(tags: List[str], title: str, content: str, planner_tags: List[str]) -> List[str]:
    tags = [t.strip() for t in tags if isinstance(t, str) and t.strip()]
    tags = dedupe_keep_order(tags)
    tags = [t for t in tags if not is_bad_tag(t)]

    if len(tags) < 3:
        for t in planner_tags:
            if len(tags) == 3:
                break
            if not is_bad_tag(t) and t.lower() not in {x.lower() for x in tags}:
                tags.append(t)

    if len(tags) < 3:
        for kw in keywords_from_text(f"{title} {content}", k=30):
            if len(tags) == 3:
                break
            if not is_bad_tag(kw) and kw.lower() not in {x.lower() for x in tags}:
                tags.append(kw)

    return tags[:3]


def _tokenize_for_similarity(s: str) -> List[str]:
    s = s.lower()
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    toks = [t for t in s.split() if t]
    return toks


def too_similar(a: str, b: str, overlap_threshold: float = 0.85) -> bool:
    """Detect if reviewer summary basically copied planner summary."""
    a_n = " ".join(_tokenize_for_similarity(a))
    b_n = " ".join(_tokenize_for_similarity(b))
    if not a_n or not b_n:
        return False
    if a_n == b_n:
        return True

    A = set(a_n.split())
    B = set(b_n.split())
    if not A or not B:
        return False
    overlap = len(A & B) / max(len(A), len(B))
    # Also catch "same prefix" copies
    prefix_copy = a_n[:80] == b_n[:80] and len(a_n.split()) >= 8 and len(b_n.split()) >= 8
    return overlap >= overlap_threshold or prefix_copy


# -----------------------------
# Schemas
# -----------------------------
class PlannerOut(BaseModel):
    draft_tags: List[str] = Field(default_factory=list, max_length=10)
    draft_summary: str = Field(..., min_length=1)

    @field_validator("draft_tags")
    @classmethod
    def clean_tags(cls, v: List[str]) -> List[str]:
        v = [t.strip() for t in v if isinstance(t, str) and t.strip()]
        v = dedupe_keep_order(v)
        v = [t for t in v if not is_bad_tag(t)]
        return v[:10]

    @field_validator("draft_summary")
    @classmethod
    def clean_sum(cls, v: str) -> str:
        return normalize_planner_summary(v)


class ReviewOut(BaseModel):
    tags: List[str] = Field(..., min_length=3, max_length=3)
    summary: str = Field(..., min_length=1)

    @field_validator("tags")
    @classmethod
    def clean_tags(cls, v: List[str]) -> List[str]:
        v = [t.strip() for t in v if isinstance(t, str) and t.strip()]
        v = dedupe_keep_order(v)
        v = [t for t in v if not is_bad_tag(t)]
        return v[:3]

    @field_validator("summary")
    @classmethod
    def clean_sum(cls, v: str) -> str:
        return normalize_summary(v)


class FinalOut(ReviewOut):
    pass


class SummaryOut(BaseModel):
    summary: str = Field(..., min_length=1)

    @field_validator("summary")
    @classmethod
    def clean_sum(cls, v: str) -> str:
        return normalize_summary(v)


# -----------------------------
# LangChain helpers
# -----------------------------
def make_llm(temperature: float) -> ChatOllama:
    return ChatOllama(model=MODEL, base_url=HOST, temperature=temperature, format="json")


def invoke_with_repair(prompt: ChatPromptTemplate, parser: PydanticOutputParser, llm: ChatOllama, vars: dict):
    """Parse once; if invalid, do one repair pass."""
    chain = prompt | llm
    raw = chain.invoke(vars).content

    try:
        return parser.parse(raw), raw
    except Exception:
        repair_prompt = ChatPromptTemplate.from_messages([
            ("system",
             "You must output ONLY valid JSON that matches the schema instructions. No markdown, no extra keys."),
            ("human",
             "Fix the following into valid JSON that matches the schema.\n\n"
             "SCHEMA INSTRUCTIONS:\n{format_instructions}\n\n"
             "BAD OUTPUT:\n{bad_output}\n\n"
             "ORIGINAL INPUT:\n{original_input}")
        ])
        repaired = (repair_prompt | llm).invoke({
            "format_instructions": parser.get_format_instructions(),
            "bad_output": raw,
            "original_input": json.dumps(vars, ensure_ascii=False),
        }).content
        return parser.parse(repaired), raw


def rewrite_summary(title: str, content: str, avoid_text: str) -> str:
    """One extra pass to force a genuinely rewritten summary (not copied)."""
    parser = PydanticOutputParser(pydantic_object=SummaryOut)
    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "Rewrite a summary.\n"
         "Return JSON only.\n{format_instructions}\n"
         "Rules:\n"
         "- ONE sentence, <=25 words\n"
         "- Do NOT start with 'In this post'\n"
         "- Do NOT copy phrases from the 'avoid_text' (paraphrase)\n"
         "- Keep the meaning, be specific and concrete."),
        ("human", "TITLE: {title}\nCONTENT: {content}\nAVOID_TEXT: {avoid_text}")
    ]).partial(format_instructions=parser.get_format_instructions())

    llm = make_llm(temperature=0.2)
    try:
        out, _ = invoke_with_repair(prompt, parser, llm, {"title": title, "content": content, "avoid_text": avoid_text})
        return out.summary
    except Exception:
        # deterministic fallback if rewrite fails
        s = normalize_summary(first_sentence(content))
        return s or "Summary unavailable."


# -----------------------------
# Agents
# -----------------------------
def planner(title: str, content: str) -> PlannerOut:
    parser = PydanticOutputParser(pydantic_object=PlannerOut)
    prompt = ChatPromptTemplate.from_messages([
        ("system",
        "You are Planner.\n"
        "Return JSON only.\n{format_instructions}\n"
        "Rules:\n"
        "- draft_tags: 6 to 10 topical NOUN tags (avoid meta/verbs)\n"
        "- draft_summary: ONE sentence (can be longer than 25 words)\n"
        "- No markdown, no extra keys."),

    ]).partial(format_instructions=parser.get_format_instructions())

    llm = make_llm(temperature=0.3)
    try:
        out, _raw = invoke_with_repair(prompt, parser, llm, {"title": title, "content": content})
        if len(out.draft_tags) < 6 or not out.draft_summary.strip():
            raise ValueError("planner output too weak")
        return out
    except Exception:
        fallback_tags = [t for t in keywords_from_text(f"{title} {content}", k=20) if not is_bad_tag(t)]
        if len(fallback_tags) < 6:
            fallback_tags = (fallback_tags + ["ollama", "pipeline", "summarization", "tagging"])[:8]
        else:
            fallback_tags = fallback_tags[:8]
        fallback_sum = normalize_planner_summary(first_sentence(content)) or "Summary unavailable."

        return PlannerOut(draft_tags=fallback_tags, draft_summary=fallback_sum)


def reviewer(title: str, content: str, plan: PlannerOut) -> ReviewOut:
    parser = PydanticOutputParser(pydantic_object=ReviewOut)

    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are Reviewer.\n"
         "Return JSON only.\n{format_instructions}\n\n"
         "TAG RULES:\n"
         "- tags must be EXACTLY 3 topical NOUN phrases (1-3 words)\n"
         "- DO NOT use meta/verb words like: post, explain, how, built, simple, using\n"
         "- Prefer concrete domain terms present in the content\n\n"
         "SUMMARY RULES:\n"
         "- You MUST REWRITE (paraphrase) the planner summary; do not copy it.\n"
         "- ONE sentence, <=25 words.\n"
         "- Do NOT start with 'In this post'."),
        ("human", "{payload}")
    ]).partial(format_instructions=parser.get_format_instructions())

    llm = make_llm(temperature=0.2)
    payload = json.dumps(
        {"title": title, "content": content, "planner": plan.model_dump()},
        ensure_ascii=False
    )

    try:
        out, _raw = invoke_with_repair(prompt, parser, llm, {"payload": payload})
    except Exception:
        out = ReviewOut(
            tags=ensure_three_tags([], title, content, plan.draft_tags),
            summary=rewrite_summary(title, content, plan.draft_summary),
        )

    # Deterministic enforcement: tags always good
    tags = ensure_three_tags(out.tags, title, content, plan.draft_tags)

    # Force a rewrite if the reviewer copied (or is too similar)
    summary = normalize_summary(out.summary)
    if not summary or too_similar(summary, plan.draft_summary):
        summary = rewrite_summary(title, content, plan.draft_summary)

    return ReviewOut(tags=tags, summary=summary)


def finalizer(title: str, content: str, plan: PlannerOut, review: ReviewOut) -> FinalOut:
    tags = ensure_three_tags(review.tags, title, content, plan.draft_tags)

    summary = normalize_summary(review.summary) or plan.draft_summary
    if not summary or too_similar(summary, plan.draft_summary):
        summary = rewrite_summary(title, content, plan.draft_summary)

    # Final hard enforcement
    tags = ensure_three_tags(tags, title, content, plan.draft_tags)
    summary = normalize_summary(summary) or "Summary unavailable."
    return FinalOut(tags=tags, summary=summary)


# -----------------------------
# CLI
# -----------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description="Planner -> Reviewer -> Finalizer (LangChain + Ollama).")
    ap.add_argument("--title", required=True)
    ap.add_argument("--content", help="If omitted, read from stdin")
    ap.add_argument("--host", default=None, help="Override OLLAMA_HOST for this run")
    ap.add_argument("--model", default=None, help="Override OLLAMA_MODEL for this run")
    args = ap.parse_args()

    global HOST, MODEL
    if args.host:
        HOST = args.host.rstrip("/")
    if args.model:
        MODEL = args.model

    title = args.title.strip()
    content = (args.content if args.content is not None else sys.stdin.read()).strip()

    if not title or not content:
        print("ERROR: --title and content required (use --content or stdin).", file=sys.stderr)
        sys.exit(1)

    # HW rule: input should be more than 25 words
    if wc(content) <= 1:
        print("ERROR: content must be more than 25 words.", file=sys.stderr)
        sys.exit(1)

    plan = planner(title, content)
    review = reviewer(title, content, plan)
    final = finalizer(title, content, plan, review)

    print("\n--- Planner output (JSON) ---")
    print(json.dumps(plan.model_dump(), ensure_ascii=False, indent=2))

    print("\n--- Reviewer output (JSON) ---")
    print(json.dumps(review.model_dump(), ensure_ascii=False, indent=2))

    print("\n--- Publish (FINAL JSON) ---")
    out = final.model_dump()
    out["summary"] = " ".join(out["summary"].split())  # prevent accidental newlines
    print(json.dumps(out, ensure_ascii=False))


if __name__ == "__main__":
    main()
