#!/usr/bin/env python3
"""
agents_demo.py — Agentic AI (Part 2) for DATA-236 HW1

Goal (per assignment):
- Input: blog title + content
- Output: exactly 3 topical tags + a one-sentence summary (<=25 words),
  produced via Planner -> Reviewer -> Finalizer, printed as valid JSON.  (No domain hardcoding)

Key reliability improvements (vs your original):
1) Reviewer prompt forbids echoing the input and forbids extra keys.
2) Finalizer enforces summary is NON-empty and one sentence.
3) Correction-call results do NOT overwrite good tags/summary unless valid.
4) Planner produces 6–10 candidate tags (so we don't fall back to junk keywords).
5) Fallback keyword extraction uses title+content and filters generic filler words.
6) extract_json is robust to JSON-in-a-string and extra surrounding text.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import urllib.request
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434").rstrip("/")
MODEL = os.environ.get("OLLAMA_MODEL", "smollm:1.7b")


# ----------------------------- Ollama call -----------------------------
def ollama_chat(messages: List[Dict[str, str]], temperature: float = 0.2) -> str:
    """
    Minimal Ollama /api/chat call. Returns assistant message content (string).
    """
    url = f"{HOST}/api/chat"
    payload = {
        "model": MODEL,
        "messages": messages,
        "stream": False,
        # Ollama supports "format": "json" to encourage strict JSON outputs
        "format": "json",
        "options": {"temperature": temperature},
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    with urllib.request.urlopen(req, timeout=120) as resp:
        out = json.loads(resp.read().decode("utf-8"))

    return out["message"]["content"]


# ----------------------------- JSON helpers -----------------------------
def extract_json(text: str) -> Any:
    """
    Robust JSON extractor:
    - Handles clean JSON
    - Handles JSON-in-a-quoted-string (double-encoded)
    - Handles extra text by extracting the first {...} block
    """
    s = text.strip()

    # Try up to 2 decode layers (covers JSON-in-a-string)
    for _ in range(2):
        try:
            obj = json.loads(s)
            if isinstance(obj, str):
                s = obj.strip()
                continue
            return obj
        except Exception:
            break

    # Fallback: first {...} block
    start = s.find("{")
    end = s.rfind("}")
    if start != -1 and end != -1 and end > start:
        return json.loads(s[start : end + 1])

    raise ValueError(f"Model did not return valid JSON. Raw: {text[:2000]}")


def wc(s: str) -> int:
    return len([w for w in re.split(r"\s+", s.strip()) if w])


def first_sentence(s: str) -> str:
    """
    Ensures output is one sentence.
    We take the first sentence-like chunk if multiple exist.
    """
    s = " ".join(s.strip().split())
    if not s:
        return ""
    # Split on common sentence terminators. Keep first chunk.
    parts = re.split(r"(?<=[.!?])\s+", s)
    return parts[0].strip() if parts else s


def split_tag_blob(s: str) -> List[str]:
    # Splits "a, b | c\n" into ["a","b","c"]
    parts = re.split(r"[,|;/\n]+", s)
    return [p.strip() for p in parts if p.strip()]


def normalize_tags(tags_any: Any) -> List[str]:
    """
    Accept list or string. Normalize, trim, de-dupe case-insensitively, keep order.
    Also handles a single string containing comma-separated tags.
    """
    items: List[str] = []

    if isinstance(tags_any, str):
        items = split_tag_blob(tags_any)
    elif isinstance(tags_any, list):
        for t in tags_any:
            if not isinstance(t, str):
                continue
            t = t.strip()
            if not t:
                continue
            # If the list contains one big comma-separated string, split it.
            if len(tags_any) == 1 and re.search(r"[,|;/\n]", t):
                items.extend(split_tag_blob(t))
            else:
                items.append(t)

    out: List[str] = []
    seen: set[str] = set()
    for t in items:
        key = t.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(t)

    return out


# Generic stopwords + common filler words that cause junk tags.
STOPWORDS = {
    "the", "a", "an", "and", "or", "to", "of", "in", "on", "for", "with", "as",
    "is", "are", "was", "were", "be", "been", "being", "this", "that", "these",
    "those", "it", "its", "they", "their", "them", "you", "your", "we", "our", "i",
    "at", "by", "from", "can", "could", "should", "would", "will", "may", "might",
    "not", "no", "yes", "but", "so", "if", "then", "than", "also", "into", "over",
    "under", "about", "across", "between", "within", "without",
    # filler / low-information words seen in test content
    "some", "test", "content", "writing", "words", "understand", "actually", "more",
    "so", "we", "reviewer", "planner", "title"
}

PLACEHOLDER_TAG_RE = re.compile(r"^(t\d+|tag\d+)$", re.IGNORECASE)

def is_bad_tag(t: str) -> bool:
    tt = t.strip().lower()
    return (
        not tt
        or tt in STOPWORDS
        or PLACEHOLDER_TAG_RE.match(tt) is not None
    )


def keywords_from_text(text: str, k: int = 10) -> List[str]:
    """
    Extract generic topical keywords (no domain hardcoding):
    - alphabetic tokens, allow hyphen
    - remove stopwords
    - prefer longer tokens
    """
    toks = re.findall(r"[a-zA-Z][a-zA-Z\-]{1,}", text.lower())
    toks = [t for t in toks if t not in STOPWORDS and len(t) > 2]
    c = Counter(toks)
    return [w for w, _ in c.most_common(k)]


# ----------------------------- Agents -----------------------------
def planner(title: str, content: str) -> Dict[str, Any]:
    system = (
        "You are Planner.\n"
        "Propose candidate topical tags and a draft summary.\n"
        'Return ONLY JSON with EXACT keys: {"draft_tags":[...], "draft_summary":"..."}\n'
        "- draft_tags: array of 6 to 10 short topical strings.\n"
        "- draft_summary: ONE sentence.\n"
        "No markdown. No extra text. No extra keys."
    )
    user = f"TITLE: {title}\nCONTENT: {content}"
    raw = ollama_chat(
        [{"role": "system", "content": system}, {"role": "user", "content": user}],
        temperature=0.3,
    )
    obj = extract_json(raw)
    if not isinstance(obj, dict):
        obj = {}

    # ✅ CHANGE: schema validation + deterministic fallback
    draft_tags = normalize_tags(obj.get("draft_tags"))
    draft_summary = obj.get("draft_summary") if isinstance(obj.get("draft_summary"), str) else ""

    if len(draft_tags) < 3 or not draft_summary.strip():
        # fallback: purely deterministic, still generic
        fallback_tags = keywords_from_text(f"{title} {content}", k=12)[:8]
        fallback_summary = first_sentence(" ".join(content.strip().split()))
        if fallback_summary and fallback_summary[-1] not in ".!?":
            fallback_summary += "."
        obj = {"draft_tags": fallback_tags, "draft_summary": fallback_summary}

    return obj



def reviewer(title: str, content: str, plan: Dict[str, Any]) -> Dict[str, Any]:
    """
    Reviewer: improve Planner output.
    Change: prevent echoing the input object; allow ONLY tags+summary keys.
    """
    system = (
        "You are Reviewer.\n"
        "Read the title/content and Planner output, then refine it.\n\n"
        "CRITICAL RULES:\n"
        "- Do NOT repeat/echo the input object.\n"
        "- Do NOT include title/content/planner fields.\n"
        "- Return ONLY JSON with EXACT keys: {\"tags\":[\"t1\",\"t2\",\"t3\"], \"summary\":\"...\"}\n"
        "- tags must be EXACTLY 3 topical strings.\n"
        "- you must always review and never return empty tags/summary.\n"
        "- summary must be ONE sentence, NON-EMPTY, <=25 words.\n"
        "No markdown. No extra text. No extra keys."
    )

    # Use a compact JSON input, but reviewer is explicitly forbidden from echoing it.
    user_obj = {"title": title, "content": content, "planner": plan}
    raw = ollama_chat(
        [{"role": "system", "content": system}, {"role": "user", "content": json.dumps(user_obj)}],
        temperature=0.2,
    )
    obj = extract_json(raw)
    if not isinstance(obj, dict):
        return {}
    if not (isinstance(obj.get("tags"), list) and isinstance(obj.get("summary"), str)):
        return {}
    return obj


# ----------------------------- Finalizer -----------------------------
def finalizer(title: str, content: str, plan: Dict[str, Any], review: Dict[str, Any]) -> Dict[str, Any]:
    """
    Deterministic validation + minimal LLM repair:
    1) Try to use Reviewer tags/summary (or tolerate misnamed keys)
    2) Fill tags from Planner then keywords
    3) Ensure summary non-empty; fallback to Planner summary if needed
    4) If still invalid -> ONE correction call
    5) Never overwrite good tags with empty correction output
    """

    def ensure_three_tags(tags: List[str]) -> List[str]:
        """Fill/trim to exactly 3 tags using Planner -> keywords fallback."""
        tags = [t.strip() for t in tags if isinstance(t, str) and t.strip()]
        # De-dupe again defensively
        deduped: List[str] = []
        seen: set[str] = set()
        for t in tags:
            key = t.lower()
            if key in seen:
                continue
            seen.add(key)
            deduped.append(t)
        tags = deduped

        tags = [t for t in tags if isinstance(t, str) and not is_bad_tag(t)]

        if len(tags) < 3:
            for t in normalize_tags(plan.get("draft_tags")):
                if len(tags) == 3:
                    break
                if t.lower() not in {x.lower() for x in tags}:
                    tags.append(t)

        if len(tags) < 3:
            # Use title+content for better topicality; still generic.
            for kw in keywords_from_text(f"{title} {content}", k=25):
                if len(tags) == 3:
                    break
                if is_bad_tag(kw):
                    continue
                if kw.lower() not in {x.lower() for x in tags}:
                    tags.append(kw)

        return tags[:3]

    def normalize_summary(s: str) -> str:
        """Enforce one sentence + <=25 words + non-empty."""
        s = " ".join(s.strip().split())
        s = first_sentence(s)
        s = s.strip()
        if not s:
            return ""

        # Ensure it ends like a sentence (optional but makes output cleaner)
        if s[-1] not in ".!?":
            s += "."

        # Enforce <=25 words
        if wc(s) > 25:
            words = [w for w in s.split() if w]
            s = " ".join(words[:25])
            s = s.rstrip(" ,;:")

        # Ensure it ends like a sentence
        if s and s[-1] not in ".!?":
            s += "."

        return s

    # -------- Step 1: pull from reviewer (tolerate occasional wrong keys) --------
    # If reviewer echoed input object, it likely has no tags/summary keys.
    tags = normalize_tags(review.get("tags") or review.get("draft_tags"))
    summary = review.get("summary") if isinstance(review.get("summary"), str) else ""

    # -------- Step 2: summary fallback to planner if missing/empty --------
    if not summary.strip():
        ps = plan.get("draft_summary")
        if isinstance(ps, str) and ps.strip():
            summary = ps.strip()

    # -------- Step 3: enforce tags = 3 deterministically --------
    tags = ensure_three_tags(tags)

    # -------- Step 4: normalize summary deterministically --------
    summary = normalize_summary(summary)

    reviewer_ok = isinstance(review.get("tags"), list) and isinstance(review.get("summary"), str) and review["summary"].strip()

    # -------- Step 5: if still invalid, ONE correction call --------
    need_fix = (
    not reviewer_ok 
    or len(tags) != 3
    or any(not isinstance(t, str) or is_bad_tag(t) for t in tags)
    or not summary.strip()
    or wc(summary) > 25
    )

    if need_fix:
        system = (
            "Fix the output.\n"
            "Return ONLY JSON with EXACT keys:\n"
            '{"tags":["t1","t2","t3"],"summary":"one sentence <=25 words"}\n'
            "Rules:\n"
            "tags cannot be placeholders like t1/t2/tag1\n"
            "- tags must be EXACTLY 3 topical strings\n"
            "- summary must be ONE sentence, NON-EMPTY, <=25 words\n"
            "- No extra keys, no markdown, no extra text"
        )
        user_obj = {"title": title, "content": content, "planner": plan, "reviewer": review}
        raw = ollama_chat(
            [{"role": "system", "content": system}, {"role": "user", "content": json.dumps(user_obj)}],
            temperature=0.1,
        )
        obj = extract_json(raw)
        if isinstance(obj, dict):
            # IMPORTANT CHANGE: do NOT overwrite good tags unless correction tags are usable
            cand_tags = normalize_tags(obj.get("tags"))
            if len(cand_tags) >= 3:
                tags = cand_tags[:3]
            else:
                tags = ensure_three_tags(tags)

            cand_sum = obj.get("summary")
            if isinstance(cand_sum, str) and cand_sum.strip():
                summary = normalize_summary(cand_sum)

            # If correction summary still empty, last fallback to planner again
            if not summary.strip():
                ps = plan.get("draft_summary")
                if isinstance(ps, str) and ps.strip():
                    summary = normalize_summary(ps)

    # -------- Final hard validation (must satisfy assignment constraints) --------
    tags = ensure_three_tags(tags)
    summary = normalize_summary(summary)

    if len(tags) != 3:
        raise RuntimeError("Finalizer could not produce exactly 3 tags.")
    if not summary.strip() or wc(summary) > 25:
        raise RuntimeError("Finalizer could not produce a non-empty <=25-word summary.")

    return {"tags": tags, "summary": summary}


# ----------------------------- CLI -----------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description="Planner -> Reviewer -> Finalizer blog tagger/summarizer (Ollama).")
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

    plan = planner(title, content)
    review = reviewer(title, content, plan)
    final = finalizer(title, content, plan, review)

    # Transcript
    print("\n--- Planner output (JSON) ---")
    print(json.dumps(plan, ensure_ascii=False, indent=2))
    print("\n--- Reviewer output (JSON) ---")
    print(json.dumps(review, ensure_ascii=False, indent=2))

    # Final publish JSON (one line, valid JSON)
    print("\n--- Publish (FINAL JSON) ---")
    print(json.dumps(final, ensure_ascii=False))


if __name__ == "__main__":
    main()

