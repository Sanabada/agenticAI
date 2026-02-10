#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import urllib.request
from collections import Counter
from typing import Any, Dict, List

HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434").rstrip("/")
MODEL = os.environ.get("OLLAMA_MODEL", "smollm:1.7b")


#  Ollama call
def ollama_chat(messages: List[Dict[str, str]], temperature: float = 0.2) -> str:
    url = f"{HOST}/api/chat"
    payload = {
        "model": MODEL,
        "messages": messages,
        "stream": False,
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


# JSON helpers
def extract_json(text: str) -> Any:
    text = text.strip()

    # direct parse
    try:
        return json.loads(text)
    except Exception:
        pass

    m = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if m:
        return json.loads(m.group(0))

    raise ValueError("Model did not return valid JSON.")


def wc(s: str) -> int:
    return len([w for w in re.split(r"\s+", s.strip()) if w])


def split_tag_blob(s: str) -> List[str]:
    # Splits "a, b | c\n" into ["a","b","c"]
    parts = re.split(r"[,|;/\n]+", s)
    return [p.strip() for p in parts if p.strip()]


def normalize_tags(tags_any: Any) -> List[str]:
    # Accepts list or string; also handle list of 1 item containing comma-separated tags.
    items: List[str] = []

    if isinstance(tags_any, str):
        items = split_tag_blob(tags_any)
    elif isinstance(tags_any, list):
        for t in tags_any:
            if isinstance(t, str):
                if re.search(r"[,|;/\n]", t) and len(tags_any) == 1:
                    items.extend(split_tag_blob(t))
                else:
                    items.append(t.strip())

    # de-dupe, keep order
    out: List[str] = []
    seen: set[str] = set()
    for t in items:
        if not t:
            continue
        key = t.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(t)

    return out


STOPWORDS = {"the","a","an","and","or","to","of","in","on","for","with","as","is","are","was","were","be","been","being",
             "this","that","these","those","it","its","they","their","them","you","your","we","our","i","at","by","from","can",
             "could","should","would","will","may","might","not","no","yes","but","so","if","then","than","also","into","over","under",
             "about","across","between","within","without","helps","help","using","use","used",}

def keywords_from_content(content: str, k: int = 6) -> List[str]:
    toks = re.findall(r"[a-zA-Z][a-zA-Z\-]{1,}", content.lower())
    toks = [t for t in toks if t not in STOPWORDS and len(t) > 2]
    c = Counter(toks)
    return [w for w, _ in c.most_common(k)]


# ----------------------------- Agents -----------------------------
def planner(title: str, content: str) -> Dict[str, Any]:
    system = (
        "You are Planner. Propose candidate topical tags and a draft one-sentence summary.\n"
        "Return ONLY JSON with keys: draft_tags (array of 6-10 strings), draft_summary (string).\n"
        "No markdown, no extra text."
    )
    user = f"TITLE: {title}\nCONTENT: {content}"
    raw = ollama_chat(
        [{"role": "system", "content": system}, {"role": "user", "content": user}],
        temperature=0.3,
    )
    return extract_json(raw)


def reviewer(title: str, content: str, plan: Dict[str, Any]) -> Dict[str, Any]:
    """system = (
        "You are Reviewer. Improve Planner output.\n"
        "Return ONLY JSON with keys: tags (array of EXACTLY 3 strings), "
        "summary (ONE sentence, <=25 words).\n"
        "No markdown, no extra text."
    )"""
    system = (
    "You are Reviewer. Read the title/content and Planner output, then improve it.\n"
    "IMPORTANT: Do NOT return the input object. Do NOT include title/content/planner fields.\n"
    'Return ONLY JSON with EXACT keys: {"tags":[...], "summary":"..."}.\n'
    "tags must be EXACTLY 3 topical strings.\n"
    "summary must be ONE sentence, NON-EMPTY, <=25 words.\n"
    "No extra keys, no markdown, no extra text."
    )
    user_obj = {"title": title, "content": content, "planner": plan}
    raw = ollama_chat(
        [{"role": "system", "content": system}, {"role": "user", "content": json.dumps(user_obj)}],
        temperature=0.2,
    )
    return extract_json(raw)


def finalizer(title: str, content: str, plan: Dict[str, Any], review: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate + fix deterministically. If still not valid, do 1 correction call to LLM.
    """

    def build_final(tags: List[str], summary: str) -> Dict[str, Any]:
        nonlocal title, content

        # enforce <=25 words
        if wc(summary) > 25:
            # try to shorten using LLM once
            short_system = (
                "Shorten the summary to ONE sentence with <=25 words.\n"
                'Return ONLY JSON: {"summary": "..."}'
            )
            raw = ollama_chat(
                [
                    {"role": "system", "content": short_system},
                    {"role": "user", "content": json.dumps({"summary": summary, "title": title, "content": content})},
                ],
                temperature=0.1,
            )
            try:
                s_obj = extract_json(raw)
                if isinstance(s_obj, dict) and isinstance(s_obj.get("summary"), str):
                    summary2 = s_obj["summary"].strip()
                    if summary2:
                        summary = summary2
            except Exception:
                pass

        # final safety: hard truncate to 25 words if still long
        if wc(summary) >= 25:
            words = [w for w in summary.split() if w]
            summary = " ".join(words[:25]).rstrip(" .") + "."

        summary = " ".join(summary.strip().split())
        return {"tags": tags, "summary": summary}

    # 1) Reviewer tags/summary
    tags = normalize_tags(review.get("tags"))
    summary = review.get("summary") if isinstance(review.get("summary"), str) else ""
    if not summary.strip():
        # fallback to Planner draft_summary if reviewer omitted/misnamed it
        ps = plan.get("draft_summary")
        if isinstance(ps, str) and ps.strip():
            summary = ps.strip()

    # 2) If tags not 3, supplement from planner draft tags
    if len(tags) != 3:
        draft_tags = normalize_tags(plan.get("draft_tags"))
        for t in draft_tags:
            if len(tags) == 3:
                break
            if t.lower() not in {x.lower() for x in tags}:
                tags.append(t)

    # 3) If still not 3, supplement from content keywords (generic, not domain-hardcoded)
    if len(tags) != 3:
        for kw in keywords_from_content(content, k=10):
            if len(tags) == 3:
                break
            if kw.lower() not in {x.lower() for x in tags}:
                tags.append(kw)

    # 4) Trim to exactly 3
    tags = tags[:3]

    # 5) If still not exactly 3 non-empty strings, do ONE correction call to LLM with explicit requirement
    if (len(tags) != 3 
        or any(not isinstance(t, str) 
        or not t.strip() for t in tags)
        or not isinstance(summary, str)
        or not summary.strip()
        or wc(summary) > 25
        ):
        system = (
            "Fix the output.\n"
            'Return ONLY JSON: {"tags": ["t1","t2","t3"], "summary": "one sentence <=25 words"}\n'
            "tags must be EXACTLY 3 strings."
        )
        user_obj = {"title": title, "content": content, "planner": plan, "reviewer": review}
        raw = ollama_chat(
            [{"role": "system", "content": system}, {"role": "user", "content": json.dumps(user_obj)}],
            temperature=0.1,
        )
        obj = extract_json(raw)
        tags = normalize_tags(obj.get("tags"))[:3]
        summary = obj.get("summary") if isinstance(obj.get("summary"), str) else summary

    # Final build
    final = build_final(tags, summary)

    # last validation
    if not isinstance(final.get("tags"), list) or len(final["tags"]) != 3:
        raise RuntimeError("Finalizer could not produce exactly 3 tags.")
    if not isinstance(final.get("summary"), str) or not final["summary"].strip() or wc(final["summary"]) > 25:
        raise RuntimeError("Finalizer could not produce <=25-word summary.")

    return final


# Main
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--title", required=True)
    ap.add_argument("--content", help="If omitted, read from stdin")
    args = ap.parse_args()

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
