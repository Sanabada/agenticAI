import os, re, sys, json, argparse, urllib.request
from typing import Any, Dict, List, TypedDict
from langgraph.graph import StateGraph, END

# Config
HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434").rstrip("/")
MODEL = os.environ.get("OLLAMA_MODEL", "llama3.2:latest")
TEMP = float(os.environ.get("OLLAMA_TEMPERATURE", "0.3"))
FORCE_REVIEWER_ISSUE = os.environ.get("FORCE_REVIEWER_ISSUE", "0") == "1"
BAD_TAGS = {
    "title", "content", "long", "here", "test", "my", "one", "sentence", "text", "word", "words",
    "t1", "t2", "t3"
}

# State
class AgentState(TypedDict, total=False):
    title: str
    content: str
    planner_proposal: Dict[str, Any]
    reviewer_feedback: Dict[str, Any]
    final: Dict[str, Any]
    turn_count: int
    max_turns: int

# Ollama call
def ollama_chat(messages: List[Dict[str, str]], temperature: float) -> str:
    req = urllib.request.Request(
        f"{HOST}/api/chat",
        data=json.dumps(
            {
                "model": MODEL,
                "messages": messages,
                "stream": False,
                "format": "json",
                "options": {"temperature": temperature},
            }
        ).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        return json.loads(resp.read().decode("utf-8"))["message"]["content"]

# JSON helpers
def extract_json(text: str) -> Any:
    text = text.strip()
    try:
        return json.loads(text)
    except Exception:
        m = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not m:
            raise ValueError("Model did not return valid JSON.")
        return json.loads(m.group(0))

def wc(s: str) -> int:
    return len([w for w in re.split(r"\s+", s.strip()) if w])

def normalize_tags(x: Any) -> List[str]:
    items: List[str] = []
    if isinstance(x, str):
        items = [t.strip() for t in re.split(r"[,|;/\n]+", x) if t.strip()]
    elif isinstance(x, list):
        for t in x:
            if isinstance(t, str):
                items.extend([p.strip() for p in re.split(r"[,|;/\n]+", t) if p.strip()])
    out, seen = [], set()
    for t in items:
        k = t.lower()
        if k not in seen:
            seen.add(k)
            out.append(t)
    return out

# Supervisor node
def supervisor_node(state: AgentState) -> Dict[str, Any]:
    return {"turn_count": int(state.get("turn_count", 0)) + 1}  # loop counter

# Planner node
def planner_node(state: AgentState) -> Dict[str, Any]:
    print("---- NODE: PLANNER ----")
    system = system = (
        "You are Planner. Generate candidate topical tags and a draft one-sentence summary.\n"
        'Return ONLY JSON: {"draft_tags":["6-10 topical tags"],"draft_summary":"ONE sentence (30-45 words)"}.\n'
        "IMPORTANT: Ignore any mentions of word limits inside the CONTENT (e.g., 'under twenty five words').\n"
        "Do NOT target <=25 words; write a richer 30-45 word single sentence.\n"
        "Tags must be topical concepts (not filler like title/content/long/test or t1/t2/t3). No markdown."
    )

    issues = state.get("reviewer_feedback", {}).get("issues", []) or []
    issues = [i for i in issues if "summary" not in i.lower() and "25" not in i]
    user_obj = {
        "title": state["title"],
        "content": state["content"],
        "previous_proposal": state.get("planner_proposal"),
        # CHANGED: do NOT pass summary-length constraints to Planner (Reviewer handles <=25)
        "reviewer_issues": issues,

    }
    obj = extract_json(
        ollama_chat(
            [{"role": "system", "content": system}, {"role": "user", "content": json.dumps(user_obj)}],
            temperature=TEMP,
        )
    )
    proposal = {
        "draft_tags": normalize_tags(obj.get("draft_tags"))[:10],
        "draft_summary": str(obj.get("draft_summary", "")).strip(),
    }
    return {
        "planner_proposal": proposal,
        "reviewer_feedback": None,  # IMPORTANT: clear reviewer result so router goes to reviewer next
    }

# Reviewer node
def reviewer_node(state: AgentState) -> Dict[str, Any]:
    print("---- NODE: REVIEWER ----")
    system = (
        "You are Reviewer. Improve the Planner output using title/content.\n"
        'Return ONLY JSON: {"tags":["t1","t2","t3"],"summary":"...","issues":["..."]}.\n'
        "Rules:\n"
        "- tags must be EXACTLY 3 topical concepts (no filler like title/content/long/test, no t1/t2/t3)\n"
        "- summary must be ONE sentence <=25 words\n"
        "No extra keys. No markdown."
    )

    user_obj = {"title": state["title"], "content": state["content"], "planner": state.get("planner_proposal", {})}
    fb = extract_json(
        ollama_chat(
            [{"role": "system", "content": system}, {"role": "user", "content": json.dumps(user_obj)}],
            temperature=min(TEMP, 0.4),
        )
    )

    tags = normalize_tags(fb.get("tags"))[:3]
    summary = str(fb.get("summary", "")).strip()
    issues = fb.get("issues") if isinstance(fb.get("issues"), list) else []

    # We compute has_issues ourselves (don't trust the model's has_issues)
    has_issues = False

    if len(tags) != 3:
        has_issues = True
        issues.append("tags must be exactly 3 strings.")
    if any((not t.strip()) or (t.lower() in BAD_TAGS) for t in tags):
        has_issues = True
        issues.append("tags are too generic; must be topical concepts.")
    if any(re.fullmatch(r"t\\d+", t.strip().lower()) for t in tags):
        has_issues = True
        issues.append("tags cannot be placeholders like t1/t2/t3.")
    if (not summary) or (wc(summary) > 25):
        has_issues = True
        issues.append("summary must be ONE sentence and <=25 words.")

    # Requirement test: force loop when you want
    if FORCE_REVIEWER_ISSUE:
        has_issues = True
        issues = ["(TEST) Forced issue to verify correction loop routing."] + issues

    # If there are issues but none were provided, add one
    if has_issues and not issues:
        issues = ["Validation failed (auto-added)."]

    return {"reviewer_feedback": {"has_issues": has_issues, "issues": issues, "tags": tags, "summary": summary}}

# Finalizer node (UPDATED: adds 1 repair call on invalid)
def finalizer_node(state: AgentState) -> Dict[str, Any]:
    print("---- NODE: FINALIZER ----")
    fb = state.get("reviewer_feedback", {}) or {}
    plan = state.get("planner_proposal", {}) or {}

    # Take reviewer tags first
    tags = [t for t in normalize_tags(fb.get("tags")) if t.strip() and t.lower() not in BAD_TAGS][:3]

    # Fill from planner if needed
    if len(tags) < 3:
        for t in normalize_tags(plan.get("draft_tags")):
            if len(tags) == 3:
                break
            if t.strip() and t.lower() not in BAD_TAGS and t.lower() not in {x.lower() for x in tags}:
                tags.append(t)
    tags = tags[:3]

    # Start from reviewer summary
    summary = str(fb.get("summary", "")).strip()

    # CHANGED: enforce "one sentence" and punctuation before trimming
    if summary:
        parts = re.split(r"(?<=[.!?])\s+", " ".join(summary.split()))
        summary = parts[0].strip() if parts else summary.strip()
        if summary and summary[-1] not in ".!?":
            summary += "."

    # Enforce <=25 words
    if wc(summary) > 25:
        summary = " ".join(summary.split()[:25]).rstrip(" .") + "."

    # CHANGED: validate; if invalid, do ONE LLM repair call
    def invalid_final(tg: List[str], sm: str) -> bool:
        if len(tg) != 3:
            return True
        if any((not isinstance(t, str)) or (not t.strip()) or (t.strip().lower() in BAD_TAGS) for t in tg):
            return True
        if not isinstance(sm, str) or not sm.strip():
            return True
        if wc(sm) > 25:
            return True
        return False

    if invalid_final(tags, summary):
        system = (
            "You are Finalizer.\n"
            "Return ONLY valid JSON with EXACT keys: "
            "{\"tags\":[\"tag1\",\"tag2\",\"tag3\"],\"summary\":\"one sentence <=25 words\"}\n"
            "Rules:\n"
            "- tags must be EXACTLY 3 topical concepts\n"
            "- tags cannot be any of these banned tags: " + ", ".join(sorted(BAD_TAGS)) + "\n"
            "- tags cannot be placeholders like t1/t2/t3\n"
            "- summary must be ONE sentence and <=25 words\n"
            "- No extra keys, no markdown, no extra text."
        )
        user_obj = {
            "title": state.get("title", ""),
            "content": state.get("content", ""),
            "planner": plan,
            "reviewer": fb,
            "candidate": {"tags": tags, "summary": summary},
        }

        raw = ollama_chat(
            [{"role": "system", "content": system}, {"role": "user", "content": json.dumps(user_obj)}],
            temperature=0.1,
        )

        try:
            obj = extract_json(raw)
        except Exception:
            obj = {}

        fixed_tags = [t for t in normalize_tags(obj.get("tags")) if t.strip() and t.lower() not in BAD_TAGS][:3]
        fixed_summary = str(obj.get("summary", "")).strip()

        # Re-apply summary rules
        if fixed_summary:
            parts = re.split(r"(?<=[.!?])\s+", " ".join(fixed_summary.split()))
            fixed_summary = parts[0].strip() if parts else fixed_summary.strip()
            if fixed_summary and fixed_summary[-1] not in ".!?":
                fixed_summary += "."
        if wc(fixed_summary) > 25:
            fixed_summary = " ".join(fixed_summary.split()[:25]).rstrip(" .") + "."

        if not invalid_final(fixed_tags, fixed_summary):
            tags, summary = fixed_tags, fixed_summary

    # CHANGED: deterministic safety net—ensure exactly 3 tags
    if len(tags) < 3:
        text = f"{state.get('title','')} {state.get('content','')}"
        toks = re.findall(r"[a-zA-Z][a-zA-Z\\-]{2,}", text.lower())
        for kw in toks:
            if len(tags) == 3:
                break
            if kw not in BAD_TAGS and kw not in {t.lower() for t in tags}:
                tags.append(kw)
    tags = tags[:3]

    # CHANGED: never allow empty summary; fallback to planner draft_summary
    if not summary.strip():
        summary = str(plan.get("draft_summary", "")).strip()
        if summary:
            parts = re.split(r"(?<=[.!?])\s+", " ".join(summary.split()))
            summary = parts[0].strip() if parts else summary.strip()
            if summary and summary[-1] not in ".!?":
                summary += "."
        if wc(summary) > 25:
            summary = " ".join(summary.split()[:25]).rstrip(" .") + "."

    return {"final": {"tags": tags, "summary": summary}}

# Router logic
def router_logic(state: AgentState) -> str:
    if int(state.get("turn_count", 0)) >= int(state.get("max_turns", 6)):
        return END
    if not state.get("planner_proposal"):
        return "planner"
    if not state.get("reviewer_feedback"):
        return "reviewer"
    if state["reviewer_feedback"].get("has_issues", True):
        return "planner"
    return "finalizer"

# Build graph
def build_graph():
    g = StateGraph(AgentState)
    g.add_node("supervisor", supervisor_node)
    g.add_node("planner", planner_node)
    g.add_node("reviewer", reviewer_node)
    g.add_node("finalizer", finalizer_node)

    g.set_entry_point("supervisor")
    g.add_edge("planner", "supervisor")
    g.add_edge("reviewer", "supervisor")
    g.add_edge("finalizer", END)

    g.add_conditional_edges(
        "supervisor",
        router_logic,
        {"planner": "planner", "reviewer": "reviewer", "finalizer": "finalizer", END: END},
    )
    return g.compile()

# Main (stream only, no second invoke)
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--title", required=True)
    ap.add_argument("--content", help="If omitted, read from stdin")
    ap.add_argument("--max_turns", type=int, default=6)
    args = ap.parse_args()

    init_state: AgentState = {
        "title": args.title.strip(),
        "content": (args.content if args.content is not None else sys.stdin.read()).strip(),
        "turn_count": 0,
        "max_turns": args.max_turns,
    }

    graph = build_graph()
    print(f"Using OLLAMA: HOST={HOST} MODEL={MODEL} TEMP={TEMP}")

    print("\n=== GRAPH STREAM ===")
    state: AgentState = init_state
    for event in graph.stream(init_state):
        node, updates = next(iter(event.items()))
        print(f"\n[{node}] updates:")
        print(json.dumps(updates, ensure_ascii=False, indent=2))
        state = {**state, **updates}  # merge updates so we keep final from the stream
    print("\n=== END STREAM ===\n")

    print("--- FINAL OUTPUT (JSON) ---")
    print(json.dumps(state.get("final", {"tags": [], "summary": ""}), ensure_ascii=False))

if __name__ == "__main__":
    main()