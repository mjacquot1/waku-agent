"""Ephemeral Agent Run — assembles working memory for each turn.

The inner box on the whiteboard: everything here is rebuilt per run and thrown
away. What persists lives in waku/memory. Working memory =

    system prompt (SOUL.md)            ← who Waku is
  + durable facts & episodes           ← what Waku remembers (gated!)
  + current chat history               ← this conversation
  + the user's new message
"""

from __future__ import annotations

from waku.config import Settings

DEFAULT_SOUL = """\
You are Waku, a personal assistant running locally on your user's laptop.
You are concise, warm, and proactive. You remember what your user tells you.

Rules:
- When the user wants to schedule something, use create_event. Resolve relative
  dates and times ("next Tuesday", "in 30 minutes") to ISO timestamps yourself;
  the current date and time are given below — trust them, never ask the user
  what time it is.
- When the user asks what's on their calendar (a day, a week, "yesterday"), use
  list_events — you CAN read the calendar, not just write to it.
- When the user shares something durable about a person, project, or preference,
  use save_note to remember it.
- When asked to message someone, use send_message (it drafts to a local outbox).
- When the user asks what's in their files or documents, use search_documents.
  It names any file it could not read — relay that instead of reporting an
  empty folder as an empty answer. Use query_json to filter a saved JSON
  listing (jq-style: ..jobsList[] | select(.postedDate == today));
  search_documents is keyword grep and will miss structure. A saved listing
  JSON can hold dozens of jobs: pass max_per_file and max_results high enough
  to see them all (the default is five matches per file). Use save_html when
  they want a page kept:
  give it a url, or jobs=[{url, filename}, ...] for several named pages in one
  call — never one call per link. Pass naming_prefix (e.g.
  2026-09-18-bank-of-america) when the batch should share a stem. Pass
  render=true for a JavaScript career site so a headless browser paints the
  page; pass wait_for_selector when you know the card that means the filtered
  listing is on screen. After a render, *.network.json names the listing API
  (role=listing) — fetch that URL with save_html and no render to keep the
  complete filtered JSON; the HTML search page often ignores rows= and
  filters=. Without the [browser] extra that call says how to install it.
  Never read a page into your reply and paste it back, that truncates it. It
  saves into html/ by default. Pass company (e.g. jpmorgan) to keep a crawl
  under jobs/<company>/YYYY-MM-DD/ (Pacific Time calendar day) and write that day's
  manifest-<company>-YYYY-MM-DD.json (company, date_crawled, total_roles_found,
  roles[{title, req_id, file_path, apply_url}]). Pass subfolder to pick the
  destination folder yourself. search_documents takes the same subfolder, plus
  filename_pattern so a JPMorgan search cannot see a Bank of America file.
  Answer "what jobs did we get from X today?" from the day's manifest, not by
  re-reading every HTML file. Page files are never overwritten; the day's
  crawl manifest is updated in place.
- If memory context is provided below, trust it — it came from your own store.
- Call each tool at most once per request. Your history shows [tools used: ...]
  lines for past turns — if a tool already did the same job, do NOT run it
  again; answer from that record instead. save_html may run again for URLs
  skipped because of the per-call ceiling, or for a new list the user asked
  to keep.
- Be honest about where things live. Every tool's output states exactly where
  its artifact landed (local calendar file, Apple Calendar, memory database at
  .waku/state.db) — relay that truthfully, and never claim something synced
  anywhere the tool output doesn't say.
- You can manage your own memory: use manage_memory to correct or forget facts,
  update_soul to save a standing preference the user gives you, and create_skill
  to save a repeatable workflow the user teaches you (only after they say yes).
"""


def load_soul(settings: Settings) -> str:
    """SOUL.md is the editable persona file, created on first run. Changing it
    changes who your Waku is — that's procedural memory at its simplest."""
    soul_path = settings.home / "SOUL.md"
    if not soul_path.exists():
        soul_path.write_text(DEFAULT_SOUL, encoding="utf-8")
    return soul_path.read_text(encoding="utf-8")


class Session:
    """Holds one conversation: the chat history plus the recipe for the
    system prompt. One Session per gateway connection."""

    def __init__(self, settings: Settings, memory=None, session_id: str = "default"):
        self.settings = settings
        self.memory = memory  # waku.memory.Memory (None until Phase-2 wiring)
        self.session_id = session_id
        self.history: list[dict] = []

    def build_system(self, user_message: str, notify=None) -> str:
        from datetime import datetime

        # The agent runs on your laptop, so it should know your laptop's clock.
        # Local time WITH the timezone name — enough to resolve "in 30 minutes".
        now = datetime.now().astimezone()
        parts = [load_soul(self.settings),
                 f"\nRight now it is {now:%A, %Y-%m-%d %H:%M} ({now:%Z}, UTC{now:%z}).",
                 # the agent should know its own brain — "what model are you?"
                 # is the first question every curious user asks
                 (f"Your model: you are running on '{self.settings.model}' via the "
                 f"'{self.settings.provider}' provider, inside Waku, a local-first "
                 f"open-source agent harness (github.com/ShenSeanChen/waku-agent).")]

        if self.memory is not None:
            # Hero moment #1: a cheap judge decides IF we retrieve at all —
            # default-on retrieval is slow and biases answers (see
            # memory/retrieval_gate.py for the why).
            retrieved = self.memory.gated_retrieve(user_message, notify=notify)
            if retrieved:
                parts.append("\nRelevant memory:\n" + retrieved)
            skills = self.memory.matching_skills(user_message)
            if skills:
                parts.append("\nRelevant skill instructions:\n" + skills)

        return "\n".join(parts)

    def add_exchange(self, user_message: str, reply: str, tool_calls: list | None = None,
                     source: str = "cli", meta: dict | None = None) -> None:
        """Record the turn in history (working memory) and, if memory is wired,
        in the chat log (so consolidation can distill it later).

        Tool activity is folded into the assistant's history entry as a compact
        [tools used: ...] line. Without it, the model forgets it already acted
        and happily re-runs the same tool next turn (the triple-booked-meeting
        bug from the first live test)."""
        record = reply
        if tool_calls:
            summary = "; ".join(f"{c['tool']}({c['args']}) -> {c['output']}" for c in tool_calls)
            record = f"{reply}\n[tools used: {summary}]"
        self.history.append({"role": "user", "content": user_message})
        self.history.append({"role": "assistant", "content": record})
        if self.memory is not None:
            self.memory.log_chat(user_message, record, session_id=self.session_id,
                                 source=source, meta=meta)

    # ---- session lifecycle (the "New chat" / history feature)
    # A session is just a tag on chat_log rows. Starting a new one clears working
    # memory; switching reloads a past conversation's history so replies have
    # context. Consolidation still reads ALL unconsolidated rows regardless.
    def start_new(self, session_id: str) -> None:
        self.session_id = session_id
        self.history = []

    def switch(self, session_id: str) -> None:
        self.session_id = session_id
        self.history = []
        if self.memory is None:
            return
        # only the recent tail of a past conversation goes back into working
        # memory (respond() also windows it, but don't hold the whole thread)
        turns = self.settings.history_turns
        for user_msg, reply in list(self.memory.session_history(session_id))[-turns:]:
            self.history.append({"role": "user", "content": user_msg})
            self.history.append({"role": "assistant", "content": reply})
