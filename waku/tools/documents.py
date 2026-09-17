"""search_documents / save_html — a local folder of documents the agent can
read, and one designated place for the HTML it writes back.

Local-first like the rest of waku's tools: the folder is `.waku/documents/`
unless WAKU_DOCS_DIR points at one you already have, and generated HTML lands
in its `html/` subfolder. Nothing is uploaded, served, or opened — the tools
return the path so the user opens the file themselves.

Reading is stdlib-only, so search is honest about its limits: it looks inside
plain-text formats (.md, .txt, .html, .json, ...) and REPORTS the files it had
to skip. A .pdf in the folder would otherwise make "no matches" mean two very
different things. Parsing pdf/docx needs a dependency the core doesn't take;
convert those to text, or put the parser behind an extra.
"""

from __future__ import annotations

import os
from datetime import UTC, datetime
from pathlib import Path

from waku.tools.registry import Tool

DOCS_DIR_ENV = "WAKU_DOCS_DIR"              # the folder itself; default <home>/documents
HTML_SUBDIR_ENV = "WAKU_DOCS_HTML_SUBDIR"   # designated output folder; default "html"

# What can be read as text. Everything else is counted and named as skipped.
TEXT_SUFFIXES = {".txt", ".md", ".markdown", ".rst", ".html", ".htm", ".xml",
                 ".json", ".yaml", ".yml", ".toml", ".ini", ".cfg",
                 ".csv", ".tsv", ".log", ".py", ".sql"}

MAX_MATCHES = 40        # matching lines returned in total
MAX_PER_FILE = 5        # so one long file can't crowd out the rest of the folder
MAX_LISTED = 50         # files named when listing the folder
SNIPPET_CHARS = 200


def docs_root(home: Path) -> Path:
    """The documents folder. Same posture as waku/tools/workspace.py: one env
    var, resolved once, so every caller agrees on which folder is meant."""
    override = os.getenv(DOCS_DIR_ENV, "").strip()
    return (Path(override).expanduser() if override else home / "documents").resolve()


def html_dir(home: Path) -> Path:
    """The designated subfolder for generated HTML. `.name` keeps the override
    to a single folder, so WAKU_DOCS_HTML_SUBDIR can't aim it up and out."""
    name = Path(os.getenv(HTML_SUBDIR_ENV, "").strip()).name
    return (docs_root(home) / (name or "html")).resolve()


def _split_by_readability(folder: Path) -> tuple[list[Path], list[Path]]:
    readable: list[Path] = []
    skipped: list[Path] = []
    for path in sorted(p for p in folder.rglob("*") if p.is_file()):
        if path.name.startswith("."):
            continue
        (readable if path.suffix.lower() in TEXT_SUFFIXES else skipped).append(path)
    return readable, skipped


def _limits_note(skipped: list[Path], unreadable: list[str], capped: list[str]) -> str:
    """Every way this search fell short of the whole folder, in one sentence.

    A silent limit must never read as "nothing there" — that is what lets the
    agent say "no matches in your markdown, but there are 3 PDFs I can't read"
    instead of "you have nothing on that". The per-file cap belongs here for the
    same reason: five hits out of fifty is a sample, not an answer.
    """
    notes = []
    if skipped:
        kinds = sorted({p.suffix.lower() or "(no extension)" for p in skipped})
        notes.append(f"{len(skipped)} file(s) not searched — no text parser for {', '.join(kinds)}")
    if unreadable:
        shown = ", ".join(unreadable[:5])
        notes.append(f"{len(unreadable)} file(s) unreadable as text: {shown}")
    if capped:
        shown = ", ".join(capped[:5])
        notes.append(f"only the first {MAX_PER_FILE} matches are shown for {shown}")
    return ("\n\nNote: " + "; ".join(notes) + ".") if notes else ""


def _inventory(folder: Path, root: Path, readable: list[Path], skipped: list[Path]) -> str:
    if not readable and not skipped:
        return f"{folder} is empty — no documents to search."
    lines = []
    for path in readable[:MAX_LISTED]:
        try:
            size = path.stat().st_size
        except OSError:
            size = 0
        lines.append(f"- {path.relative_to(root)} ({size} bytes)")
    more = f"\n(+{len(readable) - MAX_LISTED} more)" if len(readable) > MAX_LISTED else ""
    head = f"{len(readable)} searchable document(s) in {folder}:"
    return "\n".join([head, *lines]) + more + _limits_note(skipped, [], [])


def make_search_tool(home: Path) -> Tool:
    def search_documents(query: str = "", subfolder: str = "", max_results: int = MAX_MATCHES) -> str:
        root = docs_root(home)
        if not root.is_dir():
            return (f"No documents folder yet — expected {root}. Create it (or point "
                    f"{DOCS_DIR_ENV} at a folder you already have) and put files there.")

        folder = (root / subfolder).resolve() if subfolder else root
        # Resolve first, then check containment — the same check gather.py makes
        # before writing. "../../.ssh" is a folder name to a model, not an attack.
        if folder != root and root not in folder.parents:
            return f"refused: '{subfolder}' is outside the documents folder ({root})."
        if not folder.is_dir():
            return f"No such folder: {subfolder or folder} (looked inside {root})."

        readable, skipped = _split_by_readability(folder)
        if not query.strip():
            return _inventory(folder, root, readable, skipped)

        limit = max(1, min(int(max_results or MAX_MATCHES), MAX_MATCHES))
        needle = query.strip().lower()
        hits: list[str] = []
        unreadable: list[str] = []
        capped: list[str] = []
        truncated = False
        for path in readable:
            try:
                text = path.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError):
                unreadable.append(path.name)
                continue
            found = 0
            for lineno, line in enumerate(text.splitlines(), 1):
                if needle not in line.lower():
                    continue
                if found >= MAX_PER_FILE:
                    capped.append(path.name)
                    break
                if len(hits) >= limit:
                    truncated = True
                    break
                hits.append(f"- {path.relative_to(root)}:{lineno}: {line.strip()[:SNIPPET_CHARS]}")
                found += 1
            if truncated:
                break

        if not hits:
            return (f"No matches for '{query}' in {len(readable)} document(s) under {folder}."
                    + _limits_note(skipped, unreadable, capped))
        head = (f"First {len(hits)} match(es) for '{query}' in {folder} (more exist):"
                if truncated else f"{len(hits)} match(es) for '{query}' in {folder}:")
        return "\n".join([head, *hits]) + _limits_note(skipped, unreadable, capped)

    return Tool(
        name="search_documents",
        description=(
            "Search the user's local documents folder and read what the files say. Use "
            "whenever the user asks what a document contains, or to look something up in "
            "their files, notes, or saved pages. Pass a distinctive word or phrase as the "
            "query; call it with no query to list which documents are there. Results are "
            "'path:line: text'. Only plain-text formats are searched and the answer names "
            "any file it could not read — repeat that caveat instead of reporting an empty "
            "folder as an empty answer."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Word or phrase to look for. Omit to list the folder instead.",
                },
                "subfolder": {
                    "type": "string",
                    "description": "Optional subfolder to narrow the search to, e.g. 'html'",
                },
                "max_results": {
                    "type": "integer",
                    "description": f"Max matching lines to return (default/max {MAX_MATCHES})",
                },
            },
            "required": [],
        },
        fn=search_documents,
    )


def _safe_html_name(filename: str) -> str:
    """One flat `.html` file name, slugified like messages.py names its outbox
    files. `Path(...).name` drops any directory the model tried to include, so
    the designated folder stays the only destination."""
    stem = Path(filename.strip()).name
    if stem in ("", ".", ".."):
        return ""
    if stem.lower().endswith((".html", ".htm")):
        stem = stem.rsplit(".", 1)[0]
    cleaned = "".join(c if (c.isalnum() or c == "_") else "-" for c in stem)
    while "--" in cleaned:
        cleaned = cleaned.replace("--", "-")
    cleaned = cleaned.strip("-")
    return f"{cleaned[:60]}.html" if cleaned else ""


def _next_free(folder: Path, stem: str) -> Path:
    for n in range(2, 100):
        candidate = folder / f"{stem}-{n}.html"
        if not candidate.exists():
            return candidate
    return folder / f"{stem}-{datetime.now(UTC):%Y%m%dT%H%M%S}.html"


def make_save_html_tool(home: Path) -> Tool:
    def save_html(filename: str = "", html: str = "") -> str:
        # Defensive: a partial tool call should come back as a sentence the
        # model can act on, not a TypeError. Same as create_event.
        if not filename or not html:
            return ("save_html needs both a filename and the html content. "
                    "Please call it again with both.")

        name = _safe_html_name(filename)
        if not name:
            return f"refused: '{filename}' is not a usable file name."

        dest_dir = html_dir(home)
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest = (dest_dir / name).resolve()
        if dest_dir not in dest.parents:
            return f"refused: path escaped the html folder ({dest_dir})."

        body = html if html.endswith("\n") else html + "\n"
        kept = ""
        if dest.exists():
            try:
                unchanged = dest.read_text(encoding="utf-8") == body
            except (OSError, UnicodeDecodeError):
                unchanged = False
            # Same name + same content is a no-op, so a retrying model doesn't
            # litter the folder with copies. Same name + DIFFERENT content gets
            # a new file: overwriting would silently destroy a saved page.
            if unchanged:
                return f"'{name}' is already saved at {dest}, unchanged. Nothing rewritten."
            dest = _next_free(dest_dir, Path(name).stem)
            kept = f" '{name}' already existed with different content and was left untouched."

        dest.write_text(body, encoding="utf-8")
        return (f"Saved {len(body)} characters of HTML to {dest}." + kept
                + " It is a local file — nothing was published or served; open it in a "
                  "browser to view it.")

    return Tool(
        name="save_html",
        description=(
            "Save an HTML document you generated (a report, page, table, or summary) into "
            "the designated html subfolder of the user's documents folder. Use when the "
            "user asks you to write up, export, or save something as an HTML file. Pass "
            "the complete document including <html> and <body>. The file is written "
            "locally only; an existing file of the same name is never overwritten."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "filename": {
                    "type": "string",
                    "description": "File name without a path, e.g. 'week-summary.html'",
                },
                "html": {"type": "string", "description": "The complete HTML document"},
            },
            "required": ["filename", "html"],
        },
        fn=save_html,
    )
