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

save_html takes a `url` as well as literal html, and that is the interesting
half. Asking the model to echo a page back so it can be saved costs tokens
twice, and the copy is only as complete as the context window allowed — a long
job posting comes back silently truncated. Fetching in the harness instead
means the bytes go socket → disk without ever entering the prompt, so the file
is the server's exact response and the model only reads back a status line.

The fetch is plain HTTP over stdlib urllib (same as search.py and catalog.py —
the core takes no new dependencies), which cannot run JavaScript. Pages that
render client-side, like Workday and Taleo job boards, come back as their empty
shell; the tool detects that and SAYS so rather than handing over a file that
looks saved and is blank. A headless-browser backend belongs behind an extra,
the way [voice] and [telegram] are, not in the default install.
"""

from __future__ import annotations

import os
import urllib.error
import urllib.request
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import urlsplit

from waku.tools.registry import Tool

DOCS_DIR_ENV = "WAKU_DOCS_DIR"              # the folder itself; default <home>/documents
HTML_SUBDIR_ENV = "WAKU_DOCS_HTML_SUBDIR"   # designated output folder; default "html"
FETCH_MAX_MB_ENV = "WAKU_FETCH_MAX_MB"      # size ceiling for a fetched page

# Some sites answer Python-urllib/3.x with a 403 (see catalog.py for the same
# problem against model catalogs), so ask the way a browser would.
FETCH_UA = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"
FETCH_TIMEOUT = 20
DEFAULT_FETCH_MAX_MB = 25

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


def _slug_from_url(url: str) -> str:
    """A name for a page the user never named. The last path segment is usually
    the human-meaningful part (.../software-engineer-iii-26032981); when the
    path is just '/', the host is the only thing left to call it."""
    parts = urlsplit(url)
    return Path(parts.path.rstrip("/")).name or parts.netloc or "page"


def _max_fetch_bytes() -> int:
    try:
        mb = int(os.getenv(FETCH_MAX_MB_ENV, "") or DEFAULT_FETCH_MAX_MB)
    except ValueError:
        mb = DEFAULT_FETCH_MAX_MB
    return max(1, mb) * 1024 * 1024


class _FetchError(Exception):
    """Carries a sentence the model can act on, not a stack trace."""


class _HTTPOnlyRedirects(urllib.request.HTTPRedirectHandler):
    """A fetch that starts on the web has to stay on the web.

    The destination of a redirect is chosen by the server, so the scheme check
    on the url the model passed says nothing about where the request ends up.
    urllib refuses a hop to file:// on its own, but it allows ftp:// — this
    narrows the allowed set to the two schemes the tool actually claims.
    """

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        if urlsplit(newurl).scheme not in ("http", "https"):
            raise urllib.error.URLError(f"refused a redirect to '{newurl}'")
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def _looks_js_rendered(payload: bytes) -> bool:
    """Cheap check for "the server sent an app, not a page". Worth doing badly:
    a Workday posting saved as a 900-byte script tag is a silent failure the
    user only discovers when they open the file weeks later."""
    head = payload[:4000].lower()
    return (b"enable javascript" in head
            or b"you need to enable" in head
            or (len(payload) < 1024 and b"<script" in head))


def _fetch(url: str) -> tuple[bytes, str, str]:
    """GET `url` and return its raw bytes, a one-line status, and a warning if
    the response looks like a JavaScript shell.

    Bytes, deliberately — not decoded text. The point of fetching here instead
    of in the prompt is that what lands on disk is the server's exact response,
    so nothing is re-encoded, re-indented, or cut to fit a context window.
    """
    parts = urlsplit(url.strip())
    if parts.scheme not in ("http", "https"):
        raise _FetchError(f"only http and https are fetched, not '{parts.scheme or url}'")
    if not parts.netloc:
        raise _FetchError("that is not a complete url — it has no host")

    limit = _max_fetch_bytes()
    request = urllib.request.Request(url, headers={
        "User-Agent": FETCH_UA,
        "Accept": "text/html,*/*",
        # urllib does not decompress, and these bytes go straight to disk. Ask
        # for identity so the .html file is html, not a gzip blob named .html.
        "Accept-Encoding": "identity",
    })
    opener = urllib.request.build_opener(_HTTPOnlyRedirects)
    try:
        with opener.open(request, timeout=FETCH_TIMEOUT) as response:
            # One byte past the ceiling, so "too big" is detectable without
            # holding a whole oversized page in memory to find out.
            payload = response.read(limit + 1)
            status, ctype = response.status, response.headers.get("Content-Type", "unknown")
            encoding = (response.headers.get("Content-Encoding") or "identity").lower()
    except urllib.error.HTTPError as exc:
        raise _FetchError(f"the server answered HTTP {exc.code} {exc.reason}") from exc
    except Exception as exc:
        raise _FetchError(str(exc) or exc.__class__.__name__) from exc

    if len(payload) > limit:
        raise _FetchError(
            f"the page is over the {limit // (1024 * 1024)} MB ceiling. Nothing was saved — "
            f"half a page is not a copy. Raise {FETCH_MAX_MB_ENV} to keep it anyway")

    warning = ""
    if encoding != "identity":
        # Asked for identity and got something else. The bytes are still the
        # exact response, but calling that "the page" would be a lie.
        warning = (f" Warning: the server sent {encoding}-compressed bytes despite being asked "
                   "for none, so the file holds the compressed response, not readable html. "
                   f"Decompress it before reading ({encoding}).")
    elif _looks_js_rendered(payload):
        warning = (" Warning: the response is mostly a JavaScript shell, so this file is the "
                   "raw HTTP body, not the page a person sees. Boards that render client-side "
                   "(Workday, Taleo) need a headless browser to capture properly.")
    return payload, f"HTTP {status}, {ctype}, {len(payload)} bytes", warning


def _next_free(folder: Path, stem: str) -> Path:
    for n in range(2, 100):
        candidate = folder / f"{stem}-{n}.html"
        if not candidate.exists():
            return candidate
    return folder / f"{stem}-{datetime.now(UTC):%Y%m%dT%H%M%S}.html"


def make_save_html_tool(home: Path) -> Tool:
    def save_html(filename: str = "", html: str = "", url: str = "") -> str:
        # Defensive: a partial tool call should come back as a sentence the
        # model can act on, not a TypeError. Same as create_event.
        if url and html:
            return ("save_html takes either url or html, not both — url to download a live "
                    "page, html to save a document you wrote. Please call it again with one.")
        if not url and not html:
            return ("save_html needs either a url to download or the html content to write. "
                    "Please call it again with one of them.")

        warning = ""
        if url:
            try:
                payload, status, warning = _fetch(url)
            except _FetchError as exc:
                return f"Nothing saved — could not fetch {url}: {exc}."
            # A page the user pointed at by link usually has no name of its own.
            filename = filename or _slug_from_url(url)
        else:
            # Text the model composed, so normalise the trailing newline. A
            # fetched page is never touched — that is the whole promise.
            payload = (html if html.endswith("\n") else html + "\n").encode("utf-8")

        name = _safe_html_name(filename)
        if not name:
            return f"refused: '{filename}' is not a usable file name."

        dest_dir = html_dir(home)
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest = (dest_dir / name).resolve()
        if dest_dir not in dest.parents:
            return f"refused: path escaped the html folder ({dest_dir})."

        kept = ""
        if dest.exists():
            try:
                unchanged = dest.read_bytes() == payload
            except OSError:
                unchanged = False
            # Same name + same content is a no-op, so a retrying model doesn't
            # litter the folder with copies. Same name + DIFFERENT content gets
            # a new file: overwriting would silently destroy a saved page.
            if unchanged:
                return (f"'{name}' is already saved at {dest}, unchanged. "
                        "Nothing rewritten." + warning)
            dest = _next_free(dest_dir, Path(name).stem)
            kept = f" '{name}' already existed with different content and was left untouched."

        dest.write_bytes(payload)
        saved = (f"Fetched {url} ({status}) and saved it byte-for-byte to {dest}." if url
                 else f"Saved {len(payload)} bytes of HTML to {dest}.")
        return (saved + kept + warning
                + " It is a local file — nothing was published or served; open it in a "
                  "browser to view it.")

    return Tool(
        name="save_html",
        description=(
            "Save an HTML page into the designated html subfolder of the user's documents "
            "folder, either by downloading a url or by writing html you composed. Pass "
            "exactly one of them. "
            "PREFER url whenever the user points at a web page (a job posting, an "
            "article, a listing): waku downloads it over HTTP and writes the server's "
            "exact bytes to disk, so the saved file is complete and no part of the page "
            "has to pass through your context — do NOT fetch a page into your context and "
            "echo it back as html, that truncates long pages. Pass html only for a "
            "document you are authoring yourself. filename is optional for a url (it is "
            "derived from the link). The file is local only, and an existing file of the "
            "same name is never overwritten. The download cannot run JavaScript, so a "
            "page that renders client-side saves as its shell — the result says so when "
            "that happens, and you should pass that on rather than claim a clean copy."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "http(s) page to download and save verbatim",
                },
                "html": {
                    "type": "string",
                    "description": "A complete HTML document you wrote (alternative to url)",
                },
                "filename": {
                    "type": "string",
                    "description": ("File name without a path, e.g. 'week-summary.html'. "
                                    "Optional when url is given."),
                },
            },
            "required": [],
        },
        fn=save_html,
    )
