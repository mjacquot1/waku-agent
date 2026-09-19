"""search_documents / save_html — a local folder of documents the agent can
read, and one designated place for the HTML it writes back.

Local-first like the rest of waku's tools: the folder is `.waku/downloads/`
unless WAKU_DOCS_DIR points at one you already have. Generated HTML lands in
`html/` by default, or in `jobs/<company>/YYYY-MM-DD/` when save_html is given
a company (a crawl). Nothing is uploaded, served, or opened — the tools
return the path so the user opens the file themselves. search_documents takes
the same subfolder, plus filename_pattern so a JPMorgan search cannot see a
Bank of America file.

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

Plain HTTP is stdlib urllib (same as search.py and catalog.py). It cannot run
JavaScript, so a Workday or Phenom listing arrives as an empty shell. When the
optional [browser] extra is installed, a JavaScript shell is automatically
retried in headless Chromium, which waits for network idle, optionally for a
CSS selector, captures JSON XHR bodies next to the HTML (latest body per
endpoint, with the listing payload tagged role=listing), and still reports a
challenge page as a challenge page. render=true skips the urllib hop. A
career SPA often ignores `?rows=` on the HTML URL; the listing endpoint in
`*.network.json` (start/rows/filters) is the complete filtered JSON, so the
next save_html call should fetch that API URL directly — JSON is pretty-printed
and saved as `.json`, not wrapped in a fake HTML page.

urls=[...] or jobs=[{url, filename, title, req_id, apply_url}, ...] downloads
several pages in one call, because SOUL asks for one tool call per request.
jobs keeps a custom name per page (YYYY-MM-DD-company-title.html);
naming_prefix does the same from the URL slug when the caller does not name
each file. A company crawl also writes
`manifest-<company>-YYYY-MM-DD.json` in that day's folder so later questions
("what did we get from JPMC today?") read the index instead of every HTML
file. Page files are never overwritten; the day's crawl index is updated in
place.

What this does not do, on purpose: stealth plugins, residential proxies, or
CAPTCHA solving. A real Chromium is the fingerprint. Sites that still serve a
bot challenge are reported as blocked, not bypassed.
"""

from __future__ import annotations

import fnmatch
import json
import os
import re
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.parse import urlsplit
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from waku.tools.registry import Tool

DOCS_DIR_ENV = "WAKU_DOCS_DIR"              # the folder itself; default <home>/downloads
HTML_SUBDIR_ENV = "WAKU_DOCS_HTML_SUBDIR"   # designated output folder; default "html"
FETCH_MAX_MB_ENV = "WAKU_FETCH_MAX_MB"      # size ceiling for a fetched page
RENDER_TIMEOUT_ENV = "WAKU_RENDER_TIMEOUT"  # seconds the headless browser may spend

# Some sites answer Python-urllib/3.x with a 403 (see catalog.py for the same
# problem against model catalogs), so ask the way a browser would.
FETCH_UA = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"
FETCH_TIMEOUT = 20
DEFAULT_FETCH_MAX_MB = 25
DEFAULT_RENDER_TIMEOUT = 45
MAX_BATCH = 20
MAX_JSON_HITS = 30
MAX_JSON_BYTES = 1_000_000
BROWSER_INSTALL = "pip install 'waku-agent[browser]' && playwright install chromium"

# What can be read as text. Everything else is counted and named as skipped.
TEXT_SUFFIXES = {".txt", ".md", ".markdown", ".rst", ".html", ".htm", ".xml",
                 ".json", ".yaml", ".yml", ".toml", ".ini", ".cfg",
                 ".csv", ".tsv", ".log", ".py", ".sql"}

MAX_MATCHES = 40        # matching lines returned unless the caller asks for more
MAX_MATCHES_CEILING = 200  # hard cap — a listing JSON can be long; a folder search shouldn't be
MAX_PER_FILE = 5        # so one long file can't crowd out the rest of the folder
MAX_LISTED = 50         # files named when listing the folder
SNIPPET_CHARS = 200
MAX_NAME_STEM = 120     # YYYY-MM-DD-company-title needs more than a short slug
_REQ_ID_TAIL = re.compile(r"(\d{5,})$")
_ISO_DATE = re.compile(r"\d{4}-\d{2}-\d{2}$")


def docs_root(home: Path) -> Path:
    """The downloads folder. Same posture as waku/tools/workspace.py: one env
    var, resolved once, so every caller agrees on which folder is meant."""
    override = os.getenv(DOCS_DIR_ENV, "").strip()
    return (Path(override).expanduser() if override else home / "downloads").resolve()


def html_dir(home: Path) -> Path:
    """The designated subfolder for generated HTML. `.name` keeps the override
    to a single folder, so WAKU_DOCS_HTML_SUBDIR can't aim it up and out."""
    name = Path(os.getenv(HTML_SUBDIR_ENV, "").strip()).name
    return (docs_root(home) / (name or "html")).resolve()


def _pacific_tz():
    """America/Los_Angeles so folder names follow PST/PDT, not UTC.

    Slim images sometimes ship without the IANA database; UTC-8 is PST and
    is the fallback so a missing tzdata file cannot blow up a save."""
    try:
        return ZoneInfo("America/Los_Angeles")
    except ZoneInfoNotFoundError:
        return timezone(timedelta(hours=-8))


def _now() -> datetime:
    return datetime.now(_pacific_tz())


def _crawl_date() -> str:
    """Pacific Time calendar day, the folder name under jobs/<company>/."""
    return _now().strftime("%Y-%m-%d")


def _safe_segment(value: str) -> str:
    """One folder name: alphanumerics, hyphen, underscore. Lowercased so
    JPMorgan and jpmorgan land in the same place."""
    cleaned = "".join(c if (c.isalnum() or c == "_") else "-" for c in (value or "").strip())
    while "--" in cleaned:
        cleaned = cleaned.replace("--", "-")
    return cleaned.strip("-").lower()[:MAX_NAME_STEM]


def _filename_matches(path: Path, root: Path, pattern: str) -> bool:
    """Glob when the pattern has * ? [, otherwise a case-insensitive
    substring of the path relative to the documents root (or the file name).
    Matching the relative path is what makes pattern='jpmorgan' see
    jobs/jpmorgan/date/role.html even when the file itself is named by title."""
    needle = (pattern or "").strip()
    if not needle:
        return True
    try:
        rel = path.relative_to(root).as_posix()
    except ValueError:
        rel = path.name
    name = path.name
    if any(ch in needle for ch in "*?["):
        pat = needle.lower()
        return fnmatch.fnmatch(rel.lower(), pat) or fnmatch.fnmatch(name.lower(), pat)
    lowered = needle.lower()
    return lowered in rel.lower() or lowered in name.lower()


def _resolve_save_dir(home: Path, subfolder: str = "", company: str = "") -> Path | str:
    """Where a page lands. subfolder wins; else jobs/<company>/<date>; else html/.

    Returns a Path under the documents root, or an error sentence the model
    can act on. Directories are created at write time, not here."""
    root = docs_root(home)
    if (subfolder or "").strip():
        folder = (root / subfolder.strip()).resolve()
        if folder != root and root not in folder.parents:
            return f"refused: '{subfolder}' is outside the documents folder ({root})."
        return folder
    raw = (company or "").strip()
    if raw:
        slug = _safe_segment(raw)
        if not slug:
            return f"refused: '{company}' is not a usable company folder name."
        folder = (root / "jobs" / slug / _crawl_date()).resolve()
        if folder != root and root not in folder.parents:
            return f"refused: company path escaped the documents folder ({root})."
        return folder
    return html_dir(home)


def _split_by_readability(folder: Path) -> tuple[list[Path], list[Path]]:
    readable: list[Path] = []
    skipped: list[Path] = []
    for path in sorted(p for p in folder.rglob("*") if p.is_file()):
        if path.name.startswith("."):
            continue
        (readable if path.suffix.lower() in TEXT_SUFFIXES else skipped).append(path)
    return readable, skipped


def _limits_note(skipped: list[Path], unreadable: list[str], capped: list[str],
                 per_file: int = MAX_PER_FILE) -> str:
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
        notes.append(f"only the first {per_file} matches are shown for {shown}")
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
    def search_documents(query: str = "", subfolder: str = "",
                         filename_pattern: str = "",
                         max_results: int = MAX_MATCHES,
                         max_per_file: int = MAX_PER_FILE) -> str:
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
        pattern = (filename_pattern or "").strip()
        if pattern:
            readable = [p for p in readable if _filename_matches(p, root, pattern)]
            skipped = [p for p in skipped if _filename_matches(p, root, pattern)]
            if not readable and not skipped:
                return (f"No documents matching filename_pattern '{pattern}' "
                        f"under {folder}.")
        if not query.strip():
            return _inventory(folder, root, readable, skipped)

        try:
            limit = max(1, min(int(max_results or MAX_MATCHES), MAX_MATCHES_CEILING))
        except (TypeError, ValueError):
            limit = MAX_MATCHES
        try:
            per_file = max(1, min(int(max_per_file or MAX_PER_FILE), limit))
        except (TypeError, ValueError):
            per_file = min(MAX_PER_FILE, limit)
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
            for lineno, line in enumerate(_search_lines(path, text), 1):
                if needle not in line.lower():
                    continue
                if found >= per_file:
                    capped.append(path.name)
                    break
                if len(hits) >= limit:
                    truncated = True
                    break
                hits.append(f"- {path.relative_to(root)}:{lineno}: {line.strip()[:SNIPPET_CHARS]}")
                found += 1
            if truncated:
                break

        scoped = f" matching filename_pattern '{pattern}'" if pattern else ""
        if not hits:
            return (f"No matches for '{query}' in {len(readable)} document(s){scoped} "
                    f"under {folder}."
                    + _limits_note(skipped, unreadable, capped, per_file))
        head = (f"First {len(hits)} match(es) for '{query}' in {folder}{scoped} (more exist):"
                if truncated else f"{len(hits)} match(es) for '{query}' in {folder}{scoped}:")
        return "\n".join([head, *hits]) + _limits_note(skipped, unreadable, capped, per_file)

    return Tool(
        name="search_documents",
        description=(
            "Search the user's local documents folder and read what the files say. Use "
            "whenever the user asks what a document contains, or to look something up in "
            "their files, notes, or saved pages. Pass a distinctive word or phrase as the "
            "query; call it with no query to list which documents are there. Results are "
            "'path:line: text'. Only plain-text formats are searched and the answer names "
            "any file it could not read — repeat that caveat instead of reporting an empty "
            "folder as an empty answer. Pass subfolder (e.g. 'jobs/jpmorgan') to stay "
            "inside one company's crawl, or filename_pattern (e.g. 'jpmorgan' or "
            "'*.network.json') so files from other companies are not opened. Defaults "
            "to five matches per file so one listing cannot hide the rest of the folder; "
            "pass a higher max_per_file (and max_results, up to "
            f"{MAX_MATCHES_CEILING}) when reading a saved job-listing JSON. For a "
            "saved crawl, prefer the day's manifest-<company>-YYYY-MM-DD.json over "
            "grepping every HTML file."
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
                    "description": (
                        "Optional subfolder to narrow the search to, e.g. 'html' or "
                        "'jobs/jpmorgan'. Paths stay inside the documents folder."
                    ),
                },
                "filename_pattern": {
                    "type": "string",
                    "description": (
                        "Only search files whose relative path or name matches. A plain "
                        "string (e.g. 'jpmorgan') is a case-insensitive substring; *, ? "
                        "and [ turn it into a glob against the path or file name."
                    ),
                },
                "max_results": {
                    "type": "integer",
                    "description": (
                        f"Max matching lines to return (default {MAX_MATCHES}, "
                        f"ceiling {MAX_MATCHES_CEILING})"
                    ),
                },
                "max_per_file": {
                    "type": "integer",
                    "description": (
                        f"Max matching lines from one file (default {MAX_PER_FILE}). "
                        "Raise this when searching a listing JSON so every job is visible."
                    ),
                },
            },
            "required": [],
        },
        fn=search_documents,
    )


def _search_lines(path: Path, text: str) -> list[str]:
    """JSON from a listing API is often one long line. Expand it so each field
    is searchable, the way the sidecar files are already written (indent=2).
    Line numbers then refer to the pretty view, not the compact file; that is
    still more useful than a 200-character slice of a 200 KB blob."""
    if path.suffix.lower() != ".json":
        return text.splitlines()
    try:
        obj = json.loads(text)
    except (json.JSONDecodeError, ValueError):
        return text.splitlines()
    if isinstance(obj, (dict, list)):
        return json.dumps(obj, indent=2, ensure_ascii=False, default=str).splitlines()
    return text.splitlines()


def _safe_html_name(filename: str, default_ext: str = ".html") -> str:
    """One flat file name, slugified like messages.py names its outbox
    files. `Path(...).name` drops any directory the model tried to include, so
    the designated folder stays the only destination. `.json` is kept when the
    payload (or the caller) is JSON; everything else is `.html`."""
    stem = Path(filename.strip()).name
    if stem in ("", ".", ".."):
        return ""
    ext = default_ext if default_ext in (".html", ".json") else ".html"
    lower = stem.lower()
    if lower.endswith(".json"):
        ext = ".json"
        stem = stem[:-5]
    elif lower.endswith(".html"):
        ext = ".html"
        stem = stem[:-5]
    elif lower.endswith(".htm"):
        ext = ".html"
        stem = stem[:-4]
    cleaned = "".join(c if (c.isalnum() or c == "_") else "-" for c in stem)
    while "--" in cleaned:
        cleaned = cleaned.replace("--", "-")
    cleaned = cleaned.strip("-")
    return f"{cleaned[:MAX_NAME_STEM]}{ext}" if cleaned else ""


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


def _render_timeout_ms() -> int:
    try:
        seconds = int(os.getenv(RENDER_TIMEOUT_ENV, "") or DEFAULT_RENDER_TIMEOUT)
    except ValueError:
        seconds = DEFAULT_RENDER_TIMEOUT
    return max(5, seconds) * 1000


def _truthy(value) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    return str(value or "").strip().lower() in ("1", "true", "yes", "on")


def playwright_available() -> bool:
    """Import check only — Chromium may still be missing after the extra is in."""
    try:
        import playwright.sync_api  # noqa: F401
        return True
    except ImportError:
        return False


class _FetchError(Exception):
    """Carries a sentence the model can act on, not a stack trace."""


@dataclass
class FetchResult:
    payload: bytes
    status: str
    warning: str = ""
    json_hits: list = field(default_factory=list)
    rendered: bool = False
    content_type: str = ""


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


def _require_http_url(url: str) -> None:
    parts = urlsplit(url.strip())
    if parts.scheme not in ("http", "https"):
        raise _FetchError(f"only http and https are fetched, not '{parts.scheme or url}'")
    if not parts.netloc:
        raise _FetchError("that is not a complete url — it has no host")


def _looks_js_rendered(payload: bytes) -> bool:
    """Cheap check for "the server sent an app, not a page". Worth doing badly:
    a Workday posting saved as a 900-byte script tag is a silent failure the
    user only discovers when they open the file weeks later."""
    head = payload[:4000].lower()
    return (b"enable javascript" in head
            or b"you need to enable" in head
            or (len(payload) < 1024 and b"<script" in head))


def _looks_challenge(payload: bytes) -> bool:
    """Bot-protection interstitial. We report it; we do not try to walk through."""
    head = payload[:8000].lower()
    return any(marker in head for marker in (
        b"cf-browser-verification",
        b"cdn-cgi/challenge",
        b"px-captcha",
        b"_incapsula_",
        b"data-datadome",
        b"just a moment...",
        b"verify you are human",
        b"attention required",
    ))


def _js_shell_warning() -> str:
    if playwright_available():
        return (" Warning: the response is mostly a JavaScript shell. Call save_html "
                "again with render=true to paint the page in a headless browser.")
    return (" Warning: the response is mostly a JavaScript shell, so this file is the "
            "raw HTTP body, not the page a person sees. Career sites that render "
            f"client-side (Workday, Taleo, Phenom) need the [browser] extra: {BROWSER_INSTALL}")


def _challenge_warning() -> str:
    return (" Warning: this looks like a bot-protection page (Cloudflare, Akamai, "
            "PerimeterX, Datadome), not the listing. Waku does not bypass those "
            "challenges — open the URL in your own browser if you need the page.")


def _fetch(url: str) -> FetchResult:
    """GET `url` over plain HTTP. Bytes, deliberately — not decoded text."""
    _require_http_url(url)
    limit = _max_fetch_bytes()
    request = urllib.request.Request(url, headers={
        "User-Agent": FETCH_UA,
        "Accept": "text/html,application/json,*/*",
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
        warning = (f" Warning: the server sent {encoding}-compressed bytes despite being asked "
                   "for none, so the file holds the compressed response, not readable html. "
                   f"Decompress it before reading ({encoding}).")
    elif _looks_challenge(payload):
        warning = _challenge_warning()
    elif _looks_js_rendered(payload) and "json" not in (ctype or "").lower():
        warning = _js_shell_warning()
    return FetchResult(
        payload, f"HTTP {status}, {ctype}, {len(payload)} bytes", warning,
        content_type=ctype or "",
    )


class Renderer:
    """One Chromium for a whole batch, so 20 job links are not 20 cold starts."""

    def __init__(self):
        try:
            from playwright.sync_api import sync_playwright
        except ImportError as exc:
            raise _FetchError(
                f"render needs the [browser] extra. Install it with: {BROWSER_INSTALL}"
            ) from exc
        try:
            self._playwright = sync_playwright().start()
            self._browser = self._playwright.chromium.launch(headless=True)
        except Exception as exc:
            raise _FetchError(_playwright_launch_error(exc)) from exc

    def fetch(self, url: str, wait_for_selector: str = "") -> FetchResult:
        _require_http_url(url)
        timeout = _render_timeout_ms()
        limit = _max_fetch_bytes()
        json_hits: list[dict] = []
        warning_bits: list[str] = []
        context = self._browser.new_context(user_agent=FETCH_UA)
        try:
            page = context.new_page()
            page.on("response", lambda response: _capture_json(response, json_hits))
            try:
                page.goto(url, wait_until="networkidle", timeout=timeout)
            except Exception as exc:
                if not _is_timeout(exc):
                    raise
                # Long-polling boards never go idle. Save the DOM we have.
                warning_bits.append(
                    "networkidle never arrived (the page kept making requests); "
                    "saved the DOM as it stood"
                )
            selector = _css_selector(wait_for_selector)
            if selector:
                try:
                    page.wait_for_selector(selector, timeout=timeout)
                except Exception as exc:
                    if not _is_timeout(exc):
                        raise
                    warning_bits.append(
                        f"selector {selector!r} never appeared before the timeout"
                    )
            html = page.content()
        except Exception as exc:
            raise _FetchError(_playwright_nav_error(exc)) from exc
        finally:
            context.close()

        payload = html.encode("utf-8")
        if len(payload) > limit:
            raise _FetchError(
                f"the rendered page is over the {limit // (1024 * 1024)} MB ceiling. "
                f"Nothing was saved. Raise {FETCH_MAX_MB_ENV} to keep it anyway")
        _tag_listing_hits(json_hits, url)
        warning = ""
        if _looks_challenge(payload):
            warning = _challenge_warning()
        elif _looks_js_rendered(payload):
            warning = (" Warning: even after rendering in a headless browser this still "
                       "looks like a JavaScript shell.")
        if warning_bits:
            warning += " Warning: " + "; ".join(warning_bits) + "."
        extra = f", {len(json_hits)} JSON endpoint(s)" if json_hits else ""
        return FetchResult(
            payload,
            f"rendered in Chromium, {len(payload)} bytes{extra}",
            warning,
            json_hits,
            rendered=True,
            content_type="text/html",
        )

    def close(self) -> None:
        try:
            self._browser.close()
        finally:
            self._playwright.stop()


def _playwright_launch_error(exc: Exception) -> str:
    msg = str(exc)
    if "Executable doesn't exist" in msg or "playwright install" in msg.lower():
        return f"Chromium is not installed. Run: playwright install chromium ({exc})"
    return f"could not start Chromium: {exc}"


def _is_timeout(exc: Exception) -> bool:
    return "Timeout" in exc.__class__.__name__ or "timeout" in str(exc).lower()


def _playwright_nav_error(exc: Exception) -> str:
    msg = str(exc) or exc.__class__.__name__
    if _is_timeout(exc):
        return f"the page was still loading after {DEFAULT_RENDER_TIMEOUT}s ({msg})"
    return msg


def _css_selector(value: str) -> str:
    """A single CSS selector, or empty. Newlines and a 200-char cap keep this
    a wait, not a dumped stylesheet."""
    sel = (value or "").strip()
    if not sel:
        return ""
    if "\n" in sel or "\r" in sel or len(sel) > 200:
        raise _FetchError("wait_for_selector must be a single CSS selector")
    return sel


_LISTING_KEYS = frozenset({
    "jobs", "jobsearch", "jobsearchresults", "jobsearchresult",
    "requisitions", "results", "postings", "positions", "opportunities",
})


def _json_endpoint_key(url: str) -> tuple[str, str]:
    parts = urlsplit(url)
    return f"{parts.scheme}://{parts.netloc}{parts.path}", parts.query


def _listing_score(hit: dict, page_url: str) -> int:
    body = hit.get("body")
    score = 0
    if isinstance(body, list):
        score += 3 if body else 0
    elif isinstance(body, dict):
        keys = {str(k).lower() for k in body}
        if keys & _LISTING_KEYS:
            score += 5
        for value in body.values():
            if isinstance(value, list) and value and isinstance(value[0], dict):
                score += 3
                break
    page_path = urlsplit(page_url).path.rstrip("/")
    endpoint = str(hit.get("endpoint") or "").rstrip("/")
    if page_path and endpoint and (page_path in endpoint or endpoint.endswith(page_path)):
        score += 2
    endpoint_l = endpoint.lower()
    if "jobssearch" in endpoint_l or "jobsearch" in endpoint_l:
        score += 4
    return score


def _tag_listing_hits(hits: list[dict], page_url: str) -> None:
    """Mark the JSON call that looks like the listing so the agent does not
    treat analytics payloads as the job results."""
    if not hits:
        return
    scored = [(i, _listing_score(hit, page_url)) for i, hit in enumerate(hits)]
    best_i, best = max(scored, key=lambda pair: pair[1])
    for i, hit in enumerate(hits):
        hit["role"] = "listing" if i == best_i and best > 0 else "other"


def _capture_json(response, bucket: list[dict]) -> None:
    """Keep XHR/fetch JSON that career sites use instead of server-rendered HTML.

    The same endpoint is often hit several times as filters and pagination
    change the query string. Earlier bodies are dropped; the latest is kept
    and `replaced` counts how many came before. Caps apply per unique
    endpoint, not per raw response.
    """
    ctype = (response.headers.get("content-type") or "").lower()
    if "json" not in ctype:
        return
    try:
        body = response.json()
    except Exception:
        return
    try:
        encoded = json.dumps(body, default=str)
    except (TypeError, ValueError):
        return
    if len(encoded) > MAX_JSON_BYTES:
        encoded = encoded[:MAX_JSON_BYTES] + "…[truncated]"
        body = encoded
    endpoint, query = _json_endpoint_key(response.url)
    record = {
        "url": response.url,
        "endpoint": endpoint,
        "query": query,
        "status": response.status,
        "body": body,
        "replaced": 0,
        "role": "other",
    }
    for i, existing in enumerate(bucket):
        if existing.get("endpoint") == endpoint:
            record["replaced"] = int(existing.get("replaced") or 0) + 1
            bucket[i] = record
            return
    if len(bucket) >= MAX_JSON_HITS:
        return
    bucket.append(record)


def _fetch_rendered(
    url: str,
    renderer: Renderer | None = None,
    wait_for_selector: str = "",
) -> FetchResult:
    """Public so tests can monkeypatch the browser hop without launching Chromium."""
    if renderer is not None:
        return renderer.fetch(url, wait_for_selector=wait_for_selector)
    owned = Renderer()
    try:
        return owned.fetch(url, wait_for_selector=wait_for_selector)
    finally:
        owned.close()


def _obtain(
    url: str,
    render: bool,
    renderer: Renderer | None,
    wait_for_selector: str = "",
) -> FetchResult:
    _require_http_url(url)
    if render:
        if not playwright_available() and renderer is None:
            raise _FetchError(
                f"render=true needs the [browser] extra. Install it with: {BROWSER_INSTALL}"
            )
        return _fetch_rendered(url, renderer, wait_for_selector=wait_for_selector)
    result = _fetch(url)
    if _looks_js_rendered(result.payload) and playwright_available():
        painted = _fetch_rendered(url, renderer, wait_for_selector=wait_for_selector)
        painted.warning = (
            " Plain HTTP was a JavaScript shell, so this was re-fetched in a "
            "headless browser." + painted.warning
        )
        return painted
    return result


def _flatten_items(value) -> list:
    """urls/jobs arrive as a list, a newline string, or a JSON string of either."""
    if value is None or value is False:
        return []
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        if text[:1] in "[{":
            try:
                parsed = json.loads(text)
            except json.JSONDecodeError:
                parsed = None
            if parsed is not None:
                return _flatten_items(parsed)
        chunk = text.replace(",", "\n")
        return [part.strip() for part in chunk.split() if part.strip()]
    if isinstance(value, dict):
        return [value]
    if isinstance(value, list):
        return list(value)
    return [value]


@dataclass
class SaveTarget:
    url: str
    filename: str = ""
    title: str = ""
    req_id: str = ""
    apply_url: str = ""


@dataclass
class WriteOutcome:
    message: str
    dest: Path | None = None


def _as_target(item) -> SaveTarget | None:
    """One download. filename/title/req_id may be empty — filled in later."""
    if isinstance(item, dict):
        url = str(item.get("url") or item.get("href") or "").strip()
        if not url:
            return None
        req = str(
            item.get("req_id") or item.get("reqId") or item.get("requisitionId") or ""
        ).strip()
        apply_url = str(item.get("apply_url") or item.get("applyUrl") or url).strip() or url
        return SaveTarget(
            url=url,
            filename=str(item.get("filename") or item.get("name") or "").strip(),
            title=str(item.get("title") or "").strip(),
            req_id=req,
            apply_url=apply_url,
        )
    text = str(item).strip()
    return SaveTarget(url=text, apply_url=text) if text else None


def _coerce_targets(url: str, urls, jobs) -> list[SaveTarget]:
    items: list = []
    if isinstance(url, str) and url.strip():
        items.append(url.strip())
    items.extend(_flatten_items(urls))
    items.extend(_flatten_items(jobs))
    seen: set[str] = set()
    out: list[SaveTarget] = []
    for item in items:
        parsed = _as_target(item)
        if parsed is None:
            continue
        if parsed.url in seen:
            continue
        seen.add(parsed.url)
        if not parsed.apply_url:
            parsed.apply_url = parsed.url
        out.append(parsed)
    return out


def _name_for(url: str, given: str, naming_prefix: str, single_filename: str,
              batch: bool) -> str:
    if given.strip():
        return given
    if not batch and single_filename.strip():
        return single_filename
    slug = _slug_from_url(url)
    prefix = (naming_prefix or "").strip()
    return f"{prefix}-{slug}" if prefix else slug


def _payload_is_json(payload: bytes, content_type: str = "") -> bool:
    ctype = (content_type or "").lower()
    if "html" in ctype:
        return False
    if "json" in ctype:
        return True
    return payload.lstrip()[:1] in (b"{", b"[")


def _as_saved_payload(payload: bytes, content_type: str = "") -> tuple[bytes, str, bool]:
    """HTML stays byte-for-byte. JSON is pretty-printed so search_documents can
    see one field per line instead of a 200-character slice of a blob."""
    if not _payload_is_json(payload, content_type):
        return payload, ".html", False
    try:
        obj = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError):
        return payload, ".json", True
    pretty = json.dumps(obj, indent=2, ensure_ascii=False, default=str) + "\n"
    return pretty.encode("utf-8"), ".json", True


def _next_free(folder: Path, stem: str, ext: str = ".html") -> Path:
    suffix = ext if ext.startswith(".") else f".{ext}"
    for n in range(2, 100):
        candidate = folder / f"{stem}-{n}{suffix}"
        if not candidate.exists():
            return candidate
    return folder / f"{stem}-{_now():%Y%m%dT%H%M%S}{suffix}"


def _write_payload(
    home: Path,
    payload: bytes,
    filename: str,
    *,
    dest_dir: Path | None = None,
    url: str = "",
    status: str = "",
    warning: str = "",
    json_hits: list | None = None,
    content_type: str = "",
) -> WriteOutcome:
    body, json_ext, is_json = _as_saved_payload(payload, content_type)
    name = _safe_html_name(filename, default_ext=json_ext)
    if not name:
        return WriteOutcome(f"refused: '{filename}' is not a usable file name.")

    root = docs_root(home)
    dest_dir = dest_dir or html_dir(home)
    if dest_dir != root and root not in dest_dir.parents:
        return WriteOutcome(f"refused: path escaped the documents folder ({root}).")
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = (dest_dir / name).resolve()
    if dest_dir not in dest.parents:
        return WriteOutcome(f"refused: path escaped the html folder ({dest_dir}).")

    kept = ""
    if dest.exists():
        try:
            unchanged = dest.read_bytes() == body
        except OSError:
            unchanged = False
        if unchanged:
            return WriteOutcome(
                f"'{name}' is already saved at {dest}, unchanged. "
                "Nothing rewritten." + warning,
                dest,
            )
        dest = _next_free(dest_dir, Path(name).stem, Path(name).suffix or json_ext)
        kept = f" '{name}' already existed with different content and was left untouched."

    dest.write_bytes(body)
    sidecar = ""
    if json_hits:
        network = dest.with_name(dest.stem + ".network.json")
        network.write_text(json.dumps(json_hits, indent=2, default=str) + "\n", encoding="utf-8")
        sidecar = f" {len(json_hits)} JSON endpoint(s) saved beside it at {network.name}."
        listing = next((h for h in json_hits if h.get("role") == "listing"), None)
        if listing:
            listing_url = listing.get("url") or listing.get("endpoint")
            sidecar += (
                f" Listing data is the object with role=listing ({listing.get('endpoint')})."
                f" To keep the complete filtered JSON, call save_html on that listing URL"
                f" ({listing_url}) — no render needed; the HTML search page often ignores"
                f" rows= and filters= until the SPA has painted."
            )
    if url:
        if status.startswith("rendered"):
            saved = f"Rendered {url} ({status}) and saved it to {dest}."
        elif is_json:
            saved = f"Fetched {url} ({status}) and saved it as JSON to {dest}."
        else:
            saved = f"Fetched {url} ({status}) and saved it byte-for-byte to {dest}."
    else:
        saved = f"Saved {len(body)} bytes of HTML to {dest}."
    return WriteOutcome(
        saved + kept + sidecar + warning
        + " It is a local file — nothing was published or served; open it in a "
          "browser to view it.",
        dest,
    )


def _infer_req_id(filename: str, url: str) -> str:
    for text in (filename, urlsplit(url).path.rstrip("/")):
        stem = Path(text).stem if text else ""
        match = _REQ_ID_TAIL.search(stem)
        if match:
            return match.group(1)
    return ""


def _title_from_filename(stem: str, company_slug: str, req_id: str) -> str:
    parts = stem.split("-")
    if (len(parts) >= 3 and len(parts[0]) == 4 and parts[0].isdigit()
            and parts[1].isdigit() and parts[2].isdigit()):
        parts = parts[3:]
    text = "-".join(parts)
    if company_slug and text.lower().startswith(company_slug + "-"):
        text = text[len(company_slug) + 1:]
    if req_id and text.endswith("-" + req_id):
        text = text[: -(len(req_id) + 1)]
    return text.replace("-", " ").strip() or stem


def _role_record(target: SaveTarget, dest: Path, root: Path, company_slug: str) -> dict:
    try:
        rel = dest.relative_to(root).as_posix()
    except ValueError:
        rel = dest.name
    req_id = target.req_id or _infer_req_id(dest.name, target.url)
    title = target.title or _title_from_filename(dest.stem, company_slug, req_id)
    return {
        "title": title,
        "req_id": req_id,
        "file_path": rel,
        "apply_url": target.apply_url or target.url,
    }


def _write_crawl_manifest(
    dest_dir: Path,
    root: Path,
    company: str,
    roles: list[dict],
) -> str:
    """Rewrite the day's crawl index in place. Page files stay immutable;
    this file is the list of what was kept, so a second batch the same day
    must update it rather than sit beside it as manifest-2."""
    slug = _safe_segment(company)
    date = dest_dir.name if _ISO_DATE.fullmatch(dest_dir.name) else _crawl_date()
    path = dest_dir / f"manifest-{slug}-{date}.json"
    existing: list[dict] = []
    if path.exists():
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(payload, dict) and isinstance(payload.get("roles"), list):
                existing = [r for r in payload["roles"] if isinstance(r, dict)]
        except (OSError, json.JSONDecodeError, ValueError):
            existing = []
    by_key: dict[str, dict] = {}
    order: list[str] = []
    for role in existing + roles:
        key = str(role.get("apply_url") or role.get("file_path") or "")
        if not key:
            continue
        if key not in by_key:
            order.append(key)
        by_key[key] = role
    merged = [by_key[key] for key in order]
    doc = {
        "company": company.strip(),
        "date_crawled": date,
        "total_roles_found": len(merged),
        "roles": merged,
    }
    dest_dir.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    try:
        rel = path.relative_to(root).as_posix()
    except ValueError:
        rel = str(path)
    return (f" Crawl index written to {path} ({len(merged)} roles). "
            f"Answer questions about this crawl from {rel}, not by re-reading every HTML file.")


def make_save_html_tool(home: Path) -> Tool:
    def save_html(
        filename: str = "",
        html: str = "",
        url: str = "",
        urls=None,
        jobs=None,
        render=False,
        wait_for_selector: str = "",
        naming_prefix: str = "",
        company: str = "",
        subfolder: str = "",
    ) -> str:
        targets = _coerce_targets(url, urls, jobs)
        if targets and html:
            return ("save_html takes either url(s)/jobs or html, not both — url to download "
                    "a live page, html to save a document you wrote. Please call it again "
                    "with one.")
        if not targets and not html:
            return ("save_html needs a url, urls, jobs, or the html content to write. "
                    "Please call it again with one of them.")

        dest_dir = _resolve_save_dir(home, subfolder, company)
        if isinstance(dest_dir, str):
            return dest_dir
        root = docs_root(home)
        slug = _safe_segment(company)

        if html:
            payload = (html if html.endswith("\n") else html + "\n").encode("utf-8")
            return _write_payload(home, payload, filename, dest_dir=dest_dir).message

        try:
            selector = _css_selector(wait_for_selector)
        except _FetchError as exc:
            return f"Nothing saved — {exc}."

        render_wanted = _truthy(render)
        extra = targets[MAX_BATCH:]
        targets = targets[:MAX_BATCH]
        if render_wanted and not playwright_available():
            return (f"Nothing saved — render=true needs the [browser] extra. "
                    f"Install it with: {BROWSER_INSTALL}")

        # Tests monkeypatch _fetch_rendered; production opens Chromium inside
        # that function. Sharing one browser across a batch is a later win.
        batch = len(targets) > 1
        lines: list[str] = []
        saved_roles: list[tuple[SaveTarget, Path]] = []
        for target in targets:
            try:
                result = _obtain(
                    target.url, render_wanted, renderer=None,
                    wait_for_selector=selector,
                )
            except _FetchError as exc:
                lines.append(f"Nothing saved — could not fetch {target.url}: {exc}.")
                continue
            name = _name_for(target.url, target.filename, naming_prefix, filename, batch)
            outcome = _write_payload(
                home, result.payload, name, dest_dir=dest_dir, url=target.url,
                status=result.status, warning=result.warning,
                json_hits=result.json_hits, content_type=result.content_type,
            )
            lines.append(outcome.message)
            if outcome.dest is not None:
                saved_roles.append((target, outcome.dest))

        if extra:
            lines.append(
                f"Skipped {len(extra)} further URL(s) — the ceiling is {MAX_BATCH} per call. "
                "Call save_html again on a later turn with the remaining URLs."
            )

        manifest_note = ""
        if slug and saved_roles:
            roles = [_role_record(target, dest, root, slug)
                     for target, dest in saved_roles]
            manifest_note = _write_crawl_manifest(dest_dir, root, company.strip(), roles)

        if len(lines) == 1:
            return lines[0] + manifest_note
        saved_n = sum(
            1 for ln in lines
            if not ln.startswith(("Nothing saved", "Skipped", "refused:"))
        )
        return (f"Saved {saved_n} of {len(targets)} page(s):\n"
                + "\n".join(f"- {ln}" for ln in lines)
                + manifest_note)

    return Tool(
        name="save_html",
        description=(
            "Save an HTML page (or a JSON listing) into the user's documents folder, "
            "either by downloading a url or by writing html you composed. Default dest "
            "is the html/ subfolder. Pass company (e.g. 'jpmorgan') to save a crawl "
            "under jobs/<company>/YYYY-MM-DD/ (Pacific Time calendar day) and write "
            "manifest-<company>-YYYY-MM-DD.json "
            "listing {title, req_id, file_path, apply_url} for each saved page — read "
            "that manifest later instead of grepping every HTML file. Pass subfolder to "
            "pick the destination folder yourself (still inside the documents folder). Pass url, "
            "urls, or jobs, or html, not both. "
            "PREFER url whenever the user points at a web page (a job posting, an "
            "article, a listing): waku downloads it and writes it to disk, so no part of "
            "the page has to pass through your context — do NOT fetch a page into your "
            "context and echo it back as html, that truncates long pages. "
            "Pass jobs (an array of {url, filename, title, req_id, apply_url} objects) "
            "to download many named pages in ONE call — filename is kept per page, e.g. "
            "2026-09-18-bank-of-america-software-engineer-iii.html. Pass urls (strings) "
            "plus naming_prefix for the same batch when you do not have a name per "
            "link; each file becomes {naming_prefix}-{url-slug}.html. Do not call this "
            "tool once per link. "
            "Pass render=true for a JavaScript career site (Workday, Taleo, Phenom) "
            "so a headless Chromium paints the page, waits until the network is idle, "
            "and captures JSON API responses beside the HTML. Pass wait_for_selector "
            "(a CSS selector such as '.search-result-item' or '.job-search-result') "
            "when you know the card that means the FILTERED listing has loaded — "
            "without it the snapshot may be the default page, before rows= and "
            "filters= apply. After a render, the *.network.json file names the listing "
            "API (role=listing). Fetch THAT url with save_html (no render) to keep the "
            "complete filtered JSON; the HTML search URL often ignores query "
            "parameters. JSON responses are pretty-printed and saved as .json. "
            "Without the [browser] extra, render=true "
            f"explains how to install it ({BROWSER_INSTALL}). When that extra is already "
            "installed, a JavaScript shell is retried in the browser automatically. "
            "Page files are local only and never overwritten. The day's crawl manifest "
            "is updated in place. Bot-protection pages are reported as blocked, not bypassed."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "http(s) page or listing API to download and save",
                },
                "urls": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": ("Several http(s) pages to download in one call. Use this "
                                    "instead of calling save_html once per link. For a "
                                    "custom name per page, pass jobs instead."),
                },
                "jobs": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "url": {"type": "string"},
                            "filename": {"type": "string"},
                            "title": {"type": "string"},
                            "req_id": {"type": "string"},
                            "apply_url": {"type": "string"},
                        },
                        "required": ["url"],
                    },
                    "description": (
                        "Named batch: each object is {url, filename, title, req_id, "
                        "apply_url}. filename is kept, so a crawl can write "
                        "YYYY-MM-DD-company-title.html per posting. title/req_id/"
                        "apply_url fill the day's crawl manifest when company is set."
                    ),
                },
                "html": {
                    "type": "string",
                    "description": "A complete HTML document you wrote (alternative to url)",
                },
                "filename": {
                    "type": "string",
                    "description": ("File name without a path, e.g. 'week-summary.html'. "
                                    "Optional when url is given. For a batch, prefer jobs "
                                    "or naming_prefix — a single filename is only used "
                                    "when there is one URL."),
                },
                "naming_prefix": {
                    "type": "string",
                    "description": (
                        "Prepended to each URL slug in a batch, e.g. "
                        "'2026-09-18-bank-of-america' → "
                        "2026-09-18-bank-of-america-software-engineer-iii.html. "
                        "Ignored when jobs already supplies filename."
                    ),
                },
                "company": {
                    "type": "string",
                    "description": (
                        "Crawl this employer into jobs/<company>/YYYY-MM-DD/ and write "
                        "manifest-<company>-YYYY-MM-DD.json there. YYYY-MM-DD is today's "
                        "Pacific Time calendar day. e.g. 'jpmorgan'."
                    ),
                },
                "subfolder": {
                    "type": "string",
                    "description": (
                        "Destination under the documents folder, e.g. "
                        "'jobs/jpmorgan/2026-09-18'. Overrides the company default. "
                        "The filename itself still cannot contain a path."
                    ),
                },
                "render": {
                    "type": "boolean",
                    "description": ("Paint the page in a headless browser so JavaScript runs. "
                                    "Needs the [browser] extra."),
                },
                "wait_for_selector": {
                    "type": "string",
                    "description": (
                        "CSS selector to wait for before saving, e.g. '.search-result-item' "
                        "or '.job-title'. Used with render=true so the filtered cards "
                        "are on screen, not the SPA's default results."
                    ),
                },
            },
            "required": [],
        },
        fn=save_html,
    )

