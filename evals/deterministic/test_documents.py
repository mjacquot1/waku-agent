"""DETERMINISTIC EVAL — the documents folder: honest reads, non-destructive writes.

Five guarantees are worth pinning in CI, and none of them is the happy path:

1. **search_documents never launders a skip into an empty answer.** A folder
   with one .pdf and one .md must report the .pdf as unsearched, or "nothing
   about the invoice" means both "it isn't there" and "I couldn't open it".
2. **save_html never overwrites.** The model re-calls tools; if the second call
   with the same filename clobbered the first file, a generated report the user
   already opened would vanish with no trace that it ever existed.
3. **A fetched page is byte-for-byte.** The reason to download in the harness
   instead of through the prompt is that the file is the server's exact
   response. "Saved the posting" is a false claim if the bytes were re-encoded
   or cut, so the fetch tests assert on bytes off a real socket — a loopback
   HTTP server, not a patched urlopen — including the redirect and size paths
   where a wrong answer means writing something that only looks like a copy.
4. **A JavaScript shell is not silently a copy of the page.** Without the
   [browser] extra the tool says so and names the extra. With it, a patched
   renderer (never a real Chromium in CI) must be what lands on disk, and a
   batch of URLs must be one call, because SOUL forbids calling the tool once
   per link. A batch keeps a custom filename per URL (jobs=[{url, filename}])
   or a shared naming_prefix; JSON listing APIs are saved as pretty .json, not
   wrapped in HTML.
5. **search_documents does not hide a listing behind five lines.** The per-file
   cap stays five by default so one file cannot crowd the folder, but
   max_per_file (and max_results up to the ceiling) must be honoured when the
   caller is reading a saved job-listing JSON.
6. **A company crawl stays inside its own folder.** save_html(company=) writes
   jobs/<company>/YYYY-MM-DD/ plus a day's manifest; search_documents
   subfolder and filename_pattern must not return a sibling company's files.
   The day's crawl manifest is rewritten; page files are still never overwritten.

Path containment is checked the way test_gh_tool.py checks argv: assert on what
WOULD happen (the file that does/doesn't exist on disk), not on the prose.
"""

from __future__ import annotations

import json
import threading
from datetime import UTC, datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace

import pytest

from evals.helpers import ScriptedClient, make_waku, response, text_block, tool_block
from waku.tools import build_registry, documents


@pytest.fixture
def docs(tmp_path, monkeypatch):
    """A documents folder at a known path, pointed at by WAKU_DOCS_DIR."""
    folder = tmp_path / "docs"
    folder.mkdir()
    monkeypatch.setenv(documents.DOCS_DIR_ENV, str(folder))
    monkeypatch.delenv(documents.HTML_SUBDIR_ENV, raising=False)
    return folder


@pytest.fixture
def search(tmp_path, docs):
    return documents.make_search_tool(tmp_path / "home").fn


@pytest.fixture
def save(tmp_path, docs, monkeypatch):
    """Plain HTTP only. Render tests re-enable the browser hop themselves, so a
    machine with Playwright installed cannot turn a JS-shell case into a real
    Chromium launch against loopback."""
    monkeypatch.setattr(documents, "playwright_available", lambda: False)
    return documents.make_save_html_tool(tmp_path / "home").fn


def test_docs_root_defaults_to_downloads_inside_home(tmp_path, monkeypatch):
    monkeypatch.delenv(documents.DOCS_DIR_ENV, raising=False)
    home = tmp_path / "home"
    assert documents.docs_root(home) == (home / "downloads").resolve()


@pytest.fixture
def site():
    """A real HTTP server on loopback.

    Byte-for-byte is a claim about what comes off a socket, so a patched
    urlopen could not prove it: the bug would live in the reading, the
    redirect handling, or the header, which a fake never exercises. Routes are
    (status, content-type, body) plus optional extra response headers; for a
    3xx the body is the Location. `seen` holds the last request's headers.
    """
    routes: dict[str, tuple] = {}
    seen: dict[str, str] = {}

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            seen.update(self.headers)
            code, ctype, body, *rest = routes.get(self.path, (404, "text/plain", b"not found"))
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            for key, value in (rest[0] if rest else {}).items():
                self.send_header(key, value)
            if 300 <= code < 400:
                self.send_header("Location", body.decode())
                body = b""
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass        # a passing test should be silent

    class Quiet(ThreadingHTTPServer):
        def handle_error(self, request, client_address):
            pass        # the size-ceiling test hangs up mid-body, by design

    httpd = Quiet(("127.0.0.1", 0), Handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield SimpleNamespace(routes=routes, seen=seen, base=f"http://127.0.0.1:{httpd.server_port}")
    httpd.shutdown()
    httpd.server_close()


# ---------- search_documents


def test_search_reports_file_and_line(search, docs):
    (docs / "notes.md").write_text("intro\nthe invoice total was 420 EUR\n", encoding="utf-8")
    (docs / "other.txt").write_text("nothing relevant\n", encoding="utf-8")

    out = search(query="invoice")
    assert "notes.md:2" in out
    assert "420 EUR" in out
    assert "other.txt" not in out


def test_search_is_case_insensitive_and_walks_subfolders(search, docs):
    nested = docs / "2026" / "q3"
    nested.mkdir(parents=True)
    (nested / "report.md").write_text("Revenue Target met\n", encoding="utf-8")

    out = search(query="revenue target")
    assert "2026/q3/report.md:1" in out


def test_search_names_the_files_it_could_not_read(search, docs):
    """The honesty guarantee: a skipped format is stated, not swallowed."""
    (docs / "readme.md").write_text("no numbers here\n", encoding="utf-8")
    (docs / "scan.pdf").write_bytes(b"%PDF-1.4 binary")

    out = search(query="invoice")
    assert "No matches" in out
    assert "1 file(s) not searched" in out and ".pdf" in out


def test_search_with_no_query_lists_the_folder(search, docs):
    (docs / "a.md").write_text("x\n", encoding="utf-8")
    (docs / "b.txt").write_text("y\n", encoding="utf-8")

    out = search()
    assert "2 searchable document(s)" in out
    assert "a.md" in out and "b.txt" in out


def test_search_refuses_a_subfolder_outside_the_root(search, docs, tmp_path):
    """'../' is a folder name to a model, not an attack — but the file above the
    root must not come back either way."""
    (tmp_path / "secret.md").write_text("sk-live-do-not-read\n", encoding="utf-8")

    out = search(query="sk-live", subfolder="..")
    assert out.startswith("refused:")
    assert "secret.md" not in out and "do-not-read" not in out


def test_search_explains_a_missing_folder_instead_of_crashing(tmp_path, monkeypatch):
    monkeypatch.setenv(documents.DOCS_DIR_ENV, str(tmp_path / "nope"))
    out = documents.make_search_tool(tmp_path / "home").fn(query="anything")
    assert "No documents folder yet" in out and documents.DOCS_DIR_ENV in out


def test_search_says_when_one_file_had_more_matches_than_it_showed(search, docs):
    """Five hits out of fifty is a sample, not an answer. Caught by this test:
    the per-file cap was applied silently, so a partial read looked complete."""
    (docs / "big.md").write_text("hit\n" * 50, encoding="utf-8")

    out = search(query="hit")
    shown = [ln for ln in out.splitlines() if ln.startswith("- ")]
    assert len(shown) == documents.MAX_PER_FILE
    assert f"only the first {documents.MAX_PER_FILE} matches are shown for big.md" in out


def test_search_caps_total_results_and_says_so(search, docs):
    for n in range(12):
        (docs / f"f{n}.md").write_text("hit\n" * 5, encoding="utf-8")

    out = search(query="hit")
    shown = [ln for ln in out.splitlines() if ln.startswith("- ")]
    assert len(shown) == documents.MAX_MATCHES
    assert "more exist" in out


def test_search_honours_an_explicit_max_results(search, docs):
    (docs / "a.md").write_text("hit\n" * 5, encoding="utf-8")
    (docs / "b.md").write_text("hit\n" * 5, encoding="utf-8")

    out = search(query="hit", max_results=3)
    assert len([ln for ln in out.splitlines() if ln.startswith("- ")]) == 3


def test_search_honours_a_higher_max_per_file(search, docs):
    """Five hits is a sample. A listing JSON is the whole answer, so the caller
    must be able to raise the per-file cap."""
    (docs / "listing.json").write_text("hit\n" * 40, encoding="utf-8")

    out = search(query="hit", max_per_file=25, max_results=25)
    shown = [ln for ln in out.splitlines() if ln.startswith("- ")]
    assert len(shown) == 25
    assert "only the first 25 matches are shown for listing.json" in out


def test_search_expands_compact_json_so_each_job_is_a_line(search, docs):
    """A listing API is often one 200 KB line. Without expanding it, the
    snippet is 200 characters of a blob and the per-file cap is one hit."""
    payload = {"jobs": [{"title": f"Engineer {n}", "id": str(n)} for n in range(12)]}
    (docs / "html").mkdir()
    (docs / "html" / "jobssearchservlet.json").write_text(
        json.dumps(payload), encoding="utf-8")

    out = search(query="Engineer", max_per_file=20, max_results=20)
    shown = [ln for ln in out.splitlines() if ln.startswith("- ")]
    assert len(shown) == 12
    assert "Engineer 11" in out


def test_search_max_results_can_pass_the_default_but_not_the_ceiling(search, docs):
    (docs / "big.md").write_text("hit\n" * (documents.MAX_MATCHES_CEILING + 10), encoding="utf-8")
    out = search(query="hit", max_results=60, max_per_file=60)
    assert len([ln for ln in out.splitlines() if ln.startswith("- ")]) == 60

    capped = search(query="hit", max_results=9999, max_per_file=9999)
    assert len([ln for ln in capped.splitlines() if ln.startswith("- ")]) == documents.MAX_MATCHES_CEILING


def test_search_filename_pattern_ignores_other_companies(search, docs):
    """A JPMorgan grep must not open a Bank of America file, even when both
    sit under the same documents root."""
    jpmc = docs / "jobs" / "jpmorgan" / "2026-09-18"
    bofa = docs / "jobs" / "bank-of-america" / "2026-09-18"
    jpmc.mkdir(parents=True)
    bofa.mkdir(parents=True)
    (jpmc / "java.html").write_text("JPMC Java engineer\n", encoding="utf-8")
    (bofa / "java.html").write_text("BofA Java engineer\n", encoding="utf-8")

    out = search(query="Java", filename_pattern="jpmorgan")
    assert "jobs/jpmorgan" in out
    assert "JPMC Java" in out
    assert "bank-of-america" not in out
    assert "BofA" not in out


def test_search_filename_pattern_glob_matches_name_only(search, docs):
    (docs / "html").mkdir()
    (docs / "html" / "jpmorgan.network.json").write_text('{"ok": true}\n', encoding="utf-8")
    (docs / "html" / "jpmorgan.html").write_text("<html>page</html>\n", encoding="utf-8")

    listed = search(filename_pattern="*.network.json")
    assert "jpmorgan.network.json" in listed
    assert "jpmorgan.html" not in listed


def test_search_subfolder_cannot_see_a_sibling_company(search, docs):
    jpmc = docs / "jobs" / "jpmorgan" / "2026-09-18"
    bofa = docs / "jobs" / "bank-of-america" / "2026-09-18"
    jpmc.mkdir(parents=True)
    bofa.mkdir(parents=True)
    (jpmc / "role.html").write_text("secret-jpmc-token\n", encoding="utf-8")
    (bofa / "role.html").write_text("secret-bofa-token\n", encoding="utf-8")

    out = search(query="secret", subfolder="jobs/jpmorgan")
    assert "secret-jpmc-token" in out
    assert "secret-bofa-token" not in out
    assert "bank-of-america" not in out


def test_search_filename_pattern_with_no_hits_says_so(search, docs):
    (docs / "notes.md").write_text("hello\n", encoding="utf-8")
    out = search(query="hello", filename_pattern="jpmorgan")
    assert "No documents matching filename_pattern 'jpmorgan'" in out
    assert "notes.md" not in out


# ---------- save_html


def test_save_html_lands_in_the_designated_subfolder(save, docs):
    out = save(filename="summary.html", html="<html><body>hi</body></html>")

    written = docs / "html" / "summary.html"
    assert written.exists()
    assert written.read_text(encoding="utf-8").endswith("\n")
    assert str(written) in out
    assert "nothing was published" in out


def test_save_html_same_content_twice_is_a_no_op(save, docs):
    save(filename="report.html", html="<html>same</html>")
    out = save(filename="report.html", html="<html>same</html>")

    assert "unchanged" in out
    assert [p.name for p in (docs / "html").iterdir()] == ["report.html"]


def test_save_html_never_overwrites_different_content(save, docs):
    save(filename="report.html", html="<html>first</html>")
    out = save(filename="report.html", html="<html>second</html>")

    html_folder = docs / "html"
    assert "<html>first</html>" in (html_folder / "report.html").read_text(encoding="utf-8")
    assert "<html>second</html>" in (html_folder / "report-2.html").read_text(encoding="utf-8")
    assert "left untouched" in out


def test_save_html_strips_any_path_from_the_filename(save, docs, tmp_path):
    save(filename="../../escaped.html", html="<html>x</html>")

    assert not (tmp_path / "escaped.html").exists()
    assert not (docs / "escaped.html").exists()
    assert (docs / "html" / "escaped.html").exists()


def test_save_html_adds_the_extension_and_refuses_an_empty_name(save, docs):
    save(filename="week 42 digest", html="<html>x</html>")
    assert (docs / "html" / "week-42-digest.html").exists()

    assert save(filename="...", html="<html>x</html>").startswith("refused:")
    assert "needs a url" in save(filename="ok.html")


def test_save_html_refuses_url_and_html_together(save, docs):
    out = save(filename="a.html", html="<html>x</html>", url="http://127.0.0.1:1/x")
    assert "not both" in out
    assert not (docs / "html").exists()


def test_html_subdir_override_cannot_climb_out(tmp_path, docs, monkeypatch):
    monkeypatch.setenv(documents.HTML_SUBDIR_ENV, "../../elsewhere")
    assert documents.html_dir(tmp_path / "home").parent == docs.resolve()


# ---------- save_html fetching a url


PAGE = b"<html><head><title>Job</title></head><body>\r\nSoftware Engineer III \xff\r\n</body></html>"


def test_fetch_saves_the_exact_bytes(save, docs, site):
    """The whole reason to download in the harness: CRLFs, a byte that is not
    valid UTF-8, and no trailing newline all survive to disk untouched."""
    site.routes["/job"] = (200, "text/html; charset=iso-8859-1", PAGE)

    save(url=f"{site.base}/job", filename="job.html")

    assert (docs / "html" / "job.html").read_bytes() == PAGE


def test_fetch_reports_the_status_and_the_path(save, docs, site):
    site.routes["/job"] = (200, "text/html", PAGE)

    out = save(url=f"{site.base}/job", filename="job.html")

    assert "HTTP 200" in out and "text/html" in out
    assert str(docs / "html" / "job.html") in out
    assert "byte-for-byte" in out


def test_fetch_derives_the_filename_from_the_url(save, docs, site):
    site.routes["/en-us/job-detail/software-engineer-iii-26032981"] = (200, "text/html", PAGE)

    save(url=f"{site.base}/en-us/job-detail/software-engineer-iii-26032981")

    assert (docs / "html" / "software-engineer-iii-26032981.html").read_bytes() == PAGE


def test_fetch_of_a_missing_page_saves_nothing(save, docs, site):
    out = save(url=f"{site.base}/gone", filename="gone.html")

    assert "Nothing saved" in out and "404" in out
    assert not (docs / "html" / "gone.html").exists()


def test_fetch_refuses_a_non_http_scheme(save, docs, tmp_path):
    """urllib would happily open file:// — the tool must not."""
    secret = tmp_path / "secret.txt"
    secret.write_text("sk-live-do-not-read\n", encoding="utf-8")

    out = save(url=f"file://{secret}", filename="stolen.html")

    assert "only http and https" in out
    assert not (docs / "html" / "stolen.html").exists()


def test_fetch_refuses_a_redirect_to_a_local_file(save, docs, site, tmp_path):
    """The scheme check on the model's url says nothing about where the SERVER
    sends us next. urllib blocks this hop itself; the assertion is on the
    outcome, so it holds whichever layer does the refusing."""
    secret = tmp_path / "secret.txt"
    secret.write_text("sk-live-do-not-read\n", encoding="utf-8")
    site.routes["/bounce"] = (302, "text/html", f"file://{secret}".encode())

    out = save(url=f"{site.base}/bounce", filename="bounced.html")

    assert "Nothing saved" in out
    assert not (docs / "html" / "bounced.html").exists()


def test_fetch_refuses_a_redirect_to_ftp(save, docs, site):
    """The hop urllib would take and this tool won't: its built-in check allows
    ftp://, which is neither of the two schemes save_html claims to fetch."""
    site.routes["/bounce"] = (302, "text/html", b"ftp://example.invalid/payload")

    out = save(url=f"{site.base}/bounce", filename="bounced.html")

    assert "Nothing saved" in out and "refused a redirect" in out
    assert not (docs / "html" / "bounced.html").exists()


def test_fetch_still_follows_an_ordinary_redirect(save, docs, site):
    """The guard above must not have broken the normal case — job boards
    redirect constantly."""
    site.routes["/apply"] = (302, "text/html", f"{site.base}/job".encode())
    site.routes["/job"] = (200, "text/html", PAGE)

    save(url=f"{site.base}/apply", filename="job.html")

    assert (docs / "html" / "job.html").read_bytes() == PAGE


def test_fetch_refuses_a_page_over_the_size_ceiling(save, docs, site, monkeypatch):
    """Half a page is not a copy, so an oversized fetch writes nothing at all
    rather than a truncated file that reads as saved."""
    monkeypatch.setenv(documents.FETCH_MAX_MB_ENV, "1")
    site.routes["/huge"] = (200, "text/html", b"x" * (2 * 1024 * 1024))

    out = save(url=f"{site.base}/huge", filename="huge.html")

    assert "Nothing saved" in out and "1 MB ceiling" in out
    assert not (docs / "html" / "huge.html").exists()


def test_fetching_the_same_page_twice_is_a_no_op(save, docs, site):
    site.routes["/job"] = (200, "text/html", PAGE)
    save(url=f"{site.base}/job", filename="job.html")

    out = save(url=f"{site.base}/job", filename="job.html")

    assert "unchanged" in out
    assert [p.name for p in (docs / "html").iterdir()] == ["job.html"]


def test_fetch_asks_for_uncompressed_bytes_and_flags_it_if_refused(save, docs, site):
    """urllib does not decompress. A gzip body written to disk under an .html
    name is byte-for-byte correct and completely unreadable, so say so."""
    body = b"\x1f\x8b" + b"compressed-bytes" * 60
    site.routes["/gz"] = (200, "text/html", body, {"Content-Encoding": "gzip"})

    out = save(url=f"{site.base}/gz", filename="gz.html")

    assert site.seen.get("Accept-Encoding") == "identity"
    assert "gzip-compressed" in out and "not readable html" in out
    assert (docs / "html" / "gz.html").read_bytes() == body


def test_fetch_warns_when_the_page_is_a_javascript_shell(save, docs, site):
    """A Workday posting saves as a 200-byte script tag. The file is still
    written — it IS the response — but the answer must not read as a clean copy."""
    shell = (b"<html><body><script src='/app.js'></script>"
             b"<noscript>Please enable JavaScript</noscript></body></html>")
    site.routes["/workday"] = (200, "text/html", shell)

    out = save(url=f"{site.base}/workday", filename="workday.html")

    assert "JavaScript shell" in out and "[browser] extra" in out
    assert (docs / "html" / "workday.html").exists()


# ---------- render (headless browser) and batch


PAINTED = b"<html><body><h1>Software Engineer III</h1><p>posted today</p></body></html>"
SHELL = (b"<html><body><script src='/app.js'></script>"
         b"<noscript>Please enable JavaScript</noscript></body></html>")


def _fake_renderer(payload=PAINTED, json_hits=None):
    def fake(url, renderer=None, wait_for_selector=""):
        return documents.FetchResult(
            payload, f"rendered in Chromium, {len(payload)} bytes",
            json_hits=list(json_hits or []), rendered=True,
        )
    return fake


def test_render_without_playwright_saves_nothing(save, docs, site):
    site.routes["/job"] = (200, "text/html", PAGE)
    out = save(url=f"{site.base}/job", filename="job.html", render=True)
    assert "Nothing saved" in out and "[browser] extra" in out
    assert not (docs / "html" / "job.html").exists()


def test_render_writes_the_painted_page_not_the_http_shell(tmp_path, docs, site, monkeypatch):
    """The whole reason to launch Chromium: disk must hold the DOM, not the
    urllib shell. CI never starts a real browser — the hop is patched."""
    monkeypatch.setattr(documents, "playwright_available", lambda: True)
    monkeypatch.setattr(documents, "_fetch_rendered", _fake_renderer())
    site.routes["/job"] = (200, "text/html", SHELL)
    save = documents.make_save_html_tool(tmp_path / "home").fn

    out = save(url=f"{site.base}/job", filename="job.html", render=True)

    assert (docs / "html" / "job.html").read_bytes() == PAINTED
    assert "Rendered" in out and str(docs / "html" / "job.html") in out


def test_js_shell_auto_upgrades_when_playwright_is_installed(tmp_path, docs, site, monkeypatch):
    """One tool call per request: the model should not have to retry with
    render=true after a shell. If the extra is in, the retry is the harness."""
    monkeypatch.setattr(documents, "playwright_available", lambda: True)
    monkeypatch.setattr(documents, "_fetch_rendered", _fake_renderer())
    site.routes["/job"] = (200, "text/html", SHELL)
    save = documents.make_save_html_tool(tmp_path / "home").fn

    out = save(url=f"{site.base}/job", filename="job.html")

    assert (docs / "html" / "job.html").read_bytes() == PAINTED
    assert "re-fetched in a headless browser" in out


def test_render_writes_captured_json_beside_the_html(tmp_path, docs, monkeypatch):
    hits = [{"url": "https://jobs.example/api/job/1",
             "endpoint": "https://jobs.example/api/job/1",
             "query": "", "status": 200, "replaced": 0, "role": "listing",
             "body": {"reqId": "26033994", "postedDate": "2026-09-17"}}]
    monkeypatch.setattr(documents, "playwright_available", lambda: True)
    monkeypatch.setattr(documents, "_fetch_rendered", _fake_renderer(json_hits=hits))
    save = documents.make_save_html_tool(tmp_path / "home").fn

    out = save(url="https://jobs.example/job/1", filename="job.html", render=True)

    sidecar = docs / "html" / "job.network.json"
    assert sidecar.exists()
    assert "26033994" in sidecar.read_text(encoding="utf-8")
    assert "JSON endpoint(s) saved beside it" in out
    assert "role=listing" in out


def test_render_still_refuses_a_non_http_scheme(tmp_path, docs, monkeypatch):
    monkeypatch.setattr(documents, "playwright_available", lambda: True)

    def boom(url, renderer=None, wait_for_selector=""):
        raise AssertionError("the browser hop must not run for file://")

    monkeypatch.setattr(documents, "_fetch_rendered", boom)
    save = documents.make_save_html_tool(tmp_path / "home").fn
    out = save(url="file:///etc/passwd", filename="stolen.html", render=True)
    assert "only http and https" in out
    assert not (docs / "html" / "stolen.html").exists()


def test_challenge_page_is_named_not_claimed_as_the_listing(save, docs, site):
    site.routes["/blocked"] = (
        200, "text/html",
        b"<html><title>Just a moment...</title><body>verify you are human</body></html>",
    )
    out = save(url=f"{site.base}/blocked", filename="blocked.html")
    assert "bot-protection" in out
    assert (docs / "html" / "blocked.html").exists()


def test_batch_saves_each_url_in_one_call(save, docs, site):
    site.routes["/a"] = (200, "text/html", b"<html>A</html>")
    site.routes["/b"] = (200, "text/html", b"<html>B</html>")
    site.routes["/gone"] = (404, "text/plain", b"nope")

    out = save(urls=[f"{site.base}/a", f"{site.base}/gone", f"{site.base}/b"])

    assert (docs / "html" / "a.html").read_bytes() == b"<html>A</html>"
    assert (docs / "html" / "b.html").read_bytes() == b"<html>B</html>"
    assert not (docs / "html" / "gone.html").exists()
    assert "Saved 2 of 3" in out
    assert "404" in out


def test_batch_accepts_a_newline_string_and_caps_at_the_ceiling(save, docs, site):
    for n in range(documents.MAX_BATCH + 3):
        site.routes[f"/{n}"] = (200, "text/html", f"<html>{n}</html>".encode())
    blob = "\n".join(f"{site.base}/{n}" for n in range(documents.MAX_BATCH + 3))

    out = save(urls=blob)

    written = list((docs / "html").glob("*.html"))
    assert len(written) == documents.MAX_BATCH
    assert f"ceiling is {documents.MAX_BATCH}" in out


def test_schema_advertises_render_urls_and_wait_for_selector(tmp_path, docs):
    tool = documents.make_save_html_tool(tmp_path / "home")
    props = tool.input_schema["properties"]
    assert {"render", "urls", "wait_for_selector", "jobs", "naming_prefix",
            "company", "subfolder"} <= set(props)
    search = documents.make_search_tool(tmp_path / "home")
    assert {"max_per_file", "filename_pattern", "subfolder"} <= set(
        search.input_schema["properties"]
    )


class _JsonResp:
    def __init__(self, url, body, status=200):
        self.url = url
        self.status = status
        self.headers = {"content-type": "application/json"}
        self._body = body

    def json(self):
        return self._body


def test_json_capture_keeps_the_latest_body_per_endpoint():
    """Phenom hits /api/jobs?page=1 then ?page=2. The first empty list must
    not be what the agent reads."""
    bucket: list[dict] = []
    documents._capture_json(_JsonResp("https://jobs.example/api/jobs?page=1", {"jobs": []}), bucket)
    documents._capture_json(
        _JsonResp("https://jobs.example/api/jobs?page=2", {"jobs": [{"id": "26033994"}]}),
        bucket,
    )
    documents._capture_json(_JsonResp("https://jobs.example/api/telemetry", {"ok": True}), bucket)

    assert len(bucket) == 2
    listing_call = next(h for h in bucket if h["endpoint"].endswith("/api/jobs"))
    assert listing_call["query"] == "page=2"
    assert listing_call["replaced"] == 1
    assert listing_call["body"]["jobs"][0]["id"] == "26033994"


def test_json_capture_tags_the_listing_payload():
    bucket: list[dict] = []
    documents._capture_json(_JsonResp("https://jobs.example/api/telemetry", {"ok": True}), bucket)
    documents._capture_json(
        _JsonResp("https://jobs.example/api/jobs?q=engineer", {"jobs": [{"id": "1"}]}),
        bucket,
    )
    documents._tag_listing_hits(bucket, "https://jobs.example/search")

    roles = {h["endpoint"].rsplit("/", 1)[-1]: h["role"] for h in bucket}
    assert roles["jobs"] == "listing"
    assert roles["telemetry"] == "other"


def test_wait_for_selector_is_forwarded_to_the_renderer(tmp_path, docs, monkeypatch):
    seen: dict[str, str] = {}

    def fake(url, renderer=None, wait_for_selector=""):
        seen["selector"] = wait_for_selector
        return documents.FetchResult(PAINTED, "rendered in Chromium, 1 bytes", rendered=True)

    monkeypatch.setattr(documents, "playwright_available", lambda: True)
    monkeypatch.setattr(documents, "_fetch_rendered", fake)
    save = documents.make_save_html_tool(tmp_path / "home").fn
    save(url="https://jobs.example/search", render=True,
         wait_for_selector=".job-search-result")
    assert seen["selector"] == ".job-search-result"


def test_wait_for_selector_rejects_a_multiline_blob(save, docs):
    out = save(url="https://jobs.example/search", render=True,
               wait_for_selector=".a\n.b")
    assert "Nothing saved" in out and "single CSS selector" in out
    assert not list((docs / "html").glob("*.html"))


def test_jobs_keeps_a_custom_filename_per_url(save, docs, site):
    """The reason a batch used to be unusable for a crawl: urls=[] derived
    every name from the slug and dropped the caller's YYYY-MM-DD-company-title."""
    site.routes["/engineer"] = (200, "text/html", b"<html>engineer</html>")
    site.routes["/analyst"] = (200, "text/html", b"<html>analyst</html>")

    out = save(jobs=[
        {"url": f"{site.base}/engineer",
         "filename": "2026-09-18-bank-of-america-software-engineer-iii.html"},
        {"url": f"{site.base}/analyst",
         "filename": "2026-09-18-bank-of-america-business-analyst.html"},
    ])

    html_folder = docs / "html"
    assert (html_folder / "2026-09-18-bank-of-america-software-engineer-iii.html").read_bytes() == b"<html>engineer</html>"
    assert (html_folder / "2026-09-18-bank-of-america-business-analyst.html").read_bytes() == b"<html>analyst</html>"
    assert "Saved 2 of 2" in out


def test_naming_prefix_is_prepended_to_each_slug(save, docs, site):
    site.routes["/software-engineer-iii-26032981"] = (200, "text/html", b"<html>eng</html>")
    site.routes["/analyst-42"] = (200, "text/html", b"<html>an</html>")

    save(urls=[f"{site.base}/software-engineer-iii-26032981",
               f"{site.base}/analyst-42"],
         naming_prefix="2026-09-18-bank-of-america")

    names = sorted(p.name for p in (docs / "html").glob("*.html"))
    assert names == [
        "2026-09-18-bank-of-america-analyst-42.html",
        "2026-09-18-bank-of-america-software-engineer-iii-26032981.html",
    ]


def test_date_company_title_filename_is_not_clipped_at_sixty_chars(save, docs):
    """The old stem cap was 60. YYYY-MM-DD-company-title overshoots that on a
    long posting name; clipping would silently merge two different jobs."""
    long_name = "2026-09-18-bank-of-america-vice-president-software-engineer-iii.html"
    assert len(Path(long_name).stem) > 60
    save(filename=long_name, html="<html>vp</html>")
    assert (docs / "html" / long_name).exists()


def test_jobs_json_string_is_accepted(save, docs, site):
    """Models sometimes stringify the array. A TypeError would waste the turn."""
    site.routes["/a"] = (200, "text/html", b"<html>A</html>")
    blob = json.dumps([{"url": f"{site.base}/a", "filename": "named-a.html"}])
    save(jobs=blob)
    assert (docs / "html" / "named-a.html").read_bytes() == b"<html>A</html>"


def test_json_listing_api_is_pretty_printed_as_json(save, docs, site):
    """Pointing save_html at jobssearchservlet should keep readable JSON, not
    a one-line blob named .html that search_documents then truncates."""
    body = json.dumps({"jobs": [{"title": "Software Engineer III", "id": "1"}] * 3},
                      separators=(",", ":")).encode()
    site.routes["/services/jobssearchservlet"] = (200, "application/json", body)

    out = save(url=f"{site.base}/services/jobssearchservlet",
               filename="boa-technology.json")

    dest = docs / "html" / "boa-technology.json"
    assert dest.exists()
    parsed = json.loads(dest.read_text(encoding="utf-8"))
    assert parsed["jobs"][0]["title"] == "Software Engineer III"
    assert dest.read_text(encoding="utf-8").count("\n") > 3
    assert "saved it as JSON" in out
    assert not (docs / "html" / "boa-technology.html").exists()


def test_json_listing_derives_a_json_extension_from_the_url(save, docs, site):
    body = b'{"jobs":[]}'
    site.routes["/services/jobssearchservlet"] = (200, "application/json", body)
    save(url=f"{site.base}/services/jobssearchservlet")
    assert (docs / "html" / "jobssearchservlet.json").exists()


def test_listing_sidecar_names_the_api_url_to_fetch_next(tmp_path, docs, monkeypatch):
    hits = [{"url": "https://careers.example/services/jobssearchservlet?rows=100",
             "endpoint": "https://careers.example/services/jobssearchservlet",
             "query": "rows=100", "status": 200, "replaced": 0, "role": "listing",
             "body": {"jobs": [{"id": "1"}]}}]
    monkeypatch.setattr(documents, "playwright_available", lambda: True)
    monkeypatch.setattr(documents, "_fetch_rendered", _fake_renderer(json_hits=hits))
    save = documents.make_save_html_tool(tmp_path / "home").fn

    out = save(url="https://careers.example/search", filename="search.html", render=True)

    assert "jobssearchservlet?rows=100" in out
    assert "no render needed" in out


def test_json_capture_tags_a_jobssearchservlet_path():
    bucket: list[dict] = []
    documents._capture_json(_JsonResp("https://careers.example/api/telemetry", {"ok": True}), bucket)
    documents._capture_json(
        _JsonResp("https://careers.example/services/jobssearchservlet?rows=100",
                  {"count": 12, "items": [{"id": "1"}]}),
        bucket,
    )
    documents._tag_listing_hits(bucket, "https://careers.example/search")
    tagged = {h["endpoint"].rsplit("/", 1)[-1]: h["role"] for h in bucket}
    assert tagged["jobssearchservlet"] == "listing"
    assert tagged["telemetry"] == "other"


# ---------- company crawls: folder layout + day's manifest


def test_crawl_date_uses_pacific_not_utc(monkeypatch):
    """06:30 UTC on 19 Sep is still 18 Sep in Pacific Time. A UTC stamp
    would put tonight's crawl in tomorrow's folder."""
    frozen = datetime(2026, 9, 19, 6, 30, tzinfo=UTC)
    monkeypatch.setattr(
        documents, "_now", lambda: frozen.astimezone(documents._pacific_tz()))
    assert documents._crawl_date() == "2026-09-18"
    assert frozen.strftime("%Y-%m-%d") == "2026-09-19"


def test_company_saves_under_jobs_company_date(save, docs, site, monkeypatch):
    monkeypatch.setattr(documents, "_crawl_date", lambda: "2026-09-18")
    site.routes["/job"] = (200, "text/html", PAGE)

    out = save(url=f"{site.base}/job", filename="role.html", company="JPMorgan")

    dest = docs / "jobs" / "jpmorgan" / "2026-09-18" / "role.html"
    assert dest.read_bytes() == PAGE
    assert not (docs / "html" / "role.html").exists()
    assert str(dest) in out


def test_company_strips_path_from_filename_still(save, docs, tmp_path, monkeypatch):
    monkeypatch.setattr(documents, "_crawl_date", lambda: "2026-09-18")
    save(filename="../../escaped.html", html="<html>x</html>", company="jpmorgan")

    assert not (tmp_path / "escaped.html").exists()
    assert (docs / "jobs" / "jpmorgan" / "2026-09-18" / "escaped.html").exists()
    assert not (docs / "html" / "escaped.html").exists()


def test_subfolder_saves_where_asked_and_refuses_escape(save, docs, tmp_path):
    out = save(filename="role.html", html="<html>x</html>",
               subfolder="jobs/jpmorgan/2026-09-18")
    dest = docs / "jobs" / "jpmorgan" / "2026-09-18" / "role.html"
    assert dest.exists()
    assert str(dest) in out

    secret = tmp_path / "secret.html"
    refused = save(filename="nope.html", html="<html>x</html>", subfolder="..")
    assert refused.startswith("refused:")
    assert not secret.exists()


def test_company_crawl_writes_a_manifest_of_roles(save, docs, site, monkeypatch):
    monkeypatch.setattr(documents, "_crawl_date", lambda: "2026-09-18")
    site.routes["/engineer"] = (200, "text/html", b"<html>engineer</html>")
    site.routes["/analyst"] = (200, "text/html", b"<html>analyst</html>")

    out = save(
        company="jpmorgan",
        jobs=[
            {"url": f"{site.base}/engineer",
             "filename": "2026-09-18-jpmorgan-back-end-java-software-engineer-iii-210788991.html",
             "title": "Back-end Java Software Engineer III",
             "req_id": "210788991",
             "apply_url": f"{site.base}/engineer"},
            {"url": f"{site.base}/analyst",
             "filename": "2026-09-18-jpmorgan-python-engineer-210788992.html"},
        ],
    )

    folder = docs / "jobs" / "jpmorgan" / "2026-09-18"
    manifest_path = folder / "manifest-jpmorgan-2026-09-18.json"
    assert manifest_path.exists()
    assert str(manifest_path) in out
    doc = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert doc["company"] == "jpmorgan"
    assert doc["date_crawled"] == "2026-09-18"
    assert doc["total_roles_found"] == 2
    roles = doc["roles"]
    assert roles[0]["title"] == "Back-end Java Software Engineer III"
    assert roles[0]["req_id"] == "210788991"
    assert roles[0]["apply_url"] == f"{site.base}/engineer"
    assert roles[0]["file_path"] == (
        "jobs/jpmorgan/2026-09-18/"
        "2026-09-18-jpmorgan-back-end-java-software-engineer-iii-210788991.html"
    )
    assert roles[1]["req_id"] == "210788992"
    assert "python engineer" in roles[1]["title"].lower()
    assert (folder / "2026-09-18-jpmorgan-python-engineer-210788992.html").exists()


def test_company_manifest_merges_a_second_batch_the_same_day(save, docs, site, monkeypatch):
    """A follow-up call for URLs past the per-call ceiling must update the
    day's index, not leave the first batch as the whole answer."""
    monkeypatch.setattr(documents, "_crawl_date", lambda: "2026-09-18")
    site.routes["/one"] = (200, "text/html", b"<html>one</html>")
    site.routes["/two"] = (200, "text/html", b"<html>two</html>")

    save(company="jpmorgan", jobs=[
        {"url": f"{site.base}/one", "filename": "one.html",
         "title": "Role One", "req_id": "1"},
    ])
    save(company="jpmorgan", jobs=[
        {"url": f"{site.base}/two", "filename": "two.html",
         "title": "Role Two", "req_id": "2"},
    ])

    doc = json.loads(
        (docs / "jobs" / "jpmorgan" / "2026-09-18" / "manifest-jpmorgan-2026-09-18.json")
        .read_text(encoding="utf-8")
    )
    assert doc["total_roles_found"] == 2
    assert [r["req_id"] for r in doc["roles"]] == ["1", "2"]


def test_jobs_without_company_does_not_write_a_manifest(save, docs, site):
    site.routes["/a"] = (200, "text/html", b"<html>A</html>")
    save(jobs=[{"url": f"{site.base}/a", "filename": "named-a.html"}])
    html_folder = docs / "html"
    assert (html_folder / "named-a.html").exists()
    assert list(html_folder.glob("manifest-*.json")) == []
    assert not (docs / "jobs").exists()


def test_company_page_files_are_still_never_overwritten(save, docs, site, monkeypatch):
    monkeypatch.setattr(documents, "_crawl_date", lambda: "2026-09-18")
    site.routes["/job"] = (200, "text/html", b"<html>first</html>")
    save(url=f"{site.base}/job", filename="role.html", company="jpmorgan")
    site.routes["/job"] = (200, "text/html", b"<html>second</html>")
    save(url=f"{site.base}/job", filename="role.html", company="jpmorgan")

    folder = docs / "jobs" / "jpmorgan" / "2026-09-18"
    assert "<html>first</html>" in (folder / "role.html").read_text(encoding="utf-8")
    assert "<html>second</html>" in (folder / "role-2.html").read_text(encoding="utf-8")


# ---------- wiring


def test_both_tools_are_registered(tmp_path, docs):
    app = make_waku(tmp_path / "home", client=ScriptedClient([]))
    names = {s["name"] for s in app.tools.schemas()}
    assert {"search_documents", "save_html", "query_json"} <= names
    app.close()


def test_registry_exposes_the_tools_without_memory(tmp_path, docs):
    """build_registry is called with memory=None in some paths; these two must
    not quietly depend on it."""
    from waku.config import Settings

    registry = build_registry(conn=None, settings=Settings(home=tmp_path / "home"), memory=None)
    assert {"search_documents", "save_html", "query_json"} <= {s["name"] for s in registry.schemas()}


def test_scripted_turn_fires_save_html_and_writes_the_file(tmp_path, docs):
    """Offline tier: the model asks for the tool, the loop runs it, the artifact
    exists on disk — the same shape as test_tool_trigger.py's calendar case."""
    gate = response([text_block('{"retrieve": false, "query": "", "reason": "test"}')])
    script = [gate] + [
        response([tool_block("save_html", {"filename": "digest.html",
                                           "html": "<html><body>week</body></html>"})], "tool_use"),
        response([text_block("Saved it.")]),
    ]
    app = make_waku(tmp_path / "home", client=ScriptedClient(script))
    result = app.respond("save my week as an html page")

    assert [c["tool"] for c in result.tool_calls] == ["save_html"]
    assert "week" in (docs / "html" / "digest.html").read_text(encoding="utf-8")
    app.close()
