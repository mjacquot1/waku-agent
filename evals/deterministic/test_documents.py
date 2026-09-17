"""DETERMINISTIC EVAL — the documents folder: honest reads, non-destructive writes.

Two guarantees are worth pinning in CI, and they are not the happy path:

1. **search_documents never launders a skip into an empty answer.** A folder
   with one .pdf and one .md must report the .pdf as unsearched, or "nothing
   about the invoice" means both "it isn't there" and "I couldn't open it".
2. **save_html never overwrites.** The model re-calls tools; if the second call
   with the same filename clobbered the first file, a generated report the user
   already opened would vanish with no trace that it ever existed.

Path containment is checked the way test_gh_tool.py checks argv: assert on what
WOULD happen (the file that does/doesn't exist on disk), not on the prose.
"""

from __future__ import annotations

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
def save(tmp_path, docs):
    return documents.make_save_html_tool(tmp_path / "home").fn


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
    assert "needs both" in save(filename="ok.html", html="")


def test_html_subdir_override_cannot_climb_out(tmp_path, docs, monkeypatch):
    monkeypatch.setenv(documents.HTML_SUBDIR_ENV, "../../elsewhere")
    assert documents.html_dir(tmp_path / "home").parent == docs.resolve()


# ---------- wiring


def test_both_tools_are_registered(tmp_path, docs):
    app = make_waku(tmp_path / "home", client=ScriptedClient([]))
    names = {s["name"] for s in app.tools.schemas()}
    assert {"search_documents", "save_html"} <= names
    app.close()


def test_registry_exposes_the_tools_without_memory(tmp_path, docs):
    """build_registry is called with memory=None in some paths; these two must
    not quietly depend on it."""
    from waku.config import Settings

    registry = build_registry(conn=None, settings=Settings(home=tmp_path / "home"), memory=None)
    assert {"search_documents", "save_html"} <= {s["name"] for s in registry.schemas()}


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
