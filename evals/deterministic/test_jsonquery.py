"""DETERMINISTIC EVAL — query_json: field predicates, not line grep.

search_documents would return five accidental keyword hits from a listing
sidecar. query_json has to keep the jobs whose postedDate is today, drop the
ones that are not, and refuse to eval Python. The language is a subset: an
unsupported expression must say so, not run as code.
"""

from __future__ import annotations

import json
from datetime import date

import pytest

from evals.helpers import ScriptedClient, make_waku, response, text_block, tool_block
from waku.tools import build_registry, jsonquery
from waku.tools.documents import DOCS_DIR_ENV

LISTING = {
    "jobsList": [
        {"postingTitle": "AI Engineer", "postedDate": "09/18/2026",
         "area": "Technology; Operations", "jcrURL": "/job/ai", "family": "Technology"},
        {"postingTitle": "Teller", "postedDate": "09/17/2026",
         "area": "Customer Service", "jcrURL": "/job/teller", "family": "Operations"},
        {"postingTitle": "Software Engineer III", "postedDate": "09/18/2026",
         "area": "Technology", "jcrURL": "/job/swe", "family": "Technology"},
    ]
}

SIDECAR = [
    {"role": "other", "body": {}},
    {"role": "listing", "endpoint": "https://careers.example/services/jobssearchservlet",
     "body": LISTING},
]


@pytest.fixture
def docs(tmp_path, monkeypatch):
    folder = tmp_path / "docs"
    folder.mkdir()
    monkeypatch.setenv(DOCS_DIR_ENV, str(folder))
    return folder


@pytest.fixture
def query(tmp_path, docs, monkeypatch):
    monkeypatch.setattr(jsonquery, "_today", lambda: date(2026, 9, 18))
    return jsonquery.make_query_json_tool(tmp_path / "home").fn


def _write_listing(docs):
    dest = docs / "html"
    dest.mkdir()
    (dest / "job-search.network.json").write_text(
        json.dumps(SIDECAR), encoding="utf-8")
    return dest / "job-search.network.json"


# ---------- language (no files)


def test_select_today_matches_us_and_iso_dates(monkeypatch):
    monkeypatch.setattr(jsonquery, "_today", lambda: date(2026, 9, 18))
    jobs = [
        {"title": "A", "postedDate": "09/18/2026"},
        {"title": "B", "postedDate": "2026-09-18"},
        {"title": "C", "postedDate": "09/17/2026"},
        {"title": "D", "postedDate": "2026-09-17"},
    ]
    out = jsonquery.run(jobs, ".[] | select(.postedDate == today)")
    assert [j["title"] for j in out] == ["A", "B"]

    out = jsonquery.run(jobs, '.[] | select(.postedDate == "today")')
    assert [j["title"] for j in out] == ["A", "B"]


def test_recursive_descent_and_contains():
    out = jsonquery.run(SIDECAR, '..jobsList[] | select(.area | contains("Technology"))')
    assert [j["postingTitle"] for j in out] == ["AI Engineer", "Software Engineer III"]


def test_jsonpath_filter_on_nested_array(monkeypatch):
    monkeypatch.setattr(jsonquery, "_today", lambda: date(2026, 9, 18))
    out = jsonquery.run(SIDECAR, "$..jobsList[?(@.postedDate==today)]")
    assert [j["postingTitle"] for j in out] == ["AI Engineer", "Software Engineer III"]


def test_index_and_scalar_path():
    out = jsonquery.run(SIDECAR, ".[1].body.jobsList[0].postingTitle")
    assert out == ["AI Engineer"]


def test_and_or_and_not_equal():
    out = jsonquery.run(
        SIDECAR,
        '..jobsList[] | select(.postedDate == "09/18/2026" and .family == "Technology")',
    )
    assert [j["postingTitle"] for j in out] == ["AI Engineer", "Software Engineer III"]
    out = jsonquery.run(SIDECAR, '..jobsList[] | select(.family != "Technology")')
    assert [j["postingTitle"] for j in out] == ["Teller"]


def test_unsupported_jq_is_refused_not_evald():
    with pytest.raises(jsonquery.QueryError, match="subset"):
        jsonquery.run({"x": 1}, "map(.x) | add")
    with pytest.raises(jsonquery.QueryError):
        jsonquery.run({"x": 1}, "select(__import__('os').system('pwd'))")


# ---------- tool


def test_query_json_filters_the_listing_file(query, docs):
    _write_listing(docs)
    out = query(path="html/job-search.network.json",
                query="..jobsList[] | select(.postedDate == today)",
                fields=["postingTitle", "postedDate", "jcrURL"])
    assert "2 match(es)" in out
    assert "AI Engineer" in out and "Software Engineer III" in out
    assert "Teller" not in out
    assert "jcrURL" in out


def test_query_json_caps_and_says_so(query, docs):
    jobs = {"jobsList": [{"id": n, "postedDate": "09/18/2026"} for n in range(50)]}
    (docs / "big.json").write_text(json.dumps(jobs), encoding="utf-8")
    out = query(path="big.json", query="..jobsList[]", max_results=10)
    shown = [ln for ln in out.splitlines() if ln.startswith("- ")]
    assert len(shown) == 10
    assert "First 10 of 50" in out


def test_empty_select_names_the_arrays(query, docs):
    _write_listing(docs)
    out = query(path="html/job-search.network.json",
                query='..jobsList[] | select(.postingTitle == "nope")')
    assert "No matches" in out
    assert "jobsList" in out


def test_missing_query_lists_arrays(query, docs):
    _write_listing(docs)
    out = query(path="html/job-search.network.json")
    assert "Pass a query" in out
    assert "jobsList" in out


def test_omitted_path_lists_json_files(query, docs):
    _write_listing(docs)
    out = query()
    assert "needs a path" in out
    assert "job-search.network.json" in out


def test_refuses_a_path_outside_the_folder(query, docs, tmp_path):
    secret = tmp_path / "secret.json"
    secret.write_text('{"token": "sk-live"}', encoding="utf-8")
    out = query(path="../secret.json", query=".token")
    assert out.startswith("refused:")
    assert "sk-live" not in out


def test_missing_file_is_named(query, docs):
    out = query(path="nope.json", query=".[]")
    assert "No such file" in out and "nope.json" in out


def test_invalid_json_is_named(query, docs):
    (docs / "broken.json").write_text("{not json", encoding="utf-8")
    out = query(path="broken.json", query=".[]")
    assert "not valid JSON" in out


def test_unsupported_query_explains_the_subset(query, docs):
    (docs / "x.json").write_text("{}", encoding="utf-8")
    out = query(path="x.json", query="map(.foo)")
    assert "Nothing returned" in out and "subset" in out


def test_basename_is_found_under_html(query, docs):
    _write_listing(docs)
    out = query(path="job-search.network.json", query="..jobsList[]")
    assert "3 match(es)" in out


# ---------- wiring


def test_query_json_is_registered(tmp_path, docs):
    app = make_waku(tmp_path / "home", client=ScriptedClient([]))
    names = {s["name"] for s in app.tools.schemas()}
    assert "query_json" in names
    app.close()


def test_registry_exposes_query_json_without_memory(tmp_path, docs):
    from waku.config import Settings

    registry = build_registry(conn=None, settings=Settings(home=tmp_path / "home"), memory=None)
    assert "query_json" in {s["name"] for s in registry.schemas()}


def test_scripted_turn_fires_query_json(tmp_path, docs, monkeypatch):
    monkeypatch.setattr(jsonquery, "_today", lambda: date(2026, 9, 18))
    _write_listing(docs)
    gate = response([text_block('{"retrieve": false, "query": "", "reason": "test"}')])
    script = [gate] + [
        response([tool_block("query_json", {
            "path": "html/job-search.network.json",
            "query": "..jobsList[] | select(.postedDate == today)",
            "fields": ["postingTitle"],
        })], "tool_use"),
        response([text_block("Two jobs today.")]),
    ]
    app = make_waku(tmp_path / "home", client=ScriptedClient(script))
    result = app.respond("which saved jobs were posted today?")
    assert [c["tool"] for c in result.tool_calls] == ["query_json"]
    assert "AI Engineer" in result.tool_calls[0]["output"]
    assert "Teller" not in result.tool_calls[0]["output"]
    app.close()
