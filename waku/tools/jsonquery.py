"""query_json — filter a saved JSON document with a jq/JSONPath subset.

search_documents is line grep. A 28,000-line listing sidecar is not a text
file: the question is "jobs posted today in Technology", and that is a field
predicate, not a keyword. This tool walks the structure. It does not shell out
to jq and it does not add a JSONPath library — the core takes no new
dependency — so the language is a subset, and an unsupported expression says
so instead of running as Python.

Supported:
  .foo.bar          key path
  [0] / [-1]        index
  [] / .[] / [*]    iterate an array (or a dict's values)
  ..foo             every foo anywhere under the current nodes
  |                 pipe
  select(PRED)      keep nodes for which PRED is true
  [?(@.foo=="x")]   JSONPath filter on an array, same predicates

PRED is comparisons chained with and / or (and binds tighter):
  .postedDate == today          today/yesterday match ISO and US dates
  .area | contains("Technology")
  .title | startswith("AI")
  .id != null

Pass fields=["postingTitle", "jcrURL"] to keep the reply to the columns that
matter. Without fields, nulls and nested objects are dropped so a job record
does not dump twenty empty keys into the prompt.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta
from pathlib import Path

from waku.tools.documents import DOCS_DIR_ENV, docs_root
from waku.tools.registry import Tool

MAX_RESULTS = 40
MAX_RESULTS_CEILING = 200
MAX_HINTS = 8
SCALAR_FIELDS_CAP = 12
VALUE_CHARS = 200

_TODAY = object()
_YESTERDAY = object()
_MISSING = object()

_EXAMPLES = (
    '..jobsList[] | select(.postedDate == today)',
    '..jobsList[] | select(.area | contains("Technology"))',
    '$..jobsList[?(@.postedDate==today)]',
)


class QueryError(Exception):
    """A sentence the model can act on, not a stack trace."""


def run(data, query: str) -> list:
    """Evaluate `query` against already-parsed JSON. Public for tests."""
    stages = _split_stages(query)
    if not stages:
        raise QueryError("query is empty")
    nodes: list = [data]
    for stage in stages:
        if _is_select(stage):
            nodes = [n for n in nodes if _pred(n, _select_body(stage))]
        else:
            nodes = _walk_path(nodes, stage)
    return nodes


def _split_stages(query: str) -> list[str]:
    text = (query or "").strip()
    if not text:
        return []
    stages: list[str] = []
    buf: list[str] = []
    depth = 0
    quote = ""
    i = 0
    while i < len(text):
        ch = text[i]
        if quote:
            buf.append(ch)
            if ch == quote and text[i - 1] != "\\":
                quote = ""
            i += 1
            continue
        if ch in "'\"":
            quote = ch
            buf.append(ch)
            i += 1
            continue
        if ch in "([":
            depth += 1
            buf.append(ch)
            i += 1
            continue
        if ch in ")]":
            depth = max(0, depth - 1)
            buf.append(ch)
            i += 1
            continue
        if ch == "|" and depth == 0:
            stage = "".join(buf).strip()
            if not stage:
                raise QueryError("query has an empty pipe stage")
            stages.append(stage)
            buf = []
            i += 1
            continue
        buf.append(ch)
        i += 1
    tail = "".join(buf).strip()
    if tail:
        stages.append(tail)
    elif stages:
        raise QueryError("query ends with a pipe")
    return stages


def _is_select(stage: str) -> bool:
    s = stage.strip()
    return s.startswith("select(") and s.endswith(")")


def _select_body(stage: str) -> str:
    s = stage.strip()
    return s[len("select("):-1].strip()


def _walk_path(nodes: list, expr: str) -> list:
    text = expr.strip().removeprefix("$")
    i = 0
    out = list(nodes)
    while i < len(text):
        if text[i].isspace():
            i += 1
            continue
        if text.startswith(".[]", i) or text.startswith(".[*]", i):
            out = _splat(out)
            i += 4 if text.startswith(".[*]", i) else 3
            continue
        if text.startswith("..", i):
            ident, i = _read_ident(text, i + 2)
            if not ident:
                raise QueryError(".. needs a key, e.g. ..jobsList")
            out = _descend(out, ident)
            continue
        if text[i] == ".":
            if i + 1 < len(text) and text[i + 1] == "[":
                i += 1
                continue
            ident, i = _read_ident(text, i + 1)
            if not ident:
                raise QueryError("dot with no key — use .[] to iterate an array")
            out = _pluck(out, ident)
            continue
        if text[i] == "[":
            close = _match_bracket(text, i)
            out = _apply_bracket(out, text[i + 1:close])
            i = close + 1
            continue
        raise QueryError(
            f"unsupported query syntax at {text[i]!r}. This is a jq/JSONPath "
            f"subset, not full jq. Examples: {'; '.join(_EXAMPLES)}"
        )
    return out


def _read_ident(text: str, i: int) -> tuple[str, int]:
    if i < len(text) and text[i] in "'\"":
        quote = text[i]
        j = i + 1
        while j < len(text) and text[j] != quote:
            j += 1
        if j >= len(text):
            raise QueryError("unterminated quoted key")
        return text[i + 1:j], j + 1
    j = i
    while j < len(text) and (text[j].isalnum() or text[j] == "_"):
        j += 1
    return text[i:j], j


def _match_bracket(text: str, i: int) -> int:
    depth = 0
    quote = ""
    for j in range(i, len(text)):
        ch = text[j]
        if quote:
            if ch == quote and text[j - 1] != "\\":
                quote = ""
            continue
        if ch in "'\"":
            quote = ch
            continue
        if ch == "[":
            depth += 1
        elif ch == "]":
            depth -= 1
            if depth == 0:
                return j
    raise QueryError("unclosed [ in query")


def _pluck(nodes: list, key: str) -> list:
    out = []
    for node in nodes:
        if isinstance(node, dict) and key in node:
            out.append(node[key])
    return out


def _splat(nodes: list) -> list:
    out = []
    for node in nodes:
        if isinstance(node, list):
            out.extend(node)
        elif isinstance(node, dict):
            out.extend(node.values())
    return out


def _descend(nodes: list, key: str) -> list:
    found: list = []

    def walk(node) -> None:
        if isinstance(node, dict):
            if key in node:
                found.append(node[key])
            for value in node.values():
                walk(value)
        elif isinstance(node, list):
            for value in node:
                walk(value)

    for node in nodes:
        walk(node)
    return found


def _apply_bracket(nodes: list, inner: str) -> list:
    spec = inner.strip()
    if spec in ("", "*"):
        return _splat(nodes)
    if spec.lstrip("-").isdigit():
        idx = int(spec)
        out = []
        for node in nodes:
            if not isinstance(node, list):
                continue
            try:
                out.append(node[idx])
            except IndexError:
                continue
        return out
    if spec.startswith("?(") and spec.endswith(")"):
        pred = spec[2:-1].strip()
        out = []
        for node in nodes:
            if isinstance(node, list):
                out.extend(el for el in node if _pred(el, pred))
            elif _pred(node, pred):
                out.append(node)
        return out
    raise QueryError(
        f"unsupported bracket [ {spec} ]. Use [0], [], [*] or [?(@.field==value)]"
    )


def _pred(item, expr: str) -> bool:
    text = (expr or "").strip()
    if not text:
        raise QueryError("select() needs a predicate")
    return _eval_or(item, text)


def _eval_or(item, expr: str) -> bool:
    parts = _split_logic(expr, " or ")
    return any(_eval_and(item, part) for part in parts)


def _eval_and(item, expr: str) -> bool:
    parts = _split_logic(expr, " and ")
    return all(_eval_cmp(item, part) for part in parts)


def _split_logic(expr: str, sep: str) -> list[str]:
    parts: list[str] = []
    buf: list[str] = []
    depth = 0
    quote = ""
    i = 0
    lower = expr
    sep_l = sep
    while i < len(expr):
        ch = expr[i]
        if quote:
            buf.append(ch)
            if ch == quote and expr[i - 1] != "\\":
                quote = ""
            i += 1
            continue
        if ch in "'\"":
            quote = ch
            buf.append(ch)
            i += 1
            continue
        if ch in "([":
            depth += 1
            buf.append(ch)
            i += 1
            continue
        if ch in ")]":
            depth = max(0, depth - 1)
            buf.append(ch)
            i += 1
            continue
        if depth == 0 and lower[i:i + len(sep_l)].lower() == sep_l:
            parts.append("".join(buf).strip())
            buf = []
            i += len(sep_l)
            continue
        buf.append(ch)
        i += 1
    parts.append("".join(buf).strip())
    return [p for p in parts if p]


def _eval_cmp(item, expr: str) -> bool:
    text = expr.strip()
    if text.startswith("(") and text.endswith(")"):
        return _pred(item, text[1:-1])
    pipe = _split_stages(text)
    if len(pipe) == 2 and pipe[1].startswith("contains("):
        value = _parse_value(_call_arg(pipe[1], "contains"))
        return _contains(_lookup(item, pipe[0]), value)
    if len(pipe) == 2 and pipe[1].startswith("startswith("):
        value = _parse_value(_call_arg(pipe[1], "startswith"))
        left = _lookup(item, pipe[0])
        return isinstance(left, str) and isinstance(value, str) and left.startswith(value)
    for op in ("==", "!="):
        if op not in text:
            continue
        left_s, right_s = _split_once(text, op)
        left = _lookup(item, left_s)
        right = _parse_value(right_s)
        matched = _equals(left, right)
        return (not matched) if op == "!=" else matched
    raise QueryError(
        f"unsupported predicate {expr!r}. Use .field == value, .field != value, "
        f'.field | contains("x"), or .field | startswith("x")'
    )


def _split_once(text: str, op: str) -> tuple[str, str]:
    depth = 0
    quote = ""
    i = 0
    while i < len(text):
        ch = text[i]
        if quote:
            if ch == quote and text[i - 1] != "\\":
                quote = ""
            i += 1
            continue
        if ch in "'\"":
            quote = ch
            i += 1
            continue
        if ch in "([":
            depth += 1
        elif ch in ")]":
            depth = max(0, depth - 1)
        elif depth == 0 and text.startswith(op, i):
            return text[:i].strip(), text[i + len(op):].strip()
        i += 1
    raise QueryError(f"predicate is missing {op}")


def _call_arg(stage: str, name: str) -> str:
    if not (stage.startswith(name + "(") and stage.endswith(")")):
        raise QueryError(f"{name}() needs parentheses")
    return stage[len(name) + 1:-1].strip()


def _lookup(item, accessor: str):
    path = accessor.strip().removeprefix("@")
    if path in ("", "."):
        return item
    nodes = _walk_path([item], path)
    if not nodes:
        return _MISSING
    return nodes[0]


def _parse_value(raw: str):
    text = raw.strip()
    if not text:
        raise QueryError("comparison is missing a value")
    lower = text.lower()
    if lower == "today":
        return _TODAY
    if lower == "yesterday":
        return _YESTERDAY
    if lower == "true":
        return True
    if lower == "false":
        return False
    if lower == "null":
        return None
    if text[0] in "'\"" and len(text) >= 2 and text[-1] == text[0]:
        inner = text[1:-1]
        if inner.lower() == "today":
            return _TODAY
        if inner.lower() == "yesterday":
            return _YESTERDAY
        return inner
    try:
        return int(text)
    except ValueError:
        pass
    try:
        return float(text)
    except ValueError:
        pass
    raise QueryError(
        f"cannot parse value {raw!r} — quote strings, or use today / yesterday / "
        "true / false / null / a number"
    )


def _today():
    return datetime.now().astimezone().date()


def _equals(left, right) -> bool:
    if left is _MISSING:
        return False
    if right is _TODAY:
        return _matches_day(left, _today())
    if right is _YESTERDAY:
        return _matches_day(left, _today() - timedelta(days=1))
    return left == right


def _matches_day(value, day) -> bool:
    text = str(value).strip()
    iso = day.isoformat()
    us = f"{day.month:02d}/{day.day:02d}/{day.year}"
    us_short = f"{day.month}/{day.day}/{day.year}"
    eu = f"{day.day:02d}/{day.month:02d}/{day.year}"
    return text in {iso, us, us_short, eu, "today", "Today", day.strftime("%Y%m%d")}


def _contains(haystack, needle) -> bool:
    if haystack is _MISSING or needle is _TODAY or needle is _YESTERDAY:
        return False
    if isinstance(haystack, str) and isinstance(needle, str):
        return needle in haystack
    if isinstance(haystack, list):
        return needle in haystack
    return False


def _array_hints(data) -> list[str]:
    """Name arrays of objects near the top of the document so a missed .jobs
    can be retried as ..jobsList without the model grepping 28,000 lines."""
    found: list[str] = []

    def walk(node, prefix: str, depth: int) -> None:
        if len(found) >= MAX_HINTS or depth > 5:
            return
        if isinstance(node, dict):
            for key, value in node.items():
                path = f"{prefix}.{key}" if prefix else key
                if (isinstance(value, list) and value and isinstance(value[0], dict)):
                    found.append(f"{path} ({len(value)} objects)")
                walk(value, path, depth + 1)
        elif isinstance(node, list) and node:
            if prefix == "" and isinstance(node[0], dict):
                found.append(f"[] ({len(node)} objects)")
            walk(node[0], prefix + "[]" if prefix else "[]", depth + 1)

    walk(data, "", 0)
    return found


def _coerce_fields(fields) -> list[str]:
    if not fields:
        return []
    if isinstance(fields, str):
        return [part.strip() for part in fields.replace(",", " ").split() if part.strip()]
    if isinstance(fields, list):
        return [str(part).strip() for part in fields if str(part).strip()]
    return []


def _project(item, fields: list[str]):
    if not isinstance(item, dict):
        return item
    if fields:
        return {key: item.get(key) for key in fields}
    out = {}
    for key, value in item.items():
        if value is None:
            continue
        if isinstance(value, (dict, list)):
            continue
        out[key] = value
        if len(out) >= SCALAR_FIELDS_CAP:
            break
    return out


def _format_item(item) -> str:
    if isinstance(item, (dict, list)):
        blob = json.dumps(item, ensure_ascii=False, default=str)
    else:
        blob = str(item)
    if len(blob) > VALUE_CHARS * 4:
        blob = blob[:VALUE_CHARS * 4] + "…"
    return blob


def _resolve_json(root: Path, path: str) -> Path | str:
    """Return a file inside root, or a sentence explaining why not."""
    rel = (path or "").strip()
    if not rel:
        jsons = sorted(p for p in root.rglob("*.json") if p.is_file() and not p.name.startswith("."))
        if not jsons:
            return f"No JSON files in {root}."
        listing = "\n".join(f"- {p.relative_to(root)}" for p in jsons[:50])
        more = f"\n(+{len(jsons) - 50} more)" if len(jsons) > 50 else ""
        return f"query_json needs a path. JSON files in {root}:\n{listing}{more}"
    candidate = Path(rel).expanduser()
    if not candidate.is_absolute():
        candidate = root / rel
    candidate = candidate.resolve()
    if candidate != root and root not in candidate.parents:
        return f"refused: '{rel}' is outside the documents folder ({root})."
    if candidate.is_file():
        return candidate
    matches = [p for p in root.rglob(Path(rel).name) if p.is_file()]
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        shown = ", ".join(str(p.relative_to(root)) for p in matches[:5])
        return f"'{rel}' matches more than one file ({shown}). Pass the path from the documents folder."
    return f"No such file: {rel} (looked inside {root})."


def make_query_json_tool(home: Path) -> Tool:
    def query_json(path: str = "", query: str = "", fields=None, max_results: int = MAX_RESULTS) -> str:
        root = docs_root(home)
        if not root.is_dir():
            return (f"No documents folder yet — expected {root}. Create it (or point "
                    f"{DOCS_DIR_ENV} at a folder you already have) and put files there.")
        resolved = _resolve_json(root, path)
        if isinstance(resolved, str):
            return resolved
        try:
            data = json.loads(resolved.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError) as exc:
            return f"Could not read {resolved}: {exc}."
        except json.JSONDecodeError as exc:
            return f"{resolved.relative_to(root)} is not valid JSON ({exc})."

        if not (query or "").strip():
            hints = _array_hints(data)
            hint = (" Arrays of objects: " + ", ".join(hints) + "." if hints else "")
            return (f"{resolved.relative_to(root)} loaded. Pass a query such as "
                    f"{_EXAMPLES[0]!r}.{hint}")

        try:
            matches = run(data, query)
        except QueryError as exc:
            return f"Nothing returned — {exc}."

        try:
            limit = max(1, min(int(max_results or MAX_RESULTS), MAX_RESULTS_CEILING))
        except (TypeError, ValueError):
            limit = MAX_RESULTS
        columns = _coerce_fields(fields)
        projected = [_project(item, columns) for item in matches]
        shown = projected[:limit]
        rel = resolved.relative_to(root)
        if not shown:
            hints = _array_hints(data)
            hint = ""
            if hints:
                hint = (" Arrays of objects in this file: " + ", ".join(hints)
                        + ". Recursive descent: ..jobsList[]")
            return f"No matches for {query!r} in {rel}.{hint}"

        head = (f"First {len(shown)} of {len(matches)} match(es) in {rel} for {query!r}:"
                if len(matches) > len(shown)
                else f"{len(shown)} match(es) in {rel} for {query!r}:")
        lines = [head]
        for item in shown:
            lines.append("- " + _format_item(item))
        return "\n".join(lines)

    return Tool(
        name="query_json",
        description=(
            "Filter a saved JSON file in the documents folder with a jq/JSONPath "
            "subset. Use this instead of search_documents when the question is about "
            "fields in a listing (postedDate, area, title), not a keyword sitting on "
            "a line. A company crawl's day's index is "
            "jobs/<company>/YYYY-MM-DD/manifest-<company>-YYYY-MM-DD.json "
            "(YYYY-MM-DD is the Pacific Time calendar day) — query "
            "'..roles[]' (or select on title/req_id) instead of grepping HTML. "
            "Examples: '..jobsList[] | select(.postedDate == today)', "
            "'..jobsList[] | select(.area | contains(\"Technology\"))', "
            "'$..jobsList[?(@.postedDate==today)]', '..roles[]'. today/yesterday match ISO and "
            "US dates (2026-09-18 and 09/18/2026). Pass fields to keep only those "
            "keys in the reply. This is not full jq — map(), functions and "
            "assignments are refused with examples. path is relative to the "
            "documents folder (e.g. 'jobs/jpmorgan/2026-09-18/manifest-jpmorgan-2026-09-18.json')."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": ("JSON file inside the documents folder, e.g. "
                                    "'jobs/jpmorgan/2026-09-18/manifest-jpmorgan-2026-09-18.json'. "
                                    "Omit to list JSON files."),
                },
                "query": {
                    "type": "string",
                    "description": (
                        "jq/JSONPath subset: ..jobsList[] | select(.postedDate == today)"
                    ),
                },
                "fields": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Keys to keep on each matching object, e.g. postingTitle, jcrURL",
                },
                "max_results": {
                    "type": "integer",
                    "description": (
                        f"Max matches to return (default {MAX_RESULTS}, "
                        f"ceiling {MAX_RESULTS_CEILING})"
                    ),
                },
            },
            "required": [],
        },
        fn=query_json,
    )
