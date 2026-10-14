"""Load and compare promptfoo eval runs from the local promptfoo SQLite DB.

The DB is never written to: `snapshot()` copies it into memory and all analysis runs on the copy.
"""

import hashlib
import json
import os
import re
import sqlite3
from datetime import datetime
from pathlib import Path

import pandas as pd

# ResultFailureReason in src/types/results.ts: 0 = passed/unknown, 1 = assertion failed, 2 = error.
FAILURE_ERROR = 2
OUTPUT_MAX_CHARS = 2048
# Provider config keys that don't describe how the model ran: where the config came from, and
# id/label that some older configs put inside `config` by mistake.
IGNORED_CONFIG_KEYS = {"basePath", "id", "label"}
# Second id segment that names an API type rather than part of the model name.
API_TYPES = {
    "chat",
    "completion",
    "completions",
    "embedding",
    "embeddings",
    "responses",
}


def default_db_path():
    """$PROMPTFOO_CONFIG_DIR/promptfoo.db, or ~/.promptfoo/promptfoo.db."""
    config_dir = os.environ.get("PROMPTFOO_CONFIG_DIR") or Path.home() / ".promptfoo"
    return Path(config_dir).expanduser() / "promptfoo.db"


def snapshot(db_path=None):
    """Return an in-memory copy of the promptfoo DB.

    The DB is in WAL mode, so recent writes may still sit in promptfoo.db-wal. The source is
    opened read-only and copied with the SQLite backup API, which includes WAL contents and
    gives a consistent snapshot even while promptfoo is writing.
    """
    path = Path(db_path or default_db_path()).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"promptfoo DB not found: {path}")
    source = sqlite3.connect(f"{path.as_uri()}?mode=ro", uri=True)
    try:
        conn = sqlite3.connect(":memory:")
        source.backup(conn)
    finally:
        source.close()
    return conn


def model_name(provider_id):
    """'ollama:chat:qwen3-vl:4b' -> 'qwen3-vl:4b'; 'openai:gpt-4o' -> 'gpt-4o'.

    Same rule as modelName() in src/util/configDisplay.ts.
    """
    parts = str(provider_id).split(":")
    if len(parts) >= 3 and parts[1] in API_TYPES:
        return ":".join(parts[2:])
    if len(parts) >= 2:
        return ":".join(parts[1:])
    return str(provider_id)


def short_id(eval_id):
    """'eval-nUh-2026-10-14T07:01:00' -> 'nUh'."""
    parts = str(eval_id).split("-")
    return parts[1] if len(parts) > 2 and parts[0] == "eval" else str(eval_id)


def _json(text):
    if text is None:
        return None
    try:
        return json.loads(text)
    except (TypeError, ValueError):
        return None


def _scalar(value):
    """Keep scalars as-is; lists and dicts become compact JSON so they fit in one cell."""
    if isinstance(value, (list, dict)):
        return json.dumps(value, sort_keys=True)
    return value


def flatten(obj, prefix=""):
    """{'a': {'b': 1}, 'c': 2} -> {'a.b': 1, 'c': 2}."""
    out = {}
    for key, value in (obj or {}).items():
        name = f"{prefix}{key}"
        if isinstance(value, dict):
            out.update(flatten(value, f"{name}."))
        else:
            out[name] = _scalar(value)
    return out


def make_test_key(vars_):
    """Stable short hash of a test's vars, used to pair the same test across columns."""
    return hashlib.sha1(json.dumps(vars_ or {}, sort_keys=True).encode()).hexdigest()[
        :12
    ]


def _to_local(ms):
    """Epoch milliseconds -> naive local datetimes."""
    local = datetime.now().astimezone().tzinfo
    return (
        pd.to_datetime(ms, unit="ms", utc=True)
        .dt.tz_convert(local)
        .dt.tz_localize(None)
    )


def _disambiguate(titles, eval_ids):
    """Append the short eval id to titles that appear more than once."""
    counts = titles.value_counts()
    return [
        f"{t} [{short_id(e)}]" if counts.get(t, 0) > 1 else t
        for t, e in zip(titles, eval_ids)
    ]


def list_runs(conn):
    """One row per eval, newest first: eval_id, created_at, title, n_results, models."""
    rows = conn.execute(
        """
        SELECT e.id, e.created_at, e.description,
               COUNT(r.id) AS n_results,
               GROUP_CONCAT(DISTINCT json_extract(r.provider, '$.id')) AS providers
        FROM evals e LEFT JOIN eval_results r ON r.eval_id = e.id
        GROUP BY e.id
        ORDER BY e.created_at DESC
        """
    ).fetchall()
    df = pd.DataFrame(
        rows, columns=["eval_id", "created_at", "title", "n_results", "providers"]
    )
    df["created_at"] = _to_local(df["created_at"])
    df["title"] = df["title"].fillna(df["eval_id"])
    df["models"] = df["providers"].map(
        lambda s: (
            " ".join(sorted({model_name(p) for p in s.split(",")}))
            if isinstance(s, str)
            else ""
        )
    )
    return df.drop(columns="providers")


def select_runs(conn, eval_ids=None, title_filter=None, last_n=None):
    """The runs to analyze, newest first, as rows of list_runs().

    `eval_ids` wins if given. Otherwise: runs that have results, optionally narrowed by the
    case-insensitive regex `title_filter`, then the newest `last_n` (None = all).
    """
    runs = list_runs(conn)
    if eval_ids:
        missing = set(eval_ids) - set(runs["eval_id"])
        if missing:
            raise ValueError(f"Unknown eval ids: {sorted(missing)}")
        return runs[runs["eval_id"].isin(eval_ids)]
    runs = runs[runs["n_results"] > 0]
    if title_filter:
        runs = runs[
            runs["title"].str.contains(title_filter, flags=re.IGNORECASE, regex=True)
        ]
    return runs if last_n is None else runs.head(last_n)


def select_eval_ids(conn, eval_ids=None, title_filter=None, last_n=None):
    """Eval ids of select_runs()."""
    return select_runs(conn, eval_ids, title_filter, last_n)["eval_id"].tolist()


RESULT_COLUMNS = [
    "eval_id",
    "title",
    "created_at",
    "column_id",
    "prompt_idx",
    "prompt_label",
    "provider_id",
    "model",
    "provider_label",
    "test_idx",
    "test_key",
    "success",
    "score",
    "failure_reason",
    "is_error",
    "error",
    "cost",
    "latency_ms",
    "tokens_prompt",
    "tokens_completion",
    "tokens_total",
    "output",
]


def load_results(conn, eval_ids=None, title_filter=None, last_n=None):
    """One row per result across the runs chosen by select_runs().

    Fixed columns are RESULT_COLUMNS. Variable columns:
      cfg_<key>  provider config (nested keys joined with '.', basePath dropped)
      var_<name> test vars
      ns_<name>  named scores
    """
    ids = select_eval_ids(conn, eval_ids, title_filter, last_n)
    if not ids:
        return pd.DataFrame(columns=RESULT_COLUMNS)
    marks = ",".join("?" * len(ids))
    cursor = conn.cursor()
    cursor.row_factory = sqlite3.Row
    rows = cursor.execute(
        f"""
        SELECT r.eval_id, e.description AS title, e.created_at, r.prompt_idx, r.test_idx,
               r.test_case, r.prompt, r.provider, r.success, r.score, r.named_scores,
               r.cost, r.latency_ms, r.response, r.error, r.failure_reason
        FROM eval_results r JOIN evals e ON e.id = r.eval_id
        WHERE r.eval_id IN ({marks})
        ORDER BY e.created_at, r.eval_id, r.prompt_idx, r.test_idx
        """,
        ids,
    ).fetchall()

    records = []
    for row in rows:
        eval_id, prompt_idx, failure_reason = (
            row["eval_id"],
            row["prompt_idx"],
            row["failure_reason"],
        )
        test_case = _json(row["test_case"]) or {}
        prompt = _json(row["prompt"]) or {}
        provider = _json(row["provider"]) or {}
        response = _json(row["response"]) or {}
        usage = response.get("tokenUsage") or {}
        vars_ = test_case.get("vars") or {}
        config = {
            k: v
            for k, v in (provider.get("config") or {}).items()
            if k not in IGNORED_CONFIG_KEYS
        }
        raw_config = provider.get("config") or {}
        provider_id = provider.get("id", "")
        # Older configs put the label inside `config`; use it when the real one is missing.
        provider_label = provider.get("label") or raw_config.get("label") or None
        label = provider_label or model_name(provider_id)
        output = response.get("output")
        if output is not None and not isinstance(output, str):
            output = json.dumps(output)

        record = {
            "eval_id": eval_id,
            "title": row["title"] or eval_id,
            "created_at": row["created_at"],
            "column_id": f"{short_id(eval_id)} p{prompt_idx} {label}",
            "prompt_idx": prompt_idx,
            "prompt_label": prompt.get("label") or prompt.get("raw"),
            "provider_id": provider_id,
            "model": model_name(provider_id),
            "provider_label": provider_label,
            "test_idx": row["test_idx"],
            "test_key": make_test_key(vars_),
            "success": bool(row["success"]),
            "score": row["score"],
            "failure_reason": failure_reason,
            "is_error": failure_reason == FAILURE_ERROR,
            "error": row["error"],
            "cost": row["cost"],
            "latency_ms": row["latency_ms"],
            "tokens_prompt": usage.get("prompt"),
            "tokens_completion": usage.get("completion"),
            "tokens_total": usage.get("total"),
            "output": output[:OUTPUT_MAX_CHARS] if output else output,
        }
        record.update({f"cfg_{k}": v for k, v in flatten(config).items()})
        record.update({f"var_{k}": _scalar(v) for k, v in vars_.items()})
        record.update(
            {f"ns_{k}": v for k, v in (_json(row["named_scores"]) or {}).items()}
        )
        records.append(record)

    df = pd.DataFrame.from_records(records)
    df["created_at"] = _to_local(df["created_at"])
    for col in [
        "score",
        "cost",
        "latency_ms",
        "tokens_prompt",
        "tokens_completion",
        "tokens_total",
    ]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    runs = df.drop_duplicates("eval_id")
    unique_titles = dict(
        zip(runs["eval_id"], _disambiguate(runs["title"], runs["eval_id"]))
    )
    df["title"] = df["eval_id"].map(unique_titles)
    return df


def factor_columns(df):
    """Columns that describe how a column ran and vary across the loaded rows:
    model, prompt_label and cfg_*."""
    candidates = [
        "model",
        "prompt_label",
        *sorted(c for c in df.columns if c.startswith("cfg_")),
    ]
    return [c for c in candidates if c in df and df[c].nunique(dropna=False) > 1]
