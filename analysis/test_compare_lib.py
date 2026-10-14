import json
import sqlite3

import pandas as pd
import pytest
from compare_lib import (
    OUTPUT_MAX_CHARS,
    default_db_path,
    factor_columns,
    flatten,
    list_runs,
    load_results,
    make_test_key,
    model_name,
    select_eval_ids,
    short_id,
    snapshot,
)

# Subset of promptfoo's real schema (src/database/tables.ts): only the columns the lib reads.
SCHEMA = """
CREATE TABLE evals (
    id TEXT PRIMARY KEY, created_at INTEGER NOT NULL, description TEXT, config TEXT
);
CREATE TABLE eval_results (
    id TEXT PRIMARY KEY, eval_id TEXT NOT NULL, prompt_idx INTEGER, test_idx INTEGER,
    test_case TEXT, prompt TEXT, provider TEXT, latency_ms INTEGER, cost REAL,
    response TEXT, error TEXT, success INTEGER, score REAL, named_scores TEXT,
    failure_reason INTEGER
);
"""

E1 = "eval-AAA-2026-10-14T10:00:00"
E2 = "eval-BBB-2026-10-14T11:00:00"
E3 = "eval-CCC-2026-10-14T12:00:00"
Q1 = {"question": "1+1?", "expected": "2"}
Q2 = {"question": "2+2?", "expected": "4"}
Q3 = {"question": "only in run 3", "expected": "0"}


def provider(model, temperature, **extra):
    return {
        "id": f"ollama:chat:{model}",
        "label": f"id={model} temperature={temperature}",
        "config": {"basePath": "/x", "temperature": temperature, "seed": 512, **extra},
    }


def result(
    rid, eval_id, test_idx, vars_, prov, *, success=True, failure=0, output="2", **kw
):
    return {
        "id": rid,
        "eval_id": eval_id,
        "prompt_idx": kw.get("prompt_idx", 0),
        "test_idx": test_idx,
        "test_case": json.dumps({"vars": vars_}),
        "prompt": json.dumps(
            {"raw": "Answer: {{question}}", "label": "Answer: {{question}}"}
        ),
        "provider": json.dumps(prov),
        "latency_ms": kw.get("latency_ms", 100),
        "cost": kw.get("cost"),
        "response": json.dumps(
            {
                "output": output,
                "tokenUsage": {"prompt": 10, "completion": 5, "total": 15},
            }
        ),
        "error": kw.get("error"),
        "success": int(success),
        "score": 1.0 if success else 0.0,
        "named_scores": json.dumps(kw.get("named_scores", {})),
        "failure_reason": failure,
    }


ROWS = [
    result("r1", E1, 0, Q1, provider("qwen3-vl:4b", 0.2), named_scores={"accuracy": 1}),
    result(
        "r2",
        E1,
        1,
        Q2,
        provider("qwen3-vl:4b", 0.2),
        success=False,
        failure=1,
        output="5",
    ),
    result("r3", E1, 0, Q1, provider("qwen3.5:4b", 0.8), output={"answer": 2}),
    result(
        "r4",
        E1,
        1,
        Q2,
        provider("qwen3.5:4b", 0.8),
        success=False,
        failure=2,
        error="timeout",
        output=None,
        latency_ms=None,
    ),
    result(
        "r5", E2, 0, Q1, provider("qwen3-vl:4b", 0.2, options={"top_p": 0.9}), cost=0.01
    ),
    result("r6", E3, 0, Q3, provider("qwen3-vl:4b", 0.2), output="x" * 5000),
]


def build_db(path, rows=ROWS):
    conn = sqlite3.connect(path)
    conn.executescript(SCHEMA)
    conn.executemany(
        "INSERT INTO evals VALUES (?, ?, ?, ?)",
        [
            (E1, 1_760_436_000_000, "GSM8k sweep", "{}"),
            (E2, 1_760_439_600_000, "GSM8k sweep", "{}"),  # duplicate title
            (E3, 1_760_443_200_000, None, "{}"),  # no title
        ],
    )
    cols = list(rows[0])
    conn.executemany(
        f"INSERT INTO eval_results ({', '.join(cols)}) VALUES ({', '.join('?' * len(cols))})",
        [tuple(r[c] for c in cols) for r in rows],
    )
    conn.commit()
    return conn


@pytest.fixture
def db_path(tmp_path):
    path = tmp_path / "promptfoo.db"
    build_db(path).close()
    return path


@pytest.fixture
def conn(db_path):
    return snapshot(db_path)


def test_model_name():
    assert model_name("ollama:chat:qwen3-vl:4b") == "qwen3-vl:4b"
    assert model_name("openai:gpt-4o") == "gpt-4o"
    assert model_name("echo") == "echo"


def test_short_id():
    assert short_id(E1) == "AAA"
    assert short_id("custom") == "custom"


def test_flatten_nested_and_lists():
    assert flatten({"a": {"b": 1}, "c": [1, 2], "d": None}) == {
        "a.b": 1,
        "c": "[1, 2]",
        "d": None,
    }


def test_make_test_key_ignores_key_order():
    assert make_test_key({"a": 1, "b": 2}) == make_test_key({"b": 2, "a": 1})
    assert make_test_key(Q1) != make_test_key(Q2)


def test_default_db_path_respects_env(monkeypatch, tmp_path):
    monkeypatch.setenv("PROMPTFOO_CONFIG_DIR", str(tmp_path))
    assert default_db_path() == tmp_path / "promptfoo.db"
    monkeypatch.delenv("PROMPTFOO_CONFIG_DIR")
    assert default_db_path().name == "promptfoo.db"


def test_snapshot_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        snapshot(tmp_path / "nope.db")


def test_snapshot_does_not_write_source(db_path):
    before = db_path.read_bytes()
    conn = snapshot(db_path)
    conn.execute("DELETE FROM eval_results")
    assert db_path.read_bytes() == before


def test_snapshot_includes_uncheckpointed_wal(tmp_path):
    path = tmp_path / "promptfoo.db"
    writer = sqlite3.connect(path)
    writer.execute("PRAGMA journal_mode=WAL")
    writer.execute("PRAGMA wal_autocheckpoint=0")  # keep new rows in the -wal file only
    writer.executescript(SCHEMA)
    writer.execute("INSERT INTO evals VALUES (?, 1, 't', '{}')", (E1,))
    writer.commit()
    assert (tmp_path / "promptfoo.db-wal").stat().st_size > 0
    conn = snapshot(path)  # writer still open, as when promptfoo is running
    assert conn.execute("SELECT id FROM evals").fetchall() == [(E1,)]
    writer.close()


def test_list_runs(conn):
    runs = list_runs(conn)
    assert list(runs["eval_id"]) == [E3, E2, E1]  # newest first
    first = runs.set_index("eval_id").loc[E1]
    assert first["n_results"] == 4
    assert first["models"] == "qwen3-vl:4b qwen3.5:4b"
    assert (
        runs.set_index("eval_id").loc[E3, "title"] == E3
    )  # missing title falls back to id
    assert pd.api.types.is_datetime64_any_dtype(runs["created_at"])


def test_list_runs_with_no_results(tmp_path):
    path = tmp_path / "promptfoo.db"
    build_db(path, [result("r1", E1, 0, Q1, provider("m", 0.1))]).close()
    runs = list_runs(snapshot(path)).set_index("eval_id")
    assert runs.loc[E2, "n_results"] == 0
    assert runs.loc[E2, "models"] == ""


def test_select_eval_ids(conn):
    assert select_eval_ids(conn, eval_ids=[E2]) == [E2]
    assert select_eval_ids(conn, title_filter="gsm8k") == [E2, E1]  # newest first
    assert select_eval_ids(conn) == [E3, E2, E1]
    with pytest.raises(ValueError):
        select_eval_ids(conn, eval_ids=["eval-nope"])


def test_last_n_takes_newest_runs(conn):
    assert select_eval_ids(conn, last_n=2) == [E3, E2]
    assert select_eval_ids(conn, title_filter="gsm8k", last_n=1) == [E2]
    assert select_eval_ids(conn, eval_ids=[E1], last_n=1) == [E1]  # explicit ids win
    df = load_results(conn, last_n=2)
    assert set(df["eval_id"]) == {E2, E3}
    assert len(df) == 2


def test_last_n_skips_runs_without_results(tmp_path):
    path = tmp_path / "promptfoo.db"
    build_db(path, [result("r1", E1, 0, Q1, provider("m", 0.1))]).close()
    assert select_eval_ids(snapshot(path), last_n=1) == [
        E1
    ]  # E2, E3 are newer but empty


def test_load_results_columns(conn):
    df = load_results(conn, eval_ids=[E1])
    assert len(df) == 4
    row = df.iloc[0]
    assert row["column_id"] == "AAA p0 id=qwen3-vl:4b temperature=0.2"
    assert row["model"] == "qwen3-vl:4b"
    assert row["prompt_label"] == "Answer: {{question}}"
    assert row["cfg_temperature"] == 0.2
    assert "cfg_basePath" not in df
    assert row["var_question"] == "1+1?"
    assert row["test_key"] == make_test_key(Q1)
    assert row["tokens_total"] == 15
    assert row["ns_accuracy"] == 1


def test_load_results_errors_and_nulls(conn):
    df = load_results(conn, eval_ids=[E1]).set_index("test_idx", append=True)
    err = df[df["is_error"]]
    assert len(err) == 1
    assert err.iloc[0]["error"] == "timeout"
    assert pd.isna(err.iloc[0]["output"])
    assert pd.isna(err.iloc[0]["latency_ms"])
    assert df["cost"].isna().all()
    assert (df["failure_reason"] == 1).sum() == 1  # assertion failure is not an error


def test_load_results_non_string_and_long_output(conn):
    df = load_results(conn)
    r3 = df[
        (df["eval_id"] == E1) & (df["model"] == "qwen3.5:4b") & (df["test_idx"] == 0)
    ]
    assert r3["output"].iloc[0] == '{"answer": 2}'
    assert len(df.loc[df["eval_id"] == E3, "output"].iloc[0]) == OUTPUT_MAX_CHARS


def test_load_results_nested_config_and_duplicate_titles(conn):
    df = load_results(conn, title_filter="gsm8k")
    assert df.loc[df["eval_id"] == E2, "cfg_options.top_p"].iloc[0] == 0.9
    titles = df.drop_duplicates("eval_id").set_index("eval_id")["title"]
    assert titles[E1] == "GSM8k sweep [AAA]"
    assert titles[E2] == "GSM8k sweep [BBB]"


def test_load_results_unique_title_untouched(conn):
    assert set(load_results(conn, eval_ids=[E1])["title"]) == {"GSM8k sweep"}


def test_load_results_no_runs(conn):
    df = load_results(conn, title_filter="no such run")
    assert df.empty
    assert "column_id" in df


def test_label_inside_config_is_used_as_label_not_factor(tmp_path):
    prov = {
        "id": "ollama:chat:m",
        "label": "",
        "config": {"label": "temp_0.1", "temperature": 0.1},
    }
    path = tmp_path / "promptfoo.db"
    build_db(path, [result("r1", E1, 0, Q1, prov)]).close()
    df = load_results(snapshot(path))
    assert df.iloc[0]["provider_label"] == "temp_0.1"
    assert df.iloc[0]["column_id"] == "AAA p0 temp_0.1"
    assert "cfg_label" not in df


def test_factor_columns(conn):
    df = load_results(conn, eval_ids=[E1])
    assert factor_columns(df) == [
        "model",
        "cfg_temperature",
    ]  # seed and prompt don't vary
