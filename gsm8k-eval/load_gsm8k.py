import json
from pathlib import Path

DATA = Path(__file__).resolve().parent / "test.jsonl"
DEFAULT_N = 3


def create_tests(config=None):
    """Load GSM8K rows [offset, offset + n) as promptfoo test cases.

    `config` comes from the `tests: {path: ..., config: {n, offset}}` form in the
    promptfoo config.
    """
    config = config or {}
    n = int(config.get("n", DEFAULT_N))
    offset = int(config.get("offset", 0))
    tests = []
    with open(DATA) as f:
        for i, line in enumerate(f):
            if i < offset:
                continue
            if i >= offset + n:
                break
            row = json.loads(line)
            expected = row["answer"].split("####")[-1].strip().replace(",", "")
            tests.append({
                "description": f"gsm8k-{i}",
                "vars": {"question": row["question"], "expected": expected},
                "metadata": {"gsm8k_idx": i},
            })
    return tests
