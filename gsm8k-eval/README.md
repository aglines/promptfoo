# gsm8k-eval

A GSM8K eval suite for promptfoo.

- `load_gsm8k.py:create_tests`: test generator. It takes `n` and `offset` from the generator's `config` (default `n=5`).
- `numeric_match.py`: grader. It compares the last number in the output with the expected answer.
- `test.jsonl`: the GSM8K test split.
- `promptfooconfig.yaml`: a standalone config for running the suite directly.

To run sweeps, compare runs, or use the web UI, see `../eval-lab/README.md`.
