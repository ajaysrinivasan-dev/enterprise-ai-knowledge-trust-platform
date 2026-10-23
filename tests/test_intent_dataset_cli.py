"""Tests for the read-only intent dataset validation CLI."""

import json

from scripts.validate_intent_dataset import main, validate_dataset


def test_cli_reports_counts_and_readiness_without_text(tmp_path, capsys, monkeypatch) -> None:
    path = tmp_path / "annotated.jsonl"
    phrases = {
        "question": ["explain the policy", "what does the rule mean", "answer the requirement"],
        "search": ["locate the handbook clause", "find the approval section", "search the archive"],
        "sql_report": ["count indexed files", "group records by type", "show the report totals"],
        "email_draft": [
            "compose a review note",
            "prepare an owner message",
            "write a status update",
        ],
        "verify": ["check the citation", "validate the claim", "review the supplied answer"],
    }
    rows = [
        {
            "id": f"{intent}-{index}",
            "group_id": f"{intent}-{index}",
            "text": phrases[intent][index],
            "intent": intent,
        }
        for intent in ("question", "search", "sql_report", "email_draft", "verify")
        for index in range(3)
    ]
    path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
    original = path.read_bytes()
    monkeypatch.setattr("sys.argv", ["validate_intent_dataset", str(path)])

    assert main() == 0
    output = capsys.readouterr().out
    report = json.loads(output)

    assert report["example_count"] == 15
    assert report["ready"] is True
    assert report["split_counts"]["validation"] > 0
    assert "distinct request" not in output
    assert path.read_bytes() == original


def test_cli_reports_validation_failure_without_raw_content(tmp_path) -> None:
    path = tmp_path / "invalid.jsonl"
    path.write_text(
        '{"id":"same","group_id":"g1","text":"private value","intent":"question"}\n'
        '{"id":"same","group_id":"g2","text":"other value","intent":"question"}\n',
        encoding="utf-8",
    )

    report = validate_dataset(path)

    assert report["ready"] is False
    assert report["example_count"] == 0
    assert report["errors"] == ["dataset could not be loaded or failed validation"]
    assert "private value" not in json.dumps(report)
