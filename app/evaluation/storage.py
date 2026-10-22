"""SQLite persistence for evaluation runs."""

import json
import sqlite3

from app.evaluation.models import EvaluationRun


class EvaluationStore:
    """Persist evaluation metadata and metrics without document content."""

    def __init__(self, database_path: str) -> None:
        self.database_path = database_path
        with sqlite3.connect(database_path) as connection:
            connection.execute(
                "CREATE TABLE IF NOT EXISTS evaluation_runs "
                "(run_id TEXT PRIMARY KEY, dataset_version TEXT NOT NULL, "
                "created_at TEXT NOT NULL, metrics_json TEXT NOT NULL, errors_json TEXT NOT NULL)"
            )

    def save(self, run: EvaluationRun) -> None:
        """Insert or replace a run by its stable ID."""

        with sqlite3.connect(self.database_path) as connection:
            connection.execute(
                "INSERT OR REPLACE INTO evaluation_runs VALUES (?, ?, ?, ?, ?)",
                (
                    run.run_id,
                    run.dataset_version,
                    run.created_at.isoformat(),
                    json.dumps(run.metrics, sort_keys=True),
                    json.dumps(run.errors),
                ),
            )

    def list_runs(self) -> list[EvaluationRun]:
        """Return stored runs ordered by creation time."""

        with sqlite3.connect(self.database_path) as connection:
            rows = connection.execute(
                "SELECT run_id, dataset_version, created_at, metrics_json, errors_json "
                "FROM evaluation_runs ORDER BY created_at DESC"
            ).fetchall()
        from datetime import datetime

        return [
            EvaluationRun(
                run_id=row[0],
                dataset_version=row[1],
                created_at=datetime.fromisoformat(row[2]),
                metrics=json.loads(row[3]),
                errors=tuple(json.loads(row[4])),
            )
            for row in rows
        ]
