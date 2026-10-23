"""SQLite persistence for evaluation runs."""

import json
import sqlite3
from datetime import datetime

from app.evaluation.models import EvaluationConfig, EvaluationRun, ExampleOutcome


class EvaluationStore:
    """Persist evaluation metadata and metrics without document content."""

    def __init__(self, database_path: str) -> None:
        self.database_path = database_path
        with sqlite3.connect(database_path) as connection:
            connection.execute(
                "CREATE TABLE IF NOT EXISTS evaluation_runs "
                "(run_id TEXT PRIMARY KEY, dataset_version TEXT NOT NULL, "
                "created_at TEXT NOT NULL, metrics_json TEXT NOT NULL, errors_json TEXT NOT NULL, "
                "dataset_fingerprint TEXT NOT NULL DEFAULT '', "
                "dataset_status TEXT NOT NULL DEFAULT 'synthetic_smoke')"
            )
            columns = {
                row[1] for row in connection.execute("PRAGMA table_info(evaluation_runs)")
            }
            if "config_json" not in columns:
                connection.execute("ALTER TABLE evaluation_runs ADD COLUMN config_json TEXT")
            if "outcomes_json" not in columns:
                connection.execute("ALTER TABLE evaluation_runs ADD COLUMN outcomes_json TEXT")
            if "duration_ms" not in columns:
                connection.execute("ALTER TABLE evaluation_runs ADD COLUMN duration_ms REAL")
            if "dataset_fingerprint" not in columns:
                connection.execute(
                    "ALTER TABLE evaluation_runs ADD COLUMN "
                    "dataset_fingerprint TEXT NOT NULL DEFAULT ''"
                )
            if "dataset_status" not in columns:
                connection.execute(
                    "ALTER TABLE evaluation_runs ADD COLUMN "
                    "dataset_status TEXT NOT NULL DEFAULT 'synthetic_smoke'"
                )

    def save(self, run: EvaluationRun) -> None:
        """Insert or replace a run by its stable ID."""

        with sqlite3.connect(self.database_path) as connection:
            connection.execute(
                "INSERT OR REPLACE INTO evaluation_runs "
                "(run_id, dataset_version, created_at, metrics_json, errors_json, "
                "config_json, outcomes_json, duration_ms, dataset_fingerprint, dataset_status) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    run.run_id,
                    run.dataset_version,
                    run.created_at.isoformat(),
                    json.dumps(run.metrics, sort_keys=True),
                    json.dumps(run.errors),
                    json.dumps(run.config.__dict__, sort_keys=True),
                    json.dumps([outcome.__dict__ for outcome in run.outcomes], sort_keys=True),
                    run.duration_ms,
                    run.dataset_fingerprint,
                    run.dataset_status,
                ),
            )

    def list_runs(self) -> list[EvaluationRun]:
        """Return stored runs ordered by creation time."""

        with sqlite3.connect(self.database_path) as connection:
            rows = connection.execute(
                "SELECT run_id, dataset_version, created_at, metrics_json, errors_json, "
                "config_json, outcomes_json, duration_ms, dataset_fingerprint, "
                "dataset_status FROM evaluation_runs ORDER BY created_at DESC"
            ).fetchall()
        return [
            EvaluationRun(
                run_id=row[0],
                dataset_version=row[1],
                created_at=datetime.fromisoformat(row[2]),
                metrics=json.loads(row[3]),
                dataset_fingerprint=row[8],
                dataset_status=row[9],
                errors=tuple(json.loads(row[4])),
                config=EvaluationConfig(**json.loads(row[5])) if row[5] else EvaluationConfig(),
                outcomes=tuple(
                    ExampleOutcome(**item)
                    for item in (json.loads(row[6]) if row[6] else [])
                ),
                duration_ms=row[7] or 0.0,
            )
            for row in rows
        ]
