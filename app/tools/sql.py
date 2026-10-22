"""Allowlisted read-only SQL reporting."""

import re
import sqlite3
from dataclasses import dataclass
from time import monotonic


@dataclass(frozen=True)
class SqlReport:
    """Safe SQL output with bounded rows."""

    columns: tuple[str, ...]
    rows: tuple[tuple[object, ...], ...]


class ReadOnlySqlTool:
    """Execute only predefined reports against a SQLite database."""

    _REPORTS = {
        "document_counts": (
            "SELECT media_type, COUNT(*) AS count "
            "FROM documents GROUP BY media_type ORDER BY media_type"
        ),
        "documents_by_type": (
            "SELECT media_type, COUNT(*) AS count FROM documents "
            "WHERE media_type = ? GROUP BY media_type"
        ),
    }

    def __init__(
        self, database_path: str, max_rows: int = 100, timeout_seconds: float = 2.0
    ) -> None:
        self.database_path = database_path
        self.max_rows = max_rows
        self.timeout_seconds = timeout_seconds

    def run(self, report_name: str, params: tuple[object, ...] = ()) -> SqlReport:
        """Run an allowlisted report; report names are not SQL input."""

        query = self._REPORTS.get(report_name)
        if query is None or not re.fullmatch(r"[a-z_]+", report_name):
            raise ValueError("report is not allowlisted")
        if report_name == "documents_by_type" and len(params) != 1:
            raise ValueError("documents_by_type requires exactly one media type")
        if report_name == "document_counts" and params:
            raise ValueError("document_counts does not accept parameters")
        connection = sqlite3.connect(
            f"file:{self.database_path}?mode=ro",
            uri=True,
            timeout=self.timeout_seconds,
        )
        try:
            deadline = monotonic() + self.timeout_seconds
            connection.set_progress_handler(
                lambda: 1 if monotonic() > deadline else 0, 1000
            )
            cursor = connection.execute(query, params)
            rows = tuple(cursor.fetchmany(self.max_rows))
            return SqlReport(tuple(description[0] for description in cursor.description), rows)
        finally:
            connection.close()
