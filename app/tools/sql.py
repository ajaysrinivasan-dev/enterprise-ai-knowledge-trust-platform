"""Allowlisted read-only SQL reporting."""

import re
import sqlite3
from dataclasses import dataclass


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
    }

    def __init__(self, database_path: str, max_rows: int = 100) -> None:
        self.database_path = database_path
        self.max_rows = max_rows

    def run(self, report_name: str) -> SqlReport:
        """Run an allowlisted report; report names are not SQL input."""

        query = self._REPORTS.get(report_name)
        if query is None or not re.fullmatch(r"[a-z_]+", report_name):
            raise ValueError("report is not allowlisted")
        connection = sqlite3.connect(f"file:{self.database_path}?mode=ro", uri=True)
        try:
            cursor = connection.execute(query)
            rows = tuple(cursor.fetchmany(self.max_rows))
            return SqlReport(tuple(description[0] for description in cursor.description), rows)
        finally:
            connection.close()

