"""Small HTTP client used by Streamlit instead of duplicating backend logic."""

from typing import Any

import httpx


class ApiClient:
    """Call the local FastAPI backend with explicit, bounded timeouts."""

    def __init__(self, base_url: str = "http://127.0.0.1:8000", api_key: str | None = None) -> None:
        self.base_url = base_url.rstrip("/")
        self.headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}

    def list_documents(self) -> list[dict[str, Any]]:
        response = httpx.get(f"{self.base_url}/api/v1/documents", headers=self.headers, timeout=10)
        response.raise_for_status()
        return response.json()

    def upload_document(self, filename: str, content: bytes) -> dict[str, Any]:
        response = httpx.post(
            f"{self.base_url}/api/v1/documents",
            files={"file": (filename, content)},
            headers=self.headers,
            timeout=30,
        )
        response.raise_for_status()
        return response.json()

    def ask(self, question: str) -> dict[str, Any]:
        response = httpx.post(
            f"{self.base_url}/api/v1/questions",
            json={"question": question},
            headers=self.headers,
            timeout=60,
        )
        response.raise_for_status()
        return response.json()

    def run_evaluation(self, dataset_id: str = "synthetic_v1") -> dict[str, Any]:
        """Run an approved offline evaluation dataset."""

        response = httpx.post(
            f"{self.base_url}/api/v1/evaluations",
            json={"dataset_id": dataset_id},
            headers=self.headers,
            timeout=60,
        )
        response.raise_for_status()
        return response.json()

    def list_evaluations(self) -> list[dict[str, Any]]:
        """List stored evaluation results."""

        response = httpx.get(
            f"{self.base_url}/api/v1/evaluations", headers=self.headers, timeout=10
        )
        response.raise_for_status()
        return response.json()

    def training_preflight(self) -> dict[str, Any]:
        """Retrieve runtime hardware and package availability."""

        response = httpx.get(
            f"{self.base_url}/api/v1/training/preflight", headers=self.headers, timeout=10
        )
        response.raise_for_status()
        return response.json()

    def training_datasets(self) -> list[dict[str, Any]]:
        """List approved training datasets."""

        response = httpx.get(
            f"{self.base_url}/api/v1/training/datasets", headers=self.headers, timeout=10
        )
        response.raise_for_status()
        return response.json()

    def start_training(
        self, dataset_id: str, base_model: str, method: str, confirm: bool
    ) -> dict[str, Any]:
        """Start an explicitly confirmed background training job."""

        response = httpx.post(
            f"{self.base_url}/api/v1/training/runs",
            json={
                "dataset_id": dataset_id,
                "base_model": base_model,
                "method": method,
                "confirm": confirm,
            },
            headers=self.headers,
            timeout=10,
        )
        response.raise_for_status()
        return response.json()
