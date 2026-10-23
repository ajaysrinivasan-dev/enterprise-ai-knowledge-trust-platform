"""Small HTTP client used by Streamlit instead of duplicating backend logic."""

from typing import Any

import httpx


class ApiClientError(RuntimeError):
    """Base error for safe frontend API failure messages."""


class ApiAuthenticationError(ApiClientError):
    """The backend rejected the configured credential."""


class ApiAuthorizationError(ApiClientError):
    """The credential is valid but lacks permission for the operation."""


class ApiConnectionError(ApiClientError):
    """The frontend could not reach the backend."""


class ApiClient:
    """Call the local FastAPI backend with explicit, bounded timeouts."""

    def __init__(self, base_url: str = "http://127.0.0.1:8000", api_key: str | None = None) -> None:
        self.base_url = base_url.rstrip("/")
        self.headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}

    def _request(self, method: str, path: str, **kwargs: Any) -> httpx.Response:
        """Send one request and translate failures without exposing credentials."""

        try:
            response = httpx.request(
                method,
                f"{self.base_url}{path}",
                headers=self.headers,
                **kwargs,
            )
            response.raise_for_status()
            return response
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code == 401:
                raise ApiAuthenticationError(
                    "FastAPI rejected the configured frontend credential. "
                    "Set EAKTP_UI_API_KEY to a configured principal key and restart both services."
                ) from exc
            if exc.response.status_code == 403:
                raise ApiAuthorizationError(
                    "The configured frontend credential is authenticated but lacks permission "
                    "for this operation. Use a writer credential for documents."
                ) from exc
            raise ApiClientError(
                f"FastAPI returned HTTP {exc.response.status_code} for this request."
            ) from exc
        except httpx.RequestError as exc:
            raise ApiConnectionError(
                "Could not reach FastAPI. Start or restart the backend and check its URL."
            ) from exc

    def list_documents(self) -> list[dict[str, Any]]:
        return self._request("GET", "/api/v1/documents", timeout=10).json()

    def upload_document(self, filename: str, content: bytes) -> dict[str, Any]:
        return self._request(
            "POST",
            "/api/v1/documents",
            files={"file": (filename, content)},
            timeout=30,
        ).json()

    def ask(self, question: str) -> dict[str, Any]:
        return self._request(
            "POST", "/api/v1/questions", json={"question": question}, timeout=60
        ).json()

    def run_evaluation(self, dataset_id: str = "synthetic_v1") -> dict[str, Any]:
        """Run an approved offline evaluation dataset."""

        return self._request(
            "POST",
            "/api/v1/evaluations",
            json={"dataset_id": dataset_id},
            timeout=60,
        ).json()

    def list_evaluations(self) -> list[dict[str, Any]]:
        """List stored evaluation results."""

        return self._request("GET", "/api/v1/evaluations", timeout=10).json()

    def training_preflight(self) -> dict[str, Any]:
        """Retrieve runtime hardware and package availability."""

        return self._request("GET", "/api/v1/training/preflight", timeout=10).json()

    def training_datasets(self) -> list[dict[str, Any]]:
        """List approved training datasets."""

        return self._request("GET", "/api/v1/training/datasets", timeout=10).json()

    def start_training(
        self, dataset_id: str, base_model: str, method: str, confirm: bool
    ) -> dict[str, Any]:
        """Start an explicitly confirmed background training job."""

        return self._request(
            "POST",
            "/api/v1/training/runs",
            json={
                "dataset_id": dataset_id,
                "base_model": base_model,
                "method": method,
                "confirm": confirm,
            },
            timeout=10,
        ).json()
