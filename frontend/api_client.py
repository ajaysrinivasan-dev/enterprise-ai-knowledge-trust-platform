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

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:8000",
        api_key: str | None = None,
        admin_api_key: str | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self.admin_headers = (
            {"Authorization": f"Bearer {admin_api_key}"} if admin_api_key else None
        )

    def _request(self, method: str, path: str, **kwargs: Any) -> httpx.Response:
        """Send one request and translate failures without exposing credentials."""

        if self._is_admin_only(path) and self.admin_headers is None:
            if path.startswith("/api/v1/evaluations"):
                label = "Evaluation"
            elif path.startswith("/api/v1/training/"):
                label = "Training"
            else:
                label = "This operation"
            raise ApiAuthorizationError(
                f"{label} is administrator-only. Configure EAKTP_ADMIN_UI_API_KEY "
                "to an existing admin principal and restart Streamlit."
            )

        try:
            headers = self.admin_headers if self._is_admin_only(path) else self.headers
            response = httpx.request(
                method,
                f"{self.base_url}{path}",
                headers=headers,
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
                detail = ""
                try:
                    payload = exc.response.json()
                    if isinstance(payload, dict) and isinstance(payload.get("detail"), str):
                        detail = payload["detail"].strip()
                except ValueError:
                    pass

                if detail == "Administrator access required":
                    if path.startswith("/api/v1/evaluations"):
                        message = (
                            "Evaluation is administrator-only. Configure EAKTP_ADMIN_UI_API_KEY "
                            "with an authorized admin principal, then restart Streamlit."
                        )
                    elif path.startswith("/api/v1/training/"):
                        message = (
                            "Training is administrator-only. Configure EAKTP_ADMIN_UI_API_KEY "
                            "with an authorized admin principal, then restart Streamlit."
                        )
                    else:
                        message = (
                            "This operation requires administrator permission. "
                            "Ask the project administrator for access."
                        )
                elif detail == "Reader access is read-only":
                    message = (
                        "This operation changes data and requires a writer or administrator "
                        "credential. Document listing remains available to readers."
                    )
                else:
                    message = (
                        "The configured frontend credential is authenticated but lacks "
                        "permission for this operation. Check the backend role policy."
                    )
                raise ApiAuthorizationError(message) from exc
            raise ApiClientError(
                f"FastAPI returned HTTP {exc.response.status_code} for this request."
            ) from exc
        except httpx.RequestError as exc:
            raise ApiConnectionError(
                "Could not reach FastAPI. Start or restart the backend and check its URL."
            ) from exc

    @staticmethod
    def _is_admin_only(path: str) -> bool:
        return (
            path == "/api/v1/agent"
            or path.startswith("/api/v1/evaluations")
            or path.startswith("/api/v1/training/")
        )

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
