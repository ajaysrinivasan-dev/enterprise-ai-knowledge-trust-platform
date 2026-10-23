"""Bounded single-process training job runtime."""

from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from threading import Lock
from uuid import uuid4

from app.core.config import get_settings
from app.training.artifacts import list_adapters
from app.training.benchmark import benchmark_comparison, benchmark_router
from app.training.config import TrainingConfig
from app.training.dataset import (
    load_intent_dataset,
    split_intent_dataset,
    validate_training_ready,
)
from app.training.lora import load_adapter_predictor, train_adapter
from app.training.preflight import collect_preflight

_DATASETS = {"intent_v1": Path("data/training/intent_v1.jsonl")}
_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="eaktp-training")
_lock = Lock()
_jobs: dict[str, Future] = {}


def dataset_path(dataset_id: str) -> Path:
    """Resolve an approved dataset identifier only."""

    path = _DATASETS.get(dataset_id)
    if path is None:
        raise ValueError("dataset is not approved")
    return path


def list_datasets() -> list[dict[str, object]]:
    """Return safe dataset summaries."""

    summaries = []
    for dataset_id, path in _DATASETS.items():
        rows = load_intent_dataset(path)
        splits = split_intent_dataset(rows)
        summaries.append(
            {
                "dataset_id": dataset_id,
                "example_count": len(rows),
                "split_sizes": {
                    "train": len(splits.train),
                    "validation": len(splits.validation),
                    "test": len(splits.test),
                },
                "fingerprint": splits.fingerprint,
                "warnings": list(splits.warnings),
            }
        )
    return summaries


def _run_training(job_id: str, dataset_id: str, config: TrainingConfig):
    try:
        rows = load_intent_dataset(dataset_path(dataset_id))
        splits = split_intent_dataset(rows, seed=config.seed)
        validate_training_ready(splits)
        return train_adapter(splits, config, run_id=job_id, confirm=True)
    except Exception:
        output = Path(config.output_dir).resolve() / job_id
        if output.exists():
            import shutil

            shutil.rmtree(output, ignore_errors=True)
        raise


def start_training(dataset_id: str, config: TrainingConfig) -> str:
    """Start one explicit bounded job and prevent concurrent duplicate jobs."""

    dataset_path(dataset_id)
    config.validate()
    rows = load_intent_dataset(dataset_path(dataset_id))
    validate_training_ready(split_intent_dataset(rows, seed=config.seed))
    with _lock:
        if any(not future.done() for future in _jobs.values()):
            raise RuntimeError("another training run is already active")
        job_id = str(uuid4())
        _jobs[job_id] = _executor.submit(_run_training, job_id, dataset_id, config)
    return job_id


def job_status(job_id: str) -> dict[str, object]:
    """Return safe status and summary information for one job."""

    future = _jobs.get(job_id)
    if future is None:
        raise KeyError("training job not found")
    if not future.done():
        return {"job_id": job_id, "status": "running"}
    try:
        metadata = future.result()
    except Exception as exc:
        return {"job_id": job_id, "status": "failed", "error": str(exc)}
    return {"job_id": job_id, "status": "completed", "metadata": metadata.__dict__}


def preflight() -> dict[str, object]:
    """Return hardware and optional dependency status."""

    return collect_preflight(get_settings().training_output_dir).as_dict()


def adapters() -> list[dict[str, object]]:
    """List valid completed adapters under the approved output root."""

    return [metadata.__dict__ for metadata in list_adapters(get_settings().training_output_dir)]


def benchmark_adapter(adapter_id: str) -> dict[str, object]:
    """Benchmark a local adapter and baseline on the exact same held-out split."""

    metadata = next(
        (
            item
            for item in list_adapters(get_settings().training_output_dir)
            if item.run_id == adapter_id
        ),
        None,
    )
    if metadata is None:
        raise ValueError("adapter is not an approved completed run")
    rows = load_intent_dataset(dataset_path("intent_v1"))
    splits = split_intent_dataset(rows, seed=metadata.split_seed)
    texts = [row["text"] for row in splits.test]
    labels = [row["intent"] for row in splits.test]
    dataset_fingerprint = splits.fingerprint

    def baseline(text: str) -> str:
        lowered = text.casefold()
        if "email" in lowered or "draft" in lowered:
            return "email_draft"
        if "show" in lowered or "count" in lowered:
            return "sql_report"
        if "find" in lowered or "section" in lowered:
            return "search"
        if "supported" in lowered or "verify" in lowered:
            return "verify"
        return "question"

    baseline_result = benchmark_router(
        "deterministic-baseline",
        baseline,
        texts,
        labels,
        dataset_fingerprint=dataset_fingerprint,
        dataset_status="synthetic_smoke",
    )
    adapter_result = benchmark_router(
        metadata.base_model,
        load_adapter_predictor(metadata),
        texts,
        labels,
        dataset_fingerprint=dataset_fingerprint,
        dataset_status="synthetic_smoke",
        trained=True,
    )
    adapter_result = adapter_result.__class__(
        **{**adapter_result.__dict__, "adapter_id": adapter_id}
    )
    return benchmark_comparison(baseline_result, adapter_result)
