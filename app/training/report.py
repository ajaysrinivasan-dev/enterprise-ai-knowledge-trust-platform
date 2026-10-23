"""Path-free reporting for verified LoRA smoke experiments."""

from dataclasses import dataclass


@dataclass(frozen=True)
class SmokeExperimentReport:
    """Measured smoke-test facts kept separate from production evaluation."""

    model_id: str
    dataset_fingerprint: str
    dataset_status: str
    split_counts: dict[str, int]
    seed: int
    method: str
    max_length: int
    batch_size: int
    gradient_accumulation_steps: int
    learning_rate: float
    epochs: float
    training_steps: int
    adapter_reloaded: bool
    baseline_accuracy: float
    baseline_macro_f1: float
    baseline_weighted_f1: float
    baseline_latency_ms: float
    fine_tuned_accuracy: float
    fine_tuned_macro_f1: float
    fine_tuned_weighted_f1: float
    fine_tuned_latency_ms: float
    fine_tuned_failure_rate: float
    held_out_sample_count: int


def render_smoke_experiment_report(report: SmokeExperimentReport) -> str:
    """Render a concise Markdown report without text, paths, or secrets."""

    counts = ", ".join(
        f"{name}={report.split_counts[name]}"
        for name in ("train", "validation", "test")
    )
    baseline = (
        f"accuracy={report.baseline_accuracy:.4f}, "
        f"macro-F1={report.baseline_macro_f1:.4f}, "
        f"weighted-F1={report.baseline_weighted_f1:.4f}, "
        f"latency={report.baseline_latency_ms:.4f} ms"
    )
    fine_tuned = (
        f"accuracy={report.fine_tuned_accuracy:.4f}, "
        f"macro-F1={report.fine_tuned_macro_f1:.4f}, "
        f"weighted-F1={report.fine_tuned_weighted_f1:.4f}, "
        f"latency={report.fine_tuned_latency_ms:.4f} ms, "
        f"failure rate={report.fine_tuned_failure_rate:.4f}"
    )
    return f"""# LoRA Smoke Experiment

**Status:** completed smoke test; not a production evaluation

## Configuration

- Model ID: `{report.model_id}`
- Method: `{report.method}`
- Dataset fingerprint: `{report.dataset_fingerprint}`
- Dataset status: {report.dataset_status}
- Split counts: {counts}
- Split seed: {report.seed}
- Max length / batch size: {report.max_length} / {report.batch_size}
- Gradient accumulation: {report.gradient_accumulation_steps}
- Learning rate / epochs: {report.learning_rate} / {report.epochs}
- Training steps: {report.training_steps}

## Verification and metrics

- Adapter reload: {"passed" if report.adapter_reloaded else "failed"}
- Held-out samples: {report.held_out_sample_count}
- Baseline: {baseline}
- Fine-tuned: {fine_tuned}

## Limitations

- The dataset is synthetic smoke-test data and contains only
  {report.held_out_sample_count} held-out examples.
- The run used {report.training_steps} bounded training steps; these results do
  not establish model quality or production generalization.
- The fine-tuned result must not be interpreted as an improvement over the baseline.
"""
