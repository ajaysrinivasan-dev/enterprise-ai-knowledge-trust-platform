"""Tests for path-free LoRA smoke-test reporting."""

from app.training.report import SmokeExperimentReport, render_smoke_experiment_report


def test_smoke_report_contains_verified_facts_without_raw_content_or_paths() -> None:
    report = SmokeExperimentReport(
        model_id="sshleifer/tiny-gpt2",
        dataset_fingerprint="ff4cf1d7",
        dataset_status="synthetic smoke-test data only",
        split_counts={"train": 5, "validation": 5, "test": 5},
        seed=2026,
        method="lora",
        max_length=64,
        batch_size=1,
        gradient_accumulation_steps=1,
        learning_rate=2e-4,
        epochs=1.0,
        training_steps=2,
        adapter_reloaded=True,
        baseline_accuracy=0.2,
        baseline_macro_f1=0.0666667,
        baseline_weighted_f1=0.0666667,
        baseline_latency_ms=0.00026,
        fine_tuned_accuracy=0.0,
        fine_tuned_macro_f1=0.0,
        fine_tuned_weighted_f1=0.0,
        fine_tuned_latency_ms=19.4911,
        fine_tuned_failure_rate=0.0,
        held_out_sample_count=5,
    )

    rendered = render_smoke_experiment_report(report)

    assert "sshleifer/tiny-gpt2" in rendered
    assert "train=5, validation=5, test=5" in rendered
    assert "Adapter reload: passed" in rendered
    assert "not a production evaluation" in rendered
    assert "C:\\" not in rendered
    assert "C:/" not in rendered
    assert "retention policy" not in rendered
