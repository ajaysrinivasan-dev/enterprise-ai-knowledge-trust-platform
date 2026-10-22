"""Safe adapter artifact lifecycle tests."""

import pytest
from app.training.artifacts import AdapterMetadata, list_adapters, load_metadata, save_metadata


def test_adapter_metadata_round_trip_and_missing_artifact_rejection(tmp_path) -> None:
    adapter = tmp_path / "run-1"
    weights = adapter / "adapter_model.safetensors"
    weights.parent.mkdir()
    weights.write_bytes(b"test")
    metadata = AdapterMetadata(
        run_id="run-1",
        base_model="tiny",
        method="lora",
        dataset_fingerprint="abc",
        split_seed=42,
        status="completed",
        output_dir=str(adapter),
        tokenizer_path=str(adapter),
        adapter_path=str(weights),
    )
    save_metadata(adapter, metadata)

    assert load_metadata(adapter) == metadata
    assert list_adapters(tmp_path) == [metadata]

    weights.unlink()
    with pytest.raises(ValueError, match="adapter files"):
        load_metadata(adapter)
