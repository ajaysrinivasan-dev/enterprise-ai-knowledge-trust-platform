"""Safe adapter artifact metadata and lifecycle operations."""

import json
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class AdapterMetadata:
    """Metadata required to validate an adapter against its base model."""

    run_id: str
    base_model: str
    method: str
    dataset_fingerprint: str
    split_seed: int
    status: str
    output_dir: str
    tokenizer_path: str
    adapter_path: str


def save_metadata(directory: str | Path, metadata: AdapterMetadata) -> Path:
    """Write JSON metadata only; model weights remain separate files."""

    target = Path(directory)
    target.mkdir(parents=True, exist_ok=True)
    path = target / "metadata.json"
    path.write_text(json.dumps(asdict(metadata), indent=2, sort_keys=True), encoding="utf-8")
    return path


def load_metadata(directory: str | Path) -> AdapterMetadata:
    """Load and validate metadata, never deserializing arbitrary Python objects."""

    path = Path(directory) / "metadata.json"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        metadata = AdapterMetadata(**payload)
    except (OSError, json.JSONDecodeError, TypeError, ValueError) as exc:
        raise ValueError("adapter metadata is missing or invalid") from exc
    if metadata.method not in {"lora", "qlora"} or metadata.status != "completed":
        raise ValueError("adapter metadata is not a completed supported run")
    adapter_path = Path(metadata.adapter_path).resolve()
    directory_path = Path(directory).resolve()
    if directory_path not in adapter_path.parents and adapter_path != directory_path:
        raise ValueError("adapter path escapes the metadata directory")
    if not adapter_path.exists():
        raise ValueError("adapter files are missing")
    tokenizer_path = Path(metadata.tokenizer_path).resolve()
    if directory_path not in tokenizer_path.parents and tokenizer_path != directory_path:
        raise ValueError("tokenizer path escapes the metadata directory")
    if not tokenizer_path.exists():
        raise ValueError("tokenizer files are missing")
    return metadata


def list_adapters(root: str | Path) -> list[AdapterMetadata]:
    """List only valid completed adapter metadata under an approved root."""

    base = Path(root).resolve()
    if not base.exists():
        return []
    adapters = []
    for metadata_path in base.glob("*/metadata.json"):
        directory = metadata_path.parent.resolve()
        if base not in directory.parents:
            continue
        try:
            adapters.append(load_metadata(directory))
        except ValueError:
            continue
    return adapters
