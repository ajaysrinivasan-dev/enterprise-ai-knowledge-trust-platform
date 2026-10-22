"""Runtime hardware and optional dependency checks."""

import importlib.util
import platform
import shutil
import sys
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class PreflightReport:
    """Serializable environment report and actionable limitations."""

    python_version: str
    platform: str
    torch_version: str | None
    cuda_available: bool
    cuda_version: str | None
    gpu_name: str | None
    free_vram_gb: float | None
    free_disk_gb: float
    optional_packages: dict[str, bool]
    warnings: tuple[str, ...]

    def as_dict(self) -> dict:
        """Return a JSON-safe report."""

        return asdict(self)


def collect_preflight(output_dir: str = "artifacts/training") -> PreflightReport:
    """Detect hardware at runtime without downloading packages or models."""

    packages = {
        name: importlib.util.find_spec(name) is not None
        for name in ("torch", "transformers", "peft", "bitsandbytes", "datasets")
    }
    torch_version = None
    cuda_available = False
    cuda_version = None
    gpu_name = None
    free_vram_gb = None
    warnings: list[str] = []
    if packages["torch"]:
        import torch

        torch_version = torch.__version__
        cuda_available = bool(torch.cuda.is_available())
        cuda_version = torch.version.cuda
        if cuda_available:
            gpu_name = torch.cuda.get_device_name(0)
            free_bytes, _ = torch.cuda.mem_get_info(0)
            free_vram_gb = free_bytes / 1024**3
        else:
            warnings.append("CUDA is unavailable; only CPU validation or training is possible.")
    else:
        warnings.append("PyTorch is not installed; training cannot start.")
    disk_path = Path(output_dir)
    disk_path.mkdir(parents=True, exist_ok=True)
    free_disk_gb = shutil.disk_usage(disk_path).free / 1024**3
    if not packages["transformers"] or not packages["peft"]:
        warnings.append("Transformers and PEFT are required for LoRA training.")
    if not packages["bitsandbytes"]:
        warnings.append("bitsandbytes is unavailable; QLoRA cannot run.")
    return PreflightReport(
        python_version=sys.version.split()[0],
        platform=platform.platform(),
        torch_version=torch_version,
        cuda_available=cuda_available,
        cuda_version=cuda_version,
        gpu_name=gpu_name,
        free_vram_gb=free_vram_gb,
        free_disk_gb=free_disk_gb,
        optional_packages=packages,
        warnings=tuple(warnings),
    )
