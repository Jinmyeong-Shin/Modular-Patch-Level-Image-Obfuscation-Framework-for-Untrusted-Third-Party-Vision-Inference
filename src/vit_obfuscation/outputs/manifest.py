from __future__ import annotations

import json
import platform
import subprocess
from dataclasses import asdict, is_dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch

try:
    import transformers
except ImportError:  # pragma: no cover - transformers is a project dependency
    transformers = None


def _run_git(args: list[str]) -> str | None:
    try:
        completed = subprocess.run(
            ["git", *args],
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except Exception:
        return None
    if completed.returncode != 0:
        return None
    return completed.stdout.strip()


def _to_jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return _to_jsonable(asdict(value))
    if isinstance(value, dict):
        return {str(k): _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(v) for v in value]
    if hasattr(value, "__dict__") and not isinstance(value, type):
        return {
            str(k): _to_jsonable(v)
            for k, v in vars(value).items()
            if not k.startswith("_")
        }
    return value


def build_manifest(
    *,
    config,
    result_file: str | Path,
    status: str,
    error: str | None = None,
) -> dict[str, Any]:
    """Build a reproducibility manifest for a single experiment artifact."""
    return {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "status": status,
        "error": error,
        "result_file": str(result_file),
        "experiment": config.name,
        "task": config.model.task,
        "model": config.model.hf_model_name_or_path,
        "dataset": config.dataset.hf_dataset_name_or_path,
        "seed": config.seed,
        "config": _to_jsonable(config),
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "torch": torch.__version__,
            "transformers": getattr(transformers, "__version__", None),
            "cuda_available": torch.cuda.is_available(),
        },
        "git": {
            "commit": _run_git(["rev-parse", "HEAD"]),
            "status_short": _run_git(["status", "--short"]),
        },
    }


def write_manifest(
    *,
    config,
    output_dir: str | Path,
    result_file: str | Path,
    status: str,
    error: str | None = None,
) -> Path:
    """Write a per-experiment manifest next to canonical revision artifacts."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    manifest_path = output_path / f"{config.name}_manifest.json"
    manifest = build_manifest(
        config=config,
        result_file=result_file,
        status=status,
        error=error,
    )
    with manifest_path.open("w") as f:
        json.dump(manifest, f, indent=2, default=str)
    return manifest_path
