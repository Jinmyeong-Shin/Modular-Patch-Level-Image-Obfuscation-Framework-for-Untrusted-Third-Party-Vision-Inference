from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
sys.path.insert(0, str(SRC_ROOT))

from vit_obfuscation.config.experiment import ExperimentConfig
from vit_obfuscation.outputs.manifest import write_manifest


DEFAULT_OUTPUT_DIR = Path("outputs/revision_v3")

DEFAULT_ORDER = [
    "configs/experiments/clip_image_retrieval.yaml",
    "configs/experiments/clip_flickr30k_retrieval.yaml",
    "configs/experiments/medical_minimsd_segmentation.yaml",
    "configs/experiments/blip_flickr30k_captioning.yaml",
    "configs/experiments/yolos_coco.yaml",
    "configs/experiments/owlvit_coco.yaml",
    "configs/experiments/mvtec_anomaly.yaml",
]


def _result_path(config: ExperimentConfig, output_dir: Path) -> Path:
    return output_dir / f"{config.name}_results.json"


def _load_result(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        with path.open() as f:
            return json.load(f)
    except Exception:
        return None


def _is_success(path: Path) -> bool:
    data = _load_result(path)
    return bool(data and "error" not in data and "clean" in data and "obfuscated" in data)


def _write_error(
    config: ExperimentConfig,
    output_dir: Path,
    error: str,
    *,
    returncode: int | None = None,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    result = {
        "experiment": config.name,
        "model": config.model.hf_model_name_or_path,
        "task": config.model.task,
        "error": error,
    }
    if returncode is not None:
        result["returncode"] = returncode
    path = _result_path(config, output_dir)
    with path.open("w") as f:
        json.dump(result, f, indent=2)
    write_manifest(
        config=config,
        output_dir=str(output_dir),
        result_file=str(path),
        status="error",
        error=error,
    )
    return result


def _terminate_process_group(process: subprocess.Popen, grace_seconds: int = 20) -> None:
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    except PermissionError as error:
        print(f"warning: could not terminate process group {process.pid}: {error}")
        return

    deadline = time.time() + grace_seconds
    while time.time() < deadline:
        if process.poll() is not None:
            return
        time.sleep(1)

    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        return
    except PermissionError as error:
        print(f"warning: could not kill process group {process.pid}: {error}")


def run_one(config_path: Path, output_dir: Path, timeout_seconds: int) -> dict:
    config = ExperimentConfig.from_yaml(str(config_path))
    config.output_dir = str(output_dir)
    result_path = _result_path(config, output_dir)

    if _is_success(result_path):
        print(f"[skip] {config.name}: existing successful artifact")
        return _load_result(result_path) or {}

    command = [
        sys.executable,
        "-m",
        "vit_obfuscation.cli",
        "run",
        "--config",
        str(config_path),
        "--output-dir",
        str(output_dir),
    ]
    env = os.environ.copy()
    env.setdefault("PYTHONUNBUFFERED", "1")
    env["PYTHONPATH"] = os.pathsep.join(
        [str(SRC_ROOT), env["PYTHONPATH"]] if env.get("PYTHONPATH") else [str(SRC_ROOT)]
    )

    print(f"[run] {config.name}: {' '.join(command)}")
    process = subprocess.Popen(command, env=env, start_new_session=True)
    try:
        returncode = process.wait(timeout=timeout_seconds)
    except subprocess.TimeoutExpired:
        _terminate_process_group(process)
        error = f"timed out after {timeout_seconds // 60} minutes"
        print(f"[timeout] {config.name}: {error}")
        return _write_error(config, output_dir, error)

    if returncode != 0:
        data = _load_result(result_path)
        if data is not None:
            return data
        error = f"process exited with return code {returncode}"
        print(f"[failed] {config.name}: {error}")
        return _write_error(config, output_dir, error, returncode=returncode)

    data = _load_result(result_path)
    if data is None:
        error = "process completed without writing a result artifact"
        print(f"[failed] {config.name}: {error}")
        return _write_error(config, output_dir, error)

    print(f"[ok] {config.name}")
    return data


def _write_combined(results: list[dict], output_dir: Path) -> None:
    combined_path = output_dir / "all_results.json"
    existing: list[dict] = []
    if combined_path.exists():
        try:
            with combined_path.open() as f:
                existing = json.load(f)
        except Exception:
            existing = []

    by_name = {
        row.get("experiment"): row
        for row in existing
        if _is_result_success(row)
    }
    for row in results:
        if _is_result_success(row):
            by_name[row["experiment"]] = row

    output_dir.mkdir(parents=True, exist_ok=True)
    with combined_path.open("w") as f:
        json.dump(list(by_name.values()), f, indent=2, default=str)


def _is_result_success(row: object) -> bool:
    return (
        isinstance(row, dict)
        and bool(row.get("experiment"))
        and "error" not in row
        and "clean" in row
        and "obfuscated" in row
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run remaining revision-v3 experiments with per-job timeouts."
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory for result artifacts.",
    )
    parser.add_argument(
        "--timeout-minutes",
        type=int,
        default=45,
        help="Timeout per experiment.",
    )
    parser.add_argument(
        "--config",
        action="append",
        dest="configs",
        help="Config path to run. Can be passed multiple times.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Remove existing result JSONs for selected configs before running.",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    config_paths = [Path(path) for path in (args.configs or DEFAULT_ORDER)]
    timeout_seconds = args.timeout_minutes * 60

    selected: list[Path] = []
    for path in config_paths:
        if not path.exists():
            print(f"[missing] {path}")
            continue
        selected.append(path)

    if args.force:
        for path in selected:
            config = ExperimentConfig.from_yaml(str(path))
            result_path = _result_path(config, output_dir)
            if result_path.exists():
                result_path.unlink()

    results = []
    for path in selected:
        results.append(run_one(path, output_dir, timeout_seconds))

    _write_combined(results, output_dir)

    failures = [row for row in results if isinstance(row, dict) and "error" in row]
    print(
        f"[done] {len(results) - len(failures)} successful or skipped, "
        f"{len(failures)} failed"
    )
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
