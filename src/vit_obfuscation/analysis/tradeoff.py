from __future__ import annotations

import copy
import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vit_obfuscation.config.experiment import ExperimentConfig

logger = logging.getLogger(__name__)


@dataclass
class TradeoffPoint:
    param_name: str
    param_value: int
    clean_accuracy: float
    obfuscated_accuracy: float
    attack_ssim: float
    attack_psnr: float


@dataclass
class TradeoffResult:
    param_name: str
    points: list[TradeoffPoint] = field(default_factory=list)


def run_tradeoff_sweep(
    base_config: ExperimentConfig,
    param_name: str,
    param_values: list[int],
    device: str = "cuda",
) -> TradeoffResult:
    """Sweep an obfuscation parameter and measure accuracy vs. privacy trade-off.

    Args:
        base_config: Base experiment configuration to modify for each sweep point.
        param_name: Either ``"patch_size"`` or ``"group_size"``.
        param_values: Values to sweep over for the chosen parameter.
        device: Device string forwarded to the experiment runner.

    Returns:
        A :class:`TradeoffResult` containing one :class:`TradeoffPoint` per
        successfully completed parameter value.
    """
    if param_name not in ("patch_size", "group_size"):
        raise ValueError(
            f"param_name must be 'patch_size' or 'group_size', got '{param_name}'"
        )

    # Import inside function to avoid circular imports
    from vit_obfuscation.runner.runner import ExperimentRunner

    result = TradeoffResult(param_name=param_name)

    for value in param_values:
        logger.info(f"Sweep: {param_name}={value}")

        config = copy.deepcopy(base_config)
        setattr(config.obfuscation, param_name, value)
        config.name = f"{base_config.name}_{param_name}_{value}"

        try:
            runner = ExperimentRunner(config)
            run_results = runner.run()

            clean = run_results.get("clean", {})
            obfuscated = run_results.get("obfuscated", {})

            clean_accuracy = clean.get("accuracy", clean.get("score", 0.0))
            obfuscated_accuracy = obfuscated.get(
                "accuracy", obfuscated.get("score", 0.0)
            )

            attack_ssim = obfuscated.get("attack_ssim", 0.0)
            attack_psnr = obfuscated.get("attack_psnr", 0.0)

            point = TradeoffPoint(
                param_name=param_name,
                param_value=value,
                clean_accuracy=float(clean_accuracy),
                obfuscated_accuracy=float(obfuscated_accuracy),
                attack_ssim=float(attack_ssim),
                attack_psnr=float(attack_psnr),
            )
            result.points.append(point)

            logger.info(
                f"  clean_acc={point.clean_accuracy:.4f}  "
                f"obf_acc={point.obfuscated_accuracy:.4f}  "
                f"ssim={point.attack_ssim:.4f}  psnr={point.attack_psnr:.2f}"
            )
        except Exception:
            logger.exception(f"Sweep failed for {param_name}={value}")

    logger.info(
        f"Sweep complete: {len(result.points)}/{len(param_values)} points collected"
    )
    return result
