"""MedIA job matrix for QAM-Seg.

Recipes share one student (MambaLiteUNet), one split generator, and one
metric file. Nothing here downloads a dataset or imports ``mamba_ssm``.
``--smoke`` runs a short synthetic CUDA path on that student. A real run
requires local train arrays and CUDA. There is no CPU student.

Ablation names match the method note:

- ``qam-ungated``
- ``qam-gated``
- ``qam-gated-baq``
- ``qam-gated-baq-qw`` (gated + BAQ + sample MQE reweight)

``qam-qw`` is the direct PL-QW analogue: no hard gate, sample reweight
on the MQE score. ``qam-qar`` and ``qam-full`` add Quality-Guided Feature
Prototype Alignment (``--ssl-qar prototype``) and Quality-Guided CutMix
(``--ssl-cutmix``). Baselines are supervised, plain mean teacher, entropy
UA-MT (not MC dropout), and confidence-threshold pseudo-labels.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence


DEFAULT_SEEDS: tuple[int, ...] = (0, 1, 2)
DEFAULT_FRACTIONS: tuple[int, ...] = (5, 10, 20)
FULL_TOKEN = "full"


@dataclass(frozen=True)
class Recipe:
    name: str
    method: str
    gate: str
    baq: bool
    quality_reweight: str
    group: str
    include_full: bool = False
    qar: str = "off"
    cutmix: bool = False


RECIPES: dict[str, Recipe] = {
    "supervised": Recipe("supervised", "supervised", "ungated", False, "off", "supervised", True),
    "mt": Recipe("mt", "mt", "ungated", False, "off", "baseline"),
    "uamt": Recipe("uamt", "uamt", "ungated", False, "off", "baseline"),
    "pl": Recipe("pl", "pl", "ungated", False, "off", "baseline"),
    "qam-ungated": Recipe("qam-ungated", "qam", "ungated", False, "off", "ablation"),
    "qam-gated": Recipe("qam-gated", "qam", "gated", False, "off", "ablation"),
    "qam-gated-baq": Recipe("qam-gated-baq", "qam", "gated", True, "off", "ablation"),
    "qam-gated-baq-qw": Recipe("qam-gated-baq-qw", "qam", "gated", True, "sample", "ablation"),
    "qam-qw": Recipe("qam-qw", "qam", "ungated", False, "sample", "ablation"),
    "qam-qw-pixel": Recipe("qam-qw-pixel", "qam", "ungated", False, "pixel", "ablation"),
    "qam-qar": Recipe("qam-qar", "qam", "gated", True, "sample", "ablation", False, "prototype", False),
    "qam-full": Recipe("qam-full", "qam", "gated", True, "sample", "ablation", False, "prototype", True),
}

# Overnight matrix. Pixel reweight, qam-qar, and qam-full are registered and selectable via --recipes.
DEFAULT_RECIPE_NAMES: tuple[str, ...] = (
    "supervised",
    "mt",
    "uamt",
    "pl",
    "qam-ungated",
    "qam-gated",
    "qam-gated-baq",
    "qam-gated-baq-qw",
    "qam-qw",
)

ABLATION_NAMES: tuple[str, ...] = (
    "qam-ungated",
    "qam-gated",
    "qam-gated-baq",
    "qam-gated-baq-qw",
)


@dataclass(frozen=True)
class Job:
    recipe: str
    method: str
    group: str
    seed: int
    fraction_token: str
    fraction: Optional[float]
    gate: str
    baq: bool
    quality_reweight: str
    qar: str = "off"
    cutmix: bool = False


def parse_recipe_names(text: str) -> tuple[str, ...]:
    if not text or text.strip().lower() in ("all", "default"):
        return DEFAULT_RECIPE_NAMES
    names = tuple(part.strip() for part in text.split(",") if part.strip())
    unknown = [name for name in names if name not in RECIPES]
    if unknown:
        known = ", ".join(sorted(RECIPES))
        raise ValueError(f"Unknown recipe(s): {', '.join(unknown)}. Known: {known}")
    return names


def parse_seeds(text: str) -> tuple[int, ...]:
    if not text or text.strip().lower() == "default":
        return DEFAULT_SEEDS
    seeds = tuple(int(part.strip()) for part in text.split(",") if part.strip())
    if any(seed < 0 for seed in seeds):
        raise ValueError(f"Seeds must be non-negative, got {seeds}")
    if not seeds:
        raise ValueError("At least one seed is required.")
    return seeds


def parse_fractions(text: str) -> tuple[int, ...]:
    """Integer percents. ``5,10,20`` is the MedIA labeled-fraction set."""
    if not text or text.strip().lower() == "default":
        return DEFAULT_FRACTIONS
    fractions = tuple(int(part.strip()) for part in text.split(",") if part.strip())
    if not fractions:
        raise ValueError("At least one labeled fraction is required.")
    for value in fractions:
        if value <= 0 or value > 100:
            raise ValueError(f"Fraction percent must lie in 1..100, got {value}")
    return fractions


def fraction_to_float(percent: int) -> float:
    return float(percent) / 100.0


def iter_jobs(
    seeds: Sequence[int] = DEFAULT_SEEDS,
    fractions: Sequence[int] = DEFAULT_FRACTIONS,
    recipe_names: Sequence[str] = DEFAULT_RECIPE_NAMES,
) -> list[Job]:
    """One job per recipe, seed, and labeled fraction, plus supervised full."""
    jobs: list[Job] = []
    for name in recipe_names:
        recipe = RECIPES[name]
        for seed in seeds:
            for percent in fractions:
                jobs.append(
                    Job(
                        recipe=recipe.name,
                        method=recipe.method,
                        group=recipe.group,
                        seed=int(seed),
                        fraction_token=str(int(percent)),
                        fraction=fraction_to_float(int(percent)),
                        gate=recipe.gate,
                        baq=recipe.baq,
                        quality_reweight=recipe.quality_reweight,
                        qar=recipe.qar,
                        cutmix=recipe.cutmix,
                    )
                )
            if recipe.include_full:
                jobs.append(
                    Job(
                        recipe=recipe.name,
                        method=recipe.method,
                        group=recipe.group,
                        seed=int(seed),
                        fraction_token=FULL_TOKEN,
                        fraction=1.0,
                        gate=recipe.gate,
                        baq=recipe.baq,
                        quality_reweight=recipe.quality_reweight,
                        qar=recipe.qar,
                        cutmix=recipe.cutmix,
                    )
                )
    return jobs


def job_argv(
    job: Job,
    *,
    steps: Optional[int],
    batch_size: int,
    device: str,
    backbone: str,
    image_size: Optional[int],
    labeled_npy: str = "",
    mask_npy: str = "",
    split_json: str = "",
    eval_image: str = "",
    eval_mask: str = "",
    out_dir: str,
    lambda_qw: float = 1.0,
    qw_ramp: str = "gaussian",
    uncertainty_tau: float = 0.75,
    pl_threshold: float = 0.95,
) -> list[str]:
    """``train.py --ssl`` argv for one cell. No dataset URL is included."""
    step_text = str(int(steps)) if steps is not None else "<REQUIRED --steps>"
    argv = [
        "train.py",
        "--ssl",
        "--ssl-method",
        job.method,
        "--ssl-gate",
        job.gate,
        "--quality-reweight",
        job.quality_reweight,
        "--ssl-lambda-qw",
        str(float(lambda_qw)),
        "--ssl-qw-ramp",
        qw_ramp,
        "--ssl-steps",
        step_text,
        "--ssl-batch-size",
        str(int(batch_size)),
        "--ssl-device",
        device,
        "--ssl-backbone",
        backbone,
        "--ssl-seed",
        str(int(job.seed)),
        "--ssl-recipe",
        job.recipe,
        "--ssl-group",
        job.group,
        "--ssl-fraction",
        job.fraction_token,
        "--ssl-out-dir",
        out_dir,
    ]
    if image_size is not None:
        argv.extend(["--ssl-image-size", str(int(image_size))])
    if job.baq:
        argv.append("--ssl-baq")
    if getattr(job, "cutmix", False):
        argv.append("--ssl-cutmix")
    if getattr(job, "qar", "off") != "off":
        argv.extend(["--ssl-qar", str(job.qar)])
    if job.method == "uamt":
        argv.extend(["--ssl-uncertainty", "entropy", "--ssl-uncertainty-tau", str(float(uncertainty_tau))])
    if job.method == "pl":
        argv.extend(["--ssl-pl-threshold", str(float(pl_threshold))])
    if job.method == "supervised":
        argv.extend(["--ssl-lambda-max", "0"])
    if split_json:
        argv.extend(
            [
                "--ssl-split",
                split_json,
                "--ssl-labeled-npy",
                labeled_npy,
                "--ssl-mask-npy",
                mask_npy,
            ]
        )
    elif labeled_npy or mask_npy:
        argv.extend(["--ssl-labeled-npy", labeled_npy, "--ssl-mask-npy", mask_npy])
    if eval_image or eval_mask:
        argv.extend(["--ssl-eval-npy", eval_image, "--ssl-eval-mask-npy", eval_mask])
    return argv
