"""
Run a grid of ablation experiments across datasets and methods, with resume and reporting.

Ablations:
- method: no_al | baseline_al | uncertainty_cgan | vae
- uncertainty estimator: dropout | ensemble | laplace
- initial labeled proportion: e.g. 0.8, 0.5, 0.3, 0.1

This script shells out to the existing runner scripts and collects their `results.json` files
into a single `RESULTS.md` report.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


METHOD_TO_SCRIPT = {
    "no_al": "run_baseline_no_al.py",
    "baseline_al": "run_baseline_al.py",
    "uncertainty_cgan": "run_uncertainty_cgan_al.py",
    "vae": "run_vae_interp_al.py",
}


DEFAULT_DATASETS = [
    "iris",
    "wine",
    "breast_cancer",
    "two_moons",
    "circles",
    "boston",
    "mnist",
    "fashion_mnist",
    "cifar10",
]


@dataclass(frozen=True)
class TrialConfig:
    dataset: str
    method: str
    uncertainty: str
    initial_label_frac: float
    seed: int
    epochs: int
    acq_steps: int
    acq_size: int
    dropout: float
    mc_samples: int
    ensemble_size: int
    laplace_prior: float
    laplace_samples: int
    max_train_samples: Optional[int]
    max_test_samples: Optional[int]
    # method-specific
    cgan_steps: int
    synthetic_count: int
    vae_epochs: int
    vae_latent_dim: int
    batch_size: int


def log(msg: str) -> None:
    print(msg, flush=True)


def _slug(s: str) -> str:
    return s.replace("/", "-").replace(" ", "_")


def trial_dir(out_root: Path, cfg: TrialConfig) -> Path:
    parts = [
        _slug(cfg.dataset),
        f"method={cfg.method}",
        f"unc={cfg.uncertainty}",
        f"init={int(round(cfg.initial_label_frac * 100))}",
        f"seed={cfg.seed}",
        f"ep={cfg.epochs}",
        f"steps={cfg.acq_steps}",
        f"acq={cfg.acq_size}",
    ]
    if cfg.max_train_samples is not None:
        parts.append(f"maxTr={cfg.max_train_samples}")
    if cfg.max_test_samples is not None:
        parts.append(f"maxTe={cfg.max_test_samples}")
    if cfg.uncertainty == "ensemble":
        parts.append(f"ens={cfg.ensemble_size}")
    if cfg.uncertainty == "dropout":
        parts.append(f"mc={cfg.mc_samples}")
    if cfg.uncertainty == "laplace":
        parts.append(f"lapP={cfg.laplace_prior}")
        parts.append(f"lapS={cfg.laplace_samples}")
    if cfg.method == "uncertainty_cgan":
        parts.append(f"cganSteps={cfg.cgan_steps}")
        parts.append(f"syn={cfg.synthetic_count}")
    if cfg.method == "vae":
        parts.append(f"vaeEp={cfg.vae_epochs}")
        parts.append(f"z={cfg.vae_latent_dim}")
        parts.append(f"syn={cfg.synthetic_count}")

    return out_root / _slug(cfg.dataset) / "__".join(parts)


def build_command(cfg: TrialConfig, out_dir: Path, make_gif: bool) -> List[str]:
    script = METHOD_TO_SCRIPT[cfg.method]
    cmd = [
        sys.executable,
        script,
        "--dataset",
        cfg.dataset,
        "--seed",
        str(cfg.seed),
        "--epochs",
        str(cfg.epochs),
        "--dropout",
        str(cfg.dropout),
        "--initial-label-frac",
        str(cfg.initial_label_frac),
        "--uncertainty",
        cfg.uncertainty,
        "--mc-samples",
        str(cfg.mc_samples),
        "--ensemble-size",
        str(cfg.ensemble_size),
        "--laplace-prior",
        str(cfg.laplace_prior),
        "--laplace-samples",
        str(cfg.laplace_samples),
        "--output",
        str(out_dir),
    ]
    if cfg.max_train_samples is not None:
        cmd += ["--max-train-samples", str(cfg.max_train_samples)]
    if cfg.max_test_samples is not None:
        cmd += ["--max-test-samples", str(cfg.max_test_samples)]

    if cfg.method in {"baseline_al", "uncertainty_cgan", "vae"}:
        cmd += ["--acq-steps", str(cfg.acq_steps), "--acq-size", str(cfg.acq_size)]

    if cfg.method == "uncertainty_cgan":
        cmd += ["--cgan-steps", str(cfg.cgan_steps), "--synthetic-count", str(cfg.synthetic_count)]

    if cfg.method == "vae":
        cmd += [
            "--batch-size",
            str(cfg.batch_size),
            "--synthetic-count",
            str(cfg.synthetic_count),
            "--vae-epochs",
            str(cfg.vae_epochs),
            "--vae-latent-dim",
            str(cfg.vae_latent_dim),
        ]

    if make_gif:
        cmd.append("--make-gif")
    return cmd


def _valid_results(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        json.loads(path.read_text())
        return True
    except Exception:
        return False


def run_trial(cfg: TrialConfig, out_root: Path, *, make_gif: bool, resume: bool) -> Tuple[bool, Path]:
    out_dir = trial_dir(out_root, cfg)
    out_dir.mkdir(parents=True, exist_ok=True)
    results_path = out_dir / "results.json"
    if resume and _valid_results(results_path):
        return True, out_dir

    # Record config + cmd for reproducibility
    cmd = build_command(cfg, out_dir, make_gif=make_gif)
    (out_dir / "config.json").write_text(json.dumps(asdict(cfg), indent=2))
    (out_dir / "cmd.txt").write_text(" ".join(cmd) + "\n")

    log_path = out_dir / "run.log"
    with log_path.open("w") as f:
        f.write("[CMD] " + " ".join(cmd) + "\n\n")
        f.flush()
        proc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, text=True)
    return proc.returncode == 0 and _valid_results(results_path), out_dir


def tail_file(path: Path, n: int = 30) -> str:
    try:
        lines = path.read_text(errors="replace").splitlines()
    except Exception:
        return ""
    return "\n".join(lines[-n:])


def iter_trials(
    datasets: List[str],
    methods: List[str],
    uncertainties: List[str],
    label_fracs: List[float],
    seeds: List[int],
    args: argparse.Namespace,
) -> Iterable[TrialConfig]:
    for dataset in datasets:
        for method in methods:
            for unc in uncertainties:
                for frac in label_fracs:
                    for seed in seeds:
                        yield TrialConfig(
                            dataset=dataset,
                            method=method,
                            uncertainty=unc,
                            initial_label_frac=frac,
                            seed=seed,
                            epochs=args.epochs,
                            acq_steps=args.acq_steps,
                            acq_size=args.acq_size,
                            dropout=args.dropout,
                            mc_samples=args.mc_samples,
                            ensemble_size=args.ensemble_size,
                            laplace_prior=args.laplace_prior,
                            laplace_samples=args.laplace_samples,
                            max_train_samples=args.max_train_samples_for(dataset),
                            max_test_samples=args.max_test_samples_for(dataset),
                            cgan_steps=args.cgan_steps,
                            synthetic_count=args.synthetic_count,
                            vae_epochs=args.vae_epochs,
                            vae_latent_dim=args.vae_latent_dim,
                            batch_size=args.batch_size,
                        )


def collect_results(out_root: Path) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    for results_path in out_root.glob("*/**/results.json"):
        try:
            payload = json.loads(results_path.read_text())
        except Exception:
            continue
        payload["_results_path"] = str(results_path)
        payload["_run_dir"] = str(results_path.parent)
        records.append(payload)
    return records


def _method_label(method: str) -> str:
    return {
        "no_al": "NO-AL",
        "baseline_al": "Baseline-AL",
        "uncertainty_cgan": "Unc-CGAN-AL",
        "vae": "VAE-AL",
    }.get(method, method)


def write_results_md(records: List[Dict[str, Any]], path: Path) -> None:
    # Normalize into (dataset, init, unc, method) -> metrics
    index: Dict[Tuple[str, float, str, str], Dict[str, Any]] = {}
    for r in records:
        dataset = r.get("dataset")
        init = float(r.get("initial_label_frac", 0.5))
        unc = r.get("uncertainty_method", "dropout")
        method = r.get("method", None)
        # Older runners didn't store method; infer from run_dir name
        if method is None:
            run_dir = str(r.get("_run_dir", ""))
            if "run_uncertainty_cgan_al" in run_dir or "uncertainty_cgan" in run_dir:
                method = "uncertainty_cgan"
            elif "vae" in run_dir:
                method = "vae"
            elif "baseline_no_al" in run_dir or "no_al" in run_dir:
                method = "no_al"
            else:
                method = "baseline_al"
        key = (str(dataset), init, str(unc), str(method))
        index[key] = r.get("metrics", {})

    # Build row keys
    datasets = sorted({k[0] for k in index.keys()})
    inits = sorted({k[1] for k in index.keys()})
    uncs = sorted({k[2] for k in index.keys()})

    lines: List[str] = []
    lines.append("# Ablation Results\n")
    lines.append("This file is generated by `run_ablations.py`.\n")

    # Classification summary (accuracy/f1/nll)
    lines.append("## Classification (accuracy / f1_macro / nll)\n")
    header = [
        "dataset",
        "init_frac",
        "uncertainty",
        "NO-AL",
        "Baseline-AL",
        "Unc-CGAN-AL",
        "VAE-AL",
    ]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("|" + "|".join(["---"] * len(header)) + "|")
    for d in datasets:
        for init in inits:
            for unc in uncs:
                # include only if there is at least one classification metric for this row
                any_cls = any(
                    "accuracy" in index.get((d, init, unc, m), {}) for m in ["no_al", "baseline_al", "uncertainty_cgan", "vae"]
                )
                if not any_cls:
                    continue

                row = [d, f"{init:.2f}", unc]
                for m in ["no_al", "baseline_al", "uncertainty_cgan", "vae"]:
                    metrics = index.get((d, init, unc, m), {})
                    if "accuracy" in metrics:
                        acc = float(metrics.get("accuracy"))
                        f1 = float(metrics.get("f1_macro", float("nan")))
                        nll = float(metrics.get("nll", float("nan")))
                        cell = f"{acc:.4f} / {f1:.4f} / {nll:.4f}"
                    else:
                        cell = ""
                    row.append(cell)
                lines.append("| " + " | ".join(row) + " |")

    # Regression summary (rmse/mae/mse)
    lines.append("\n## Regression (rmse / mae / mse)\n")
    header = [
        "dataset",
        "init_frac",
        "uncertainty",
        "NO-AL",
        "Baseline-AL",
        "Unc-CGAN-AL",
        "VAE-AL",
    ]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("|" + "|".join(["---"] * len(header)) + "|")
    for d in datasets:
        for init in inits:
            for unc in uncs:
                row = [d, f"{init:.2f}", unc]
                any_reg = False
                for m in ["no_al", "baseline_al", "uncertainty_cgan", "vae"]:
                    metrics = index.get((d, init, unc, m), {})
                    if "rmse" in metrics or "mae" in metrics or "mse" in metrics:
                        any_reg = True
                        rmse = metrics.get("rmse", float("nan"))
                        mae = metrics.get("mae", float("nan"))
                        mse = metrics.get("mse", float("nan"))
                        row.append(f"{rmse:.4f} / {mae:.4f} / {mse:.4f}")
                    else:
                        row.append("")
                if any_reg:
                    lines.append("| " + " | ".join(row) + " |")

    # Full metrics appendix
    lines.append("\n## Full Metrics (JSON)\n")
    lines.append("| dataset | init_frac | uncertainty | method | metrics |")
    lines.append("|---|---:|---|---|---|")
    for (d, init, unc, m), metrics in sorted(index.items(), key=lambda x: (x[0][0], x[0][1], x[0][2], x[0][3])):
        lines.append(
            "| "
            + " | ".join(
                [
                    d,
                    f"{init:.2f}",
                    unc,
                    _method_label(m),
                    "`" + json.dumps(metrics, sort_keys=True) + "`",
                ]
            )
            + " |"
        )

    path.write_text("\n".join(lines) + "\n")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run ablations across datasets/methods and write RESULTS.md.")
    p.add_argument("--out-root", default="ablations_out", help="Root folder for all runs.")
    p.add_argument("--datasets", default=",".join(DEFAULT_DATASETS), help="Comma-separated datasets.")
    p.add_argument("--methods", default="no_al,baseline_al,uncertainty_cgan,vae", help="Comma-separated methods.")
    p.add_argument("--uncertainties", default="dropout,ensemble,laplace", help="Comma-separated uncertainty methods.")
    p.add_argument("--label-fracs", default="0.8,0.5,0.3,0.1", help="Comma-separated initial label fractions.")
    p.add_argument("--seeds", default="42", help="Comma-separated seeds.")
    p.add_argument("--resume", action="store_true", help="Skip trials with an existing valid results.json.")
    p.add_argument("--make-gif", action="store_true", help="Also create GIFs in each run directory.")
    p.add_argument("--results-md", default="RESULTS.md", help="Path to write the results markdown file.")

    # Common hyperparams
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--acq-steps", type=int, default=5)
    p.add_argument("--acq-size", type=int, default=20)
    p.add_argument("--dropout", type=float, default=0.3)
    p.add_argument("--mc-samples", type=int, default=20)
    p.add_argument("--ensemble-size", type=int, default=5)
    p.add_argument("--laplace-prior", type=float, default=1.0)
    p.add_argument("--laplace-samples", type=int, default=30)

    # Method-specific hyperparams
    p.add_argument("--cgan-steps", type=int, default=200)
    p.add_argument("--synthetic-count", type=int, default=200)
    p.add_argument("--vae-epochs", type=int, default=5)
    p.add_argument("--vae-latent-dim", type=int, default=32)
    p.add_argument("--batch-size", type=int, default=256)

    # Optional dataset caps (fast mode)
    p.add_argument("--fast", action="store_true", help="Use smaller sample caps for vision datasets.")
    p.add_argument("--max-train-samples", type=int, default=None, help="Global train cap (overrides --fast).")
    p.add_argument("--max-test-samples", type=int, default=None, help="Global test cap (overrides --fast).")

    args = p.parse_args()

    # Resolve per-dataset caps.
    def max_train_samples_for(dataset: str) -> Optional[int]:
        if args.max_train_samples is not None:
            return args.max_train_samples
        if not args.fast:
            return None
        if dataset.lower() in {"mnist", "fashion_mnist", "fashion-mnist", "fmnist", "cifar10", "cifar-10"}:
            return 5000
        return None

    def max_test_samples_for(dataset: str) -> Optional[int]:
        if args.max_test_samples is not None:
            return args.max_test_samples
        if not args.fast:
            return None
        if dataset.lower() in {"mnist", "fashion_mnist", "fashion-mnist", "fmnist", "cifar10", "cifar-10"}:
            return 1000
        return None

    args.max_train_samples_for = max_train_samples_for  # type: ignore[attr-defined]
    args.max_test_samples_for = max_test_samples_for  # type: ignore[attr-defined]
    return args


def main() -> None:
    args = parse_args()
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    uncertainties = [u.strip() for u in args.uncertainties.split(",") if u.strip()]
    label_fracs = [float(x.strip()) for x in args.label_fracs.split(",") if x.strip()]
    seeds = [int(x.strip()) for x in args.seeds.split(",") if x.strip()]

    trials = list(iter_trials(datasets, methods, uncertainties, label_fracs, seeds, args))
    total = len(trials)
    done = 0
    failed: List[str] = []

    start_time = time.time()
    for i, cfg in enumerate(trials, start=1):
        run_dir = trial_dir(out_root, cfg)
        results_path = run_dir / "results.json"
        if args.resume and _valid_results(results_path):
            done += 1
            continue

        t0 = time.time()
        log(f"[{i}/{total}] dataset={cfg.dataset} method={cfg.method} unc={cfg.uncertainty} init={cfg.initial_label_frac:.2f}")
        ok, _ = run_trial(cfg, out_root, make_gif=args.make_gif, resume=args.resume)
        dt = time.time() - t0
        done += 1

        # ETA
        elapsed = time.time() - start_time
        rate = elapsed / max(done, 1)
        eta = rate * (total - done)
        status = "OK" if ok else "FAIL"
        log(f"  -> {status} in {dt:.1f}s | progress {done}/{total} | ETA {eta/60:.1f} min")
        if not ok:
            failed.append(str(run_dir))
            log_path = run_dir / "run.log"
            if log_path.exists():
                log(f"  -> see {log_path}")
                snippet = tail_file(log_path, n=25)
                if snippet:
                    log("  -> last log lines:")
                    for line in snippet.splitlines():
                        log("     " + line)

    records = collect_results(out_root)
    write_results_md(records, Path(args.results_md))
    log(f"Wrote results to {args.results_md}")
    if failed:
        log("Failures:")
        for d in failed:
            log(f" - {d}")


if __name__ == "__main__":
    main()
