"""Hyperparameter grid sweep: GRASP + ConCare backbone on MIMIC-III mortality.

Usage:
    # Local MIMIC-III (on H200 / supercomputer)
    python sweep_concare_grasp.py --root /path/to/mimic3 --code-mapping

    # Synthetic MIMIC-III (no credentials needed)
    python sweep_concare_grasp.py --code-mapping

    # Dev subset (fast sanity check)
    python sweep_concare_grasp.py --dev

Results are appended to sweep_results.csv in the working directory.
"""

import argparse
import csv
import itertools
import os
import tempfile
import time

import torch

from pyhealth.datasets import MIMIC3Dataset, get_dataloader, split_by_patient
from pyhealth.models import GRASP
from pyhealth.tasks import MortalityPredictionMIMIC3
from pyhealth.trainer import Trainer

# ── Sweep grid ──────────────────────────────────────────────────────────
GRID = {
    "embedding_dim": [8, 16, 32],
    "hidden_dim": [16, 32, 64],
    "cluster_num": [4, 8],
    "lr": [1e-3, 5e-4],
    "weight_decay": [1e-4, 1e-3],
    "dropout": [0.5],
}


def build_combos(grid):
    keys = list(grid.keys())
    for vals in itertools.product(*grid.values()):
        yield dict(zip(keys, vals))


def run_one(combo, train_dl, val_dl, test_dl, sample_dataset, results_path, monitor):
    """Train + evaluate one hyperparameter combination."""
    print(f"\n{'=' * 60}")
    print(f"Config: {combo}")
    print(f"{'=' * 60}")

    model = GRASP(
        dataset=sample_dataset,
        embedding_dim=combo["embedding_dim"],
        hidden_dim=combo["hidden_dim"],
        cluster_num=combo["cluster_num"],
        block="ConCare",
        dropout=combo["dropout"],
    )

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params:,}")

    trainer = Trainer(
        model=model,
        metrics=["roc_auc", "pr_auc", "f1", "accuracy"],
    )

    t0 = time.time()
    trainer.train(
        train_dataloader=train_dl,
        val_dataloader=val_dl,
        epochs=50,
        monitor=monitor,
        patience=10,
        weight_decay=combo["weight_decay"],
        optimizer_params={"lr": combo["lr"]},
    )
    train_time = time.time() - t0

    test_results = trainer.evaluate(test_dl)

    row = {**combo, "n_params": n_params, "train_time_s": round(train_time, 1)}
    row.update({f"test_{k}": round(v, 4) for k, v in test_results.items()})

    # Append to CSV
    file_exists = os.path.exists(results_path)
    with open(results_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)

    print(f"\nTest results: {test_results}")
    print(f"Train time: {train_time:.0f}s")

    # Free GPU memory between runs
    del model, trainer
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    return test_results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        default="https://storage.googleapis.com/pyhealth/Synthetic_MIMIC-III",
        help="Path to MIMIC-III data (local dir or GCS URL)",
    )
    parser.add_argument(
        "--cache-dir",
        default=None,
        help="Cache directory for parsed dataset (reused across runs)",
    )
    parser.add_argument("--dev", action="store_true", help="Use dev subset")
    parser.add_argument(
        "--code-mapping", action="store_true", help="Enable ICD→CCS code mapping"
    )
    parser.add_argument(
        "--monitor", default="roc_auc",
        help="Metric to monitor for early stopping (default: roc_auc, matching notebook)",
    )
    parser.add_argument(
        "--output", default="sweep_results.csv", help="CSV output path"
    )
    args = parser.parse_args()

    cache_dir = args.cache_dir or tempfile.mkdtemp(prefix="pyhealth_sweep_")

    # ── Data ────────────────────────────────────────────────────────────
    print("Loading MIMIC-III...")
    print(f"  root: {args.root}")
    print(f"  cache_dir: {cache_dir}")
    base_dataset = MIMIC3Dataset(
        root=args.root,
        tables=["DIAGNOSES_ICD", "PROCEDURES_ICD", "PRESCRIPTIONS"],
        cache_dir=cache_dir,
        dev=args.dev,
    )
    base_dataset.stats()

    if args.code_mapping:
        task = MortalityPredictionMIMIC3(
            code_mapping={
                "conditions": ("ICD9CM", "CCSCM"),
                "procedures": ("ICD9PROC", "CCSPROC"),
                "drugs": ("NDC", "ATC"),
            }
        )
    else:
        task = MortalityPredictionMIMIC3()

    samples = base_dataset.set_task(task)

    # ── Dataset stats (matches notebook cells 4 & 6) ───────────────────
    print(f"\nGenerated {len(samples)} samples")
    print(f"Input schema: {samples.input_schema}")
    print(f"Output schema: {samples.output_schema}")

    print("\nProcessor Vocabulary Sizes:")
    for key, proc in samples.input_processors.items():
        if hasattr(proc, "code_vocab"):
            print(f"  {key}: {len(proc.code_vocab)} codes (including <pad>, <unk>)")

    mortality_count = sum(float(s.get("mortality", 0)) for s in samples)
    print(f"\nMortality rate: {mortality_count / len(samples) * 100:.2f}%")
    print(f"Positive: {int(mortality_count)}, Negative: {len(samples) - int(mortality_count)}")

    train_ds, val_ds, test_ds = split_by_patient(samples, [0.8, 0.1, 0.1], seed=42)
    print(f"\nTraining samples: {len(train_ds)}")
    print(f"Validation samples: {len(val_ds)}")
    print(f"Test samples: {len(test_ds)}")

    train_dl = get_dataloader(train_ds, batch_size=32, shuffle=True)
    val_dl = get_dataloader(val_ds, batch_size=32, shuffle=False)
    test_dl = get_dataloader(test_ds, batch_size=32, shuffle=False)
    print(f"\nTraining batches: {len(train_dl)}")
    print(f"Validation batches: {len(val_dl)}")
    print(f"Test batches: {len(test_dl)}")

    # ── Sweep ───────────────────────────────────────────────────────────
    combos = list(build_combos(GRID))
    print(f"\nTotal configurations: {len(combos)}")

    for i, combo in enumerate(combos, 1):
        print(f"\n[{i}/{len(combos)}]")
        try:
            run_one(combo, train_dl, val_dl, test_dl, samples, args.output, args.monitor)
        except Exception as e:
            print(f"FAILED: {e}")
            # Log failure and continue
            with open(args.output, "a") as f:
                f.write(f"# FAILED: {combo} — {e}\n")

    print(f"\nDone. Results saved to {args.output}")


if __name__ == "__main__":
    main()
