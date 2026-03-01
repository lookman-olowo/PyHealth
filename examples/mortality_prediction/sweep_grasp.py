"""Hyperparameter grid sweep: GRASP backbone comparison on MIMIC-III mortality.

Paper runs (embedding strategy x backbone comparison):
    ROOT=/path/to/mimic3

    # ── 1. GRU + code_mapping (primary sweep) ──────────────────────────
    python sweep_grasp.py --block GRU --code-mapping --root $ROOT

    # ── 2. GRU + no mapping (baseline) ─────────────────────────────────
    python sweep_grasp.py --block GRU --root $ROOT

    # ── 3. ConCare + code_mapping (done — 72 configs, all F1=0.0) ──────
    # python sweep_grasp.py --block ConCare --code-mapping --root $ROOT

    # ── 4. ConCare + no mapping (skip — proven architectural mismatch) ─
    # python sweep_grasp.py --block ConCare --root $ROOT

    # ── 5. GRU + KEEP embeddings (after KEEP implementation) ───────────
    # TODO: add --keep-embeddings flag

    # ── 6. Titan + code_mapping (extension, after Titan backbone) ──────
    # TODO: add --block Titan

    # ── 7. Titan + KEEP embeddings (extension) ─────────────────────────
    # TODO: add --block Titan --keep-embeddings

Usage:
    # GRU backbone with code_mapping
    python sweep_grasp.py --block GRU --code-mapping --root /path/to/mimic3

    # Custom grid override
    python sweep_grasp.py --block GRU --grid '{"embedding_dim":[16,32],"hidden_dim":[32]}'

    # Resume a crashed sweep (skips completed configs)
    python sweep_grasp.py --block GRU --code-mapping --resume --output-dir sweep/GRU_2026XXXX_XXXXXX_with-mapping

Results are saved to:
    sweep/{BLOCK}_{YYYYMMDD}_{HHMMSS}_{mapping}/results.csv

Naming examples:
    sweep/GRU_20260301_143022_with-mapping/results.csv
    sweep/ConCare_20260228_091500_no-mapping/results.csv

Long-running sweeps (surviving timeouts/disconnects):
    # Use tmux so the sweep survives notebook/SSH disconnects
    tmux new -s sweep
    python sweep_grasp.py --block GRU --code-mapping --root $ROOT
    # ctrl+b, d to detach — reconnect later with: tmux attach -t sweep

    # Or use nohup
    nohup python sweep_grasp.py --block GRU --code-mapping --root $ROOT > sweep.log 2>&1 &

    # If it crashes, resume from where it left off
    python sweep_grasp.py --block GRU --code-mapping --resume \
        --output-dir sweep/GRU_2026XXXX_XXXXXX_with-mapping
"""

import argparse
import csv
import itertools
import json
import os
import tempfile
import time
from datetime import datetime

import torch

from pyhealth.datasets import MIMIC3Dataset, get_dataloader, split_by_patient
from pyhealth.models import GRASP
from pyhealth.tasks import MortalityPredictionMIMIC3
from pyhealth.trainer import Trainer

# ── Default grids per backbone ──────────────────────────────────────────
DEFAULT_GRIDS = {
    "GRU": {
        "embedding_dim": [16, 32, 64],
        "hidden_dim": [16, 32, 64],
        "cluster_num": [4, 8],
        "lr": [1e-3, 5e-4],
        "weight_decay": [0.0, 1e-4, 1e-3],
        "dropout": [0.5],
    },
    "ConCare": {
        "embedding_dim": [8, 16, 32],
        "hidden_dim": [16, 32, 64],
        "cluster_num": [4, 8],
        "lr": [1e-3, 5e-4],
        "weight_decay": [1e-4, 1e-3],
        "dropout": [0.5],
    },
    "LSTM": {
        "embedding_dim": [16, 32, 64],
        "hidden_dim": [16, 32, 64],
        "cluster_num": [4, 8],
        "lr": [1e-3, 5e-4],
        "weight_decay": [0.0, 1e-4, 1e-3],
        "dropout": [0.5],
    },
}


def build_combos(grid):
    keys = list(grid.keys())
    for vals in itertools.product(*grid.values()):
        yield dict(zip(keys, vals))


def load_completed(results_path):
    """Load already-completed configs from an existing CSV to enable resume."""
    completed = set()
    if not os.path.exists(results_path):
        return completed
    with open(results_path, "r") as f:
        for line in f:
            if line.startswith("#"):
                continue
        f.seek(0)
        reader = csv.DictReader(
            (line for line in f if not line.startswith("#"))
        )
        for row in reader:
            try:
                key = (
                    float(row["embedding_dim"]),
                    float(row["hidden_dim"]),
                    float(row["cluster_num"]),
                    float(row["lr"]),
                    float(row["weight_decay"]),
                    float(row["dropout"]),
                )
                completed.add(key)
            except (KeyError, ValueError):
                continue
    return completed


def combo_key(combo):
    return (
        float(combo["embedding_dim"]),
        float(combo["hidden_dim"]),
        float(combo["cluster_num"]),
        float(combo["lr"]),
        float(combo["weight_decay"]),
        float(combo["dropout"]),
    )


def run_one(combo, block, train_dl, val_dl, test_dl, sample_dataset, results_path, monitor):
    """Train + evaluate one hyperparameter combination."""
    print(f"\n{'=' * 60}")
    print(f"Block: {block} | Config: {combo}")
    print(f"{'=' * 60}")

    model = GRASP(
        dataset=sample_dataset,
        embedding_dim=combo["embedding_dim"],
        hidden_dim=combo["hidden_dim"],
        cluster_num=combo["cluster_num"],
        block=block,
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
        patience=15,
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


def make_output_dir(block, code_mapping, output_root="sweep"):
    """Create dated output directory."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    mapping_tag = "with-mapping" if code_mapping else "no-mapping"
    dirname = f"{block}_{timestamp}_{mapping_tag}"
    path = os.path.join(output_root, dirname)
    os.makedirs(path, exist_ok=True)
    return path


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--block", default="GRU", choices=["GRU", "ConCare", "LSTM"],
        help="Backbone model (default: GRU)",
    )
    parser.add_argument(
        "--root",
        default="https://storage.googleapis.com/pyhealth/Synthetic_MIMIC-III",
        help="Path to MIMIC-III data (local dir or GCS URL)",
    )
    parser.add_argument(
        "--cache-dir", default=None,
        help="Cache directory for parsed dataset (reused across runs)",
    )
    parser.add_argument("--dev", action="store_true", help="Use dev subset")
    parser.add_argument(
        "--code-mapping", action="store_true", help="Enable ICD→CCS code mapping",
    )
    parser.add_argument(
        "--monitor", default="roc_auc",
        help="Metric to monitor for early stopping (default: roc_auc)",
    )
    parser.add_argument(
        "--grid", default=None,
        help='JSON string to override the default grid, e.g. \'{"embedding_dim":[16,32]}\'',
    )
    parser.add_argument(
        "--output-root", default="sweep",
        help="Root directory for sweep outputs (default: sweep/)",
    )
    parser.add_argument(
        "--output-dir", default=None,
        help="Exact output directory (overrides auto-naming)",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Skip configs already in the output CSV",
    )
    args = parser.parse_args()

    # ── Grid ────────────────────────────────────────────────────────────
    if args.grid:
        grid = json.loads(args.grid)
        base = DEFAULT_GRIDS.get(args.block, DEFAULT_GRIDS["GRU"])
        for k, v in base.items():
            grid.setdefault(k, v)
    else:
        grid = DEFAULT_GRIDS.get(args.block, DEFAULT_GRIDS["GRU"])

    # ── Output ──────────────────────────────────────────────────────────
    if args.output_dir:
        out_dir = args.output_dir
        os.makedirs(out_dir, exist_ok=True)
    else:
        out_dir = make_output_dir(args.block, args.code_mapping, args.output_root)

    results_path = os.path.join(out_dir, "results.csv")

    # Save run config for reproducibility
    config = {
        "block": args.block,
        "root": args.root,
        "dev": args.dev,
        "code_mapping": args.code_mapping,
        "monitor": args.monitor,
        "grid": grid,
    }
    with open(os.path.join(out_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    cache_dir = args.cache_dir or tempfile.mkdtemp(prefix="pyhealth_sweep_")

    # ── Data ────────────────────────────────────────────────────────────
    print(f"Sweep: GRASP + {args.block}")
    print(f"Output: {out_dir}")
    print(f"Loading MIMIC-III...")
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

    # ── Dataset stats ───────────────────────────────────────────────────
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
    combos = list(build_combos(grid))
    print(f"\nGrid: {grid}")
    print(f"Total configurations: {len(combos)}")

    # Resume support
    completed = load_completed(results_path) if args.resume else set()
    if completed:
        print(f"Resuming: {len(completed)} configs already completed, skipping them")

    for i, combo in enumerate(combos, 1):
        if args.resume and combo_key(combo) in completed:
            print(f"\n[{i}/{len(combos)}] SKIP (already completed)")
            continue

        print(f"\n[{i}/{len(combos)}]")
        try:
            run_one(combo, args.block, train_dl, val_dl, test_dl, samples, results_path, args.monitor)
        except Exception as e:
            print(f"FAILED: {e}")
            with open(results_path, "a") as f:
                f.write(f"# FAILED: {combo} — {e}\n")

    print(f"\nDone. Results saved to {results_path}")


if __name__ == "__main__":
    main()
