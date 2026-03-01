"""Tests for the GRASP sweep script utilities and end-to-end mini sweep.

Tests cover grid building, resume logic, output directory naming,
and a minimal 1-config sweep using synthetic data.
"""

import csv
import json
import os
import sys
import tempfile
import unittest

import torch

# Import sweep utilities
sys.path.insert(
    0,
    os.path.join(
        os.path.dirname(__file__),
        "..",
        "..",
        "examples",
        "mortality_prediction",
    ),
)
from sweep_grasp import (
    DEFAULT_GRIDS,
    build_combos,
    combo_key,
    load_completed,
    make_output_dir,
    run_one,
)

from pyhealth.datasets import create_sample_dataset, get_dataloader
from pyhealth.models import GRASP


class TestBuildCombos(unittest.TestCase):
    """Tests for grid combination generation."""

    def test_simple_grid(self):
        grid = {"a": [1, 2], "b": [3, 4]}
        combos = list(build_combos(grid))
        self.assertEqual(len(combos), 4)
        self.assertIn({"a": 1, "b": 3}, combos)
        self.assertIn({"a": 2, "b": 4}, combos)

    def test_single_values(self):
        grid = {"a": [1], "b": [2], "c": [3]}
        combos = list(build_combos(grid))
        self.assertEqual(len(combos), 1)
        self.assertEqual(combos[0], {"a": 1, "b": 2, "c": 3})

    def test_default_grid_counts(self):
        """Verify expected config counts for each backbone."""
        gru = list(build_combos(DEFAULT_GRIDS["GRU"]))
        concare = list(build_combos(DEFAULT_GRIDS["ConCare"]))
        lstm = list(build_combos(DEFAULT_GRIDS["LSTM"]))
        # GRU: 3*3*2*2*3*1 = 108
        self.assertEqual(len(gru), 108)
        # ConCare: 3*3*2*2*2*1 = 72
        self.assertEqual(len(concare), 72)
        # LSTM same as GRU
        self.assertEqual(len(lstm), 108)


class TestComboKey(unittest.TestCase):
    """Tests for combo hashing used in resume logic."""

    def test_key_from_dict(self):
        combo = {
            "embedding_dim": 16,
            "hidden_dim": 32,
            "cluster_num": 4,
            "lr": 1e-3,
            "weight_decay": 1e-4,
            "dropout": 0.5,
        }
        key = combo_key(combo)
        self.assertEqual(key, (16.0, 32.0, 4.0, 1e-3, 1e-4, 0.5))

    def test_key_consistency(self):
        """Same combo should produce the same key regardless of int/float."""
        combo_int = {"embedding_dim": 16, "hidden_dim": 32, "cluster_num": 4,
                     "lr": 0.001, "weight_decay": 0.0001, "dropout": 0.5}
        combo_float = {"embedding_dim": 16.0, "hidden_dim": 32.0, "cluster_num": 4.0,
                       "lr": 1e-3, "weight_decay": 1e-4, "dropout": 0.5}
        self.assertEqual(combo_key(combo_int), combo_key(combo_float))


class TestLoadCompleted(unittest.TestCase):
    """Tests for CSV resume parsing."""

    def test_nonexistent_file(self):
        completed = load_completed("/tmp/nonexistent_sweep_results_xyz.csv")
        self.assertEqual(completed, set())

    def test_parses_csv(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            writer = csv.writer(f)
            writer.writerow(["embedding_dim", "hidden_dim", "cluster_num",
                             "lr", "weight_decay", "dropout", "n_params",
                             "test_roc_auc"])
            writer.writerow([16, 32, 4, 0.001, 0.0001, 0.5, 50000, 0.65])
            writer.writerow([8, 16, 8, 0.0005, 0.001, 0.5, 20000, 0.55])
            f.flush()
            path = f.name

        try:
            completed = load_completed(path)
            self.assertEqual(len(completed), 2)
            self.assertIn((16.0, 32.0, 4.0, 0.001, 0.0001, 0.5), completed)
            self.assertIn((8.0, 16.0, 8.0, 0.0005, 0.001, 0.5), completed)
        finally:
            os.unlink(path)

    def test_skips_comment_lines(self):
        """Failed runs are logged as # comments and should be skipped."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            writer = csv.writer(f)
            writer.writerow(["embedding_dim", "hidden_dim", "cluster_num",
                             "lr", "weight_decay", "dropout", "n_params",
                             "test_roc_auc"])
            writer.writerow([16, 32, 4, 0.001, 0.0001, 0.5, 50000, 0.65])
            f.write("# FAILED: {'embedding_dim': 8} — OOM error\n")
            f.flush()
            path = f.name

        try:
            completed = load_completed(path)
            self.assertEqual(len(completed), 1)
        finally:
            os.unlink(path)


class TestMakeOutputDir(unittest.TestCase):
    """Tests for output directory naming."""

    def test_with_mapping(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = make_output_dir("GRU", True, output_root=tmp)
            self.assertIn("GRU_", path)
            self.assertIn("_with-mapping", path)
            self.assertTrue(os.path.isdir(path))

    def test_no_mapping(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = make_output_dir("ConCare", False, output_root=tmp)
            self.assertIn("ConCare_", path)
            self.assertIn("_no-mapping", path)
            self.assertTrue(os.path.isdir(path))


class TestRunOne(unittest.TestCase):
    """Integration test: run a single config on tiny synthetic data."""

    def setUp(self):
        torch.manual_seed(42)
        self.samples = [
            {"patient_id": "p0", "visit_id": "v0",
             "conditions": ["c1", "c2", "c3"], "procedures": ["p1"], "label": 0},
            {"patient_id": "p0", "visit_id": "v1",
             "conditions": ["c1", "c3"], "procedures": ["p2", "p3"], "label": 1},
            {"patient_id": "p1", "visit_id": "v0",
             "conditions": ["c2", "c4"], "procedures": ["p1"], "label": 0},
            {"patient_id": "p1", "visit_id": "v1",
             "conditions": ["c3", "c4", "c1"], "procedures": ["p2"], "label": 1},
        ]
        self.dataset = create_sample_dataset(
            samples=self.samples,
            input_schema={"conditions": "sequence", "procedures": "sequence"},
            output_schema={"label": "binary"},
            dataset_name="test",
        )
        self.train_dl = get_dataloader(self.dataset, batch_size=2, shuffle=True)
        self.val_dl = get_dataloader(self.dataset, batch_size=2, shuffle=False)
        self.test_dl = get_dataloader(self.dataset, batch_size=2, shuffle=False)

    def test_gru_single_config(self):
        """Run one GRU config end-to-end and verify CSV output."""
        with tempfile.TemporaryDirectory() as tmp:
            results_path = os.path.join(tmp, "results.csv")
            combo = {
                "embedding_dim": 4,
                "hidden_dim": 4,
                "cluster_num": 2,
                "lr": 1e-3,
                "weight_decay": 0.0,
                "dropout": 0.5,
            }
            result = run_one(
                combo, "GRU", self.train_dl, self.val_dl, self.test_dl,
                self.dataset, results_path, "roc_auc",
            )

            # Check return value
            self.assertIn("roc_auc", result)
            self.assertIn("pr_auc", result)
            self.assertIn("f1", result)

            # Check CSV was written
            self.assertTrue(os.path.exists(results_path))
            with open(results_path) as f:
                reader = csv.DictReader(f)
                rows = list(reader)
            self.assertEqual(len(rows), 1)
            self.assertEqual(int(rows[0]["embedding_dim"]), 4)
            self.assertIn("n_params", rows[0])
            self.assertIn("test_roc_auc", rows[0])

    def test_concare_single_config(self):
        """Run one ConCare config end-to-end without crashing."""
        with tempfile.TemporaryDirectory() as tmp:
            results_path = os.path.join(tmp, "results.csv")
            combo = {
                "embedding_dim": 4,
                "hidden_dim": 4,
                "cluster_num": 2,
                "lr": 1e-3,
                "weight_decay": 0.0,
                "dropout": 0.5,
            }
            result = run_one(
                combo, "ConCare", self.train_dl, self.val_dl, self.test_dl,
                self.dataset, results_path, "roc_auc",
            )
            self.assertIn("roc_auc", result)

    def test_resume_skips_completed(self):
        """Verify that a completed config is detected and can be skipped."""
        with tempfile.TemporaryDirectory() as tmp:
            results_path = os.path.join(tmp, "results.csv")
            combo = {
                "embedding_dim": 4,
                "hidden_dim": 4,
                "cluster_num": 2,
                "lr": 1e-3,
                "weight_decay": 0.0,
                "dropout": 0.5,
            }
            # Run once
            run_one(
                combo, "GRU", self.train_dl, self.val_dl, self.test_dl,
                self.dataset, results_path, "roc_auc",
            )

            # Load completed and verify the combo is found
            completed = load_completed(results_path)
            self.assertIn(combo_key(combo), completed)


if __name__ == "__main__":
    unittest.main()
