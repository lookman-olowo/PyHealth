"""Unit tests for LMM (Neural Long-term Memory) model.

Author: Colton Loew

Tests cover:
    - LMMLayer forward pass shapes and mask handling
    - LMMLayer edge cases (single visit, all-zeros mask, determinism)
    - LMM model initialization, forward, backward, embed flag
    - GRASP integration with block="LMM"
    - Custom hyperparameters (memory_depth, ablation flags)
"""

import unittest

import torch
from pyhealth.datasets import create_sample_dataset, get_dataloader
from pyhealth.models.lmm import LMM, LMMLayer

SAMPLES = [
    {
        "patient_id": "patient-0",
        "visit_id": "visit-0",
        "conditions": ["cond-33", "cond-86", "cond-80"],
        "procedures": ["proc-12", "proc-45"],
        "label": 1,
    },
    {
        "patient_id": "patient-1",
        "visit_id": "visit-1",
        "conditions": ["cond-12", "cond-52"],
        "procedures": ["proc-23"],
        "label": 0,
    },
]


class TestLMMLayer(unittest.TestCase):
    """Tests for the standalone LMMLayer module."""

    def setUp(self):
        torch.manual_seed(42)
        self.input_size = 8
        self.hidden_size = 4
        self.batch_size = 2
        self.seq_len = 5
        self.layer = LMMLayer(
            self.input_size, self.hidden_size,
        )

    def test_output_shapes(self):
        x = torch.randn(
            self.batch_size, self.seq_len, self.input_size,
        )
        outputs, last = self.layer(x)
        self.assertEqual(
            outputs.shape,
            (self.batch_size, self.seq_len, self.hidden_size),
        )
        self.assertEqual(
            last.shape, (self.batch_size, self.hidden_size),
        )

    def test_with_mask(self):
        x = torch.randn(
            self.batch_size, self.seq_len, self.input_size,
        )
        mask = torch.tensor([[1, 1, 1, 0, 0], [1, 1, 0, 0, 0]])
        outputs, last = self.layer(x, mask)
        self.assertEqual(
            last.shape, (self.batch_size, self.hidden_size),
        )

    def test_gradient_flows(self):
        x = torch.randn(
            self.batch_size, self.seq_len, self.input_size,
        )
        _, last = self.layer(x)
        last.sum().backward()
        has_grad = any(
            p.grad is not None for p in self.layer.parameters()
        )
        self.assertTrue(has_grad)

    def test_sequence_length_one(self):
        x = torch.randn(self.batch_size, 1, self.input_size)
        outputs, last = self.layer(x)
        self.assertEqual(outputs.shape, (self.batch_size, 1, self.hidden_size))
        self.assertEqual(last.shape, (self.batch_size, self.hidden_size))

    def test_all_zeros_mask(self):
        x = torch.randn(
            self.batch_size, self.seq_len, self.input_size,
        )
        mask = torch.zeros(self.batch_size, self.seq_len, dtype=torch.int)
        outputs, last = self.layer(x, mask)
        # lengths clamped to 1, so last = outputs[:, 0, :]
        self.assertEqual(
            last.shape, (self.batch_size, self.hidden_size),
        )

    def test_determinism(self):
        x = torch.randn(
            self.batch_size, self.seq_len, self.input_size,
        )
        torch.manual_seed(42)
        layer_a = LMMLayer(self.input_size, self.hidden_size)
        torch.manual_seed(42)
        layer_b = LMMLayer(self.input_size, self.hidden_size)

        layer_a.eval()
        layer_b.eval()
        with torch.no_grad():
            out_a, last_a = layer_a(x)
            out_b, last_b = layer_b(x)
        self.assertTrue(torch.allclose(out_a, out_b))
        self.assertTrue(torch.allclose(last_a, last_b))


class TestLMM(unittest.TestCase):
    """Tests for the full LMM BaseModel."""

    @classmethod
    def setUpClass(cls):
        torch.manual_seed(42)
        cls.dataset = create_sample_dataset(
            samples=SAMPLES,
            input_schema={
                "conditions": "sequence",
                "procedures": "sequence",
            },
            output_schema={"label": "binary"},
            dataset_name="test",
        )
        cls.loader = get_dataloader(
            cls.dataset, batch_size=2, shuffle=True,
        )
        cls.batch = next(iter(cls.loader))

    def _make_model(self, **kwargs):
        defaults = dict(
            dataset=self.dataset,
            embedding_dim=8,
            hidden_dim=4,
        )
        defaults.update(kwargs)
        return LMM(**defaults)

    def test_model_initialization(self):
        model = self._make_model()
        self.assertIsInstance(model, LMM)
        self.assertEqual(model.embedding_dim, 8)
        self.assertEqual(model.hidden_dim, 4)
        self.assertEqual(len(model.feature_keys), 2)
        self.assertEqual(model.label_key, "label")

    def test_model_forward(self):
        model = self._make_model()
        with torch.no_grad():
            ret = model(**self.batch)
        for key in ("loss", "y_prob", "y_true", "logit"):
            self.assertIn(key, ret)
        self.assertEqual(ret["y_prob"].shape[0], 2)
        self.assertEqual(ret["y_prob"].shape[1], 1)
        self.assertEqual(ret["loss"].dim(), 0)
        self.assertFalse(torch.isnan(ret["loss"]))

    def test_model_backward(self):
        model = self._make_model()
        ret = model(**self.batch)
        ret["loss"].backward()
        has_grad = any(
            p.grad is not None for p in model.parameters()
        )
        self.assertTrue(has_grad)

    def test_model_with_embedding(self):
        model = self._make_model()
        batch = dict(self.batch)
        batch["embed"] = True
        with torch.no_grad():
            ret = model(**batch)
        self.assertIn("embed", ret)
        # 2 features * hidden_dim=4
        self.assertEqual(ret["embed"].shape, (2, 2 * 4))

    def test_custom_hyperparameters(self):
        model = self._make_model(
            embedding_dim=16,
            hidden_dim=8,
            memory_depth=3,
        )
        with torch.no_grad():
            ret = model(**self.batch)
        for key in ("loss", "y_prob", "y_true", "logit"):
            self.assertIn(key, ret)

    def test_no_momentum(self):
        model = self._make_model(use_momentum=False)
        with torch.no_grad():
            ret = model(**self.batch)
        self.assertFalse(torch.isnan(ret["loss"]))

    def test_no_weight_decay(self):
        model = self._make_model(use_weight_decay=False)
        with torch.no_grad():
            ret = model(**self.batch)
        self.assertFalse(torch.isnan(ret["loss"]))


class TestGRASPWithLMM(unittest.TestCase):
    """Tests for GRASP with block='LMM'."""

    @classmethod
    def setUpClass(cls):
        torch.manual_seed(42)
        cls.dataset = create_sample_dataset(
            samples=SAMPLES,
            input_schema={
                "conditions": "sequence",
                "procedures": "sequence",
            },
            output_schema={"label": "binary"},
            dataset_name="test",
        )
        cls.loader = get_dataloader(
            cls.dataset, batch_size=2, shuffle=True,
        )
        cls.batch = next(iter(cls.loader))

    def test_grasp_lmm_forward(self):
        from pyhealth.models import GRASP

        model = GRASP(
            dataset=self.dataset,
            embedding_dim=8,
            hidden_dim=4,
            cluster_num=2,
            block="LMM",
        )
        with torch.no_grad():
            ret = model(**self.batch)
        for key in ("loss", "y_prob", "y_true", "logit"):
            self.assertIn(key, ret)
        self.assertFalse(torch.isnan(ret["loss"]))

    def test_grasp_lmm_backward(self):
        from pyhealth.models import GRASP

        model = GRASP(
            dataset=self.dataset,
            embedding_dim=8,
            hidden_dim=4,
            cluster_num=2,
            block="LMM",
        )
        ret = model(**self.batch)
        ret["loss"].backward()
        has_grad = any(
            p.grad is not None for p in model.parameters()
        )
        self.assertTrue(has_grad)


if __name__ == "__main__":
    unittest.main()
