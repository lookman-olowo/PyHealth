# Author: Colton Loew
# Paper title: Titans: Learning to Memorize at Test Time
# Paper link: https://arxiv.org/abs/2501.00663
# Description: Neural Long-term Memory (LMM) module from Titans,
#     adapted as a sequence backbone for EHR clinical prediction.
#     Uses surprise-based memorization with momentum and adaptive
#     forgetting to preferentially encode rare clinical events.

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from pyhealth.datasets import SampleDataset
from pyhealth.models import BaseModel
from pyhealth.models.embedding import EmbeddingModel


def _build_memory_mlp(
    hidden_size: int,
    depth: int = 2,
) -> nn.Sequential:
    """Build the memory MLP used as the neural long-term memory.

    Args:
        hidden_size: input and output dimension of the memory.
        depth: number of layers. Must be >= 2 for non-linear memory.

    Returns:
        An nn.Sequential MLP with SiLU activations.
    """
    inner = hidden_size * 2
    layers: list[nn.Module] = []
    layers.append(nn.Linear(hidden_size, inner))
    layers.append(nn.SiLU())
    for _ in range(depth - 2):
        layers.append(nn.Linear(inner, inner))
        layers.append(nn.SiLU())
    layers.append(nn.Linear(inner, hidden_size))
    return nn.Sequential(*layers)


class LMMLayer(nn.Module):
    """Neural Long-term Memory layer for sequence encoding.

    Paper: Ali Behrouz et al. Titans: Learning to Memorize at
        Test Time. arXiv 2501.00663, 2025.

    This layer implements the LMM (Neural Long-term Memory) module
    from Titans as a drop-in replacement for RNNLayer. It uses
    surprise-based gradient updates with momentum to preferentially
    memorize unexpected inputs, making it suited for EHR data where
    rare clinical events drive predictions.

    The memory is a small MLP whose weights evolve per-timestep via
    a surprise signal (gradient of an associative loss). Data-dependent
    gates control momentum decay, forgetting, and learning rate.

    This layer is used in the LMM model and as a backbone in GRASP.
    But it can also be used as a standalone layer.

    Args:
        input_size: input feature size.
        hidden_size: hidden feature size (output dimension).
        memory_depth: number of layers in the memory MLP. Must be
            >= 2 for non-linear memory. Default is 2.
        use_momentum: if True, surprise signal includes momentum
            from previous timesteps. Default is True.
        use_weight_decay: if True, memory weights undergo adaptive
            forgetting each timestep. Default is True.
        dropout: dropout rate. Default is 0.5.

    Examples:
        >>> from pyhealth.models import LMMLayer
        >>> import torch
        >>> x = torch.randn(3, 50, 64)  # [batch, seq_len, input]
        >>> layer = LMMLayer(64, 32)
        >>> outputs, last = layer(x)
        >>> outputs.shape
        torch.Size([3, 50, 32])
        >>> last.shape
        torch.Size([3, 32])
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        memory_depth: int = 2,
        use_momentum: bool = True,
        use_weight_decay: bool = True,
        dropout: float = 0.5,
    ) -> None:
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.memory_depth = memory_depth
        self.use_momentum = use_momentum
        self.use_weight_decay = use_weight_decay

        # Key, value, query projections
        self.proj_k = nn.Linear(input_size, hidden_size)
        self.proj_v = nn.Linear(input_size, hidden_size)
        self.proj_q = nn.Linear(input_size, hidden_size)

        # Memory MLP — weights serve as initial memory state
        self.memory = _build_memory_mlp(hidden_size, memory_depth)

        # Data-dependent gates
        self.gate_theta = nn.Linear(input_size, 1)  # learning rate
        if use_momentum:
            self.gate_eta = nn.Linear(input_size, 1)  # momentum decay
        if use_weight_decay:
            self.gate_alpha = nn.Linear(input_size, 1)  # forgetting

        self.dropout_layer = nn.Dropout(dropout)

    def _surprise_loss(
        self,
        weights: dict[str, torch.Tensor],
        k_t: torch.Tensor,
        v_t: torch.Tensor,
    ) -> torch.Tensor:
        """Associative memory loss: ||memory(k) - v||^2.

        Args:
            weights: current memory MLP weights (plain tensors).
            k_t: key tensor of shape [B, hidden_size].
            v_t: value tensor of shape [B, hidden_size].

        Returns:
            Scalar loss tensor.
        """
        pred = torch.func.functional_call(
            self.memory, weights, (k_t,)
        )
        return (pred - v_t).pow(2).mean()

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward propagation.

        Args:
            x: a tensor of shape [batch size, sequence len, input_size].
            mask: an optional tensor of shape [batch size, sequence len],
                where 1 indicates valid and 0 indicates invalid.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - outputs: [batch size, sequence len, hidden_size],
                    per-step memory retrievals.
                - last_outputs: [batch size, hidden_size],
                    retrieval at the last valid timestep.
        """
        x = self.dropout_layer(x)
        batch_size, seq_len, _ = x.shape

        if mask is None:
            lengths = torch.full(
                (batch_size,), seq_len, dtype=torch.int64
            )
        else:
            lengths = torch.sum(mask.int(), dim=-1).cpu()
        lengths = torch.clamp(lengths, min=1)

        # Clone memory weights as working copy (detached from params)
        mem_weights = {
            k: v.clone()
            for k, v in dict(self.memory.named_parameters()).items()
        }

        # Initialize momentum buffers
        S = {
            k: torch.zeros_like(v) for k, v in mem_weights.items()
        }

        # Compute all projections at once for efficiency
        keys = self.proj_k(x)    # [B, T, H]
        values = self.proj_v(x)  # [B, T, H]
        queries = self.proj_q(x)  # [B, T, H]

        # Compute all gates at once
        theta_all = F.softplus(self.gate_theta(x))  # [B, T, 1]
        if self.use_momentum:
            eta_all = torch.sigmoid(self.gate_eta(x))  # [B, T, 1]
        if self.use_weight_decay:
            alpha_all = torch.sigmoid(
                self.gate_alpha(x)
            )  # [B, T, 1]

        # Pre-allocate output tensor
        outputs = torch.zeros(
            batch_size, seq_len, self.hidden_size,
            device=x.device, dtype=x.dtype,
        )

        # Compute surprise gradient function
        grad_fn = torch.func.grad(self._surprise_loss)

        # Sequential processing over timesteps
        for t in range(seq_len):
            k_t = keys[:, t, :]    # [B, H]
            v_t = values[:, t, :]  # [B, H]
            q_t = queries[:, t, :]  # [B, H]

            # Gate values for this timestep (mean over batch for
            # weight-level updates — weights are shared across batch)
            theta_t = theta_all[:, t, :].mean().clamp(max=1.0)
            if self.use_momentum:
                eta_t = eta_all[:, t, :].mean()
            if self.use_weight_decay:
                alpha_t = alpha_all[:, t, :].mean()

            # Compute surprise: gradient of associative loss
            surprise_grad = grad_fn(mem_weights, k_t, v_t)

            # Clamp surprise gradients to prevent explosion
            surprise_grad = {
                name: g.clamp(-1.0, 1.0)
                for name, g in surprise_grad.items()
            }

            # Update momentum
            for name in mem_weights:
                if self.use_momentum:
                    S[name] = (
                        eta_t * S[name]
                        - theta_t * surprise_grad[name]
                    )
                else:
                    S[name] = -theta_t * surprise_grad[name]

            # Update memory weights
            for name in mem_weights:
                if self.use_weight_decay:
                    mem_weights[name] = (
                        (1 - alpha_t) * mem_weights[name]
                        + S[name]
                    ).detach()
                else:
                    mem_weights[name] = (
                        mem_weights[name] + S[name]
                    ).detach()

            # Retrieve from memory (differentiable for outer training)
            output_t = torch.func.functional_call(
                self.memory, mem_weights, (q_t,)
            )
            outputs[:, t, :] = output_t

        # Extract last valid timestep (matches RNNLayer pattern)
        last_outputs = outputs[
            torch.arange(batch_size), (lengths - 1), :
        ]

        return outputs, last_outputs


class LMM(BaseModel):
    """Neural Long-term Memory model for clinical prediction.

    Paper: Ali Behrouz et al. Titans: Learning to Memorize at
        Test Time. arXiv 2501.00663, 2025.

    This model applies a separate LMM layer for each feature, and
    then concatenates the final hidden states. The concatenated
    representations are fed into a fully connected layer to make
    predictions.

    The LMM layer encodes patient sequences using a surprise-based
    neural memory that preferentially memorizes unexpected inputs.
    This makes it suited for EHR data where rare clinical events
    (e.g., a sudden lab spike or new diagnosis) drive predictions.

    Args:
        dataset: the dataset to train the model. It is used to
            query certain information such as the set of all tokens.
        embedding_dim: the embedding dimension. Default is 128.
        hidden_dim: the hidden dimension. Default is 128.
        **kwargs: other parameters for LMMLayer
            (e.g., memory_depth, use_momentum, use_weight_decay,
            dropout).

    Examples:
        >>> from pyhealth.datasets import create_sample_dataset
        >>> samples = [
        ...     {
        ...         "patient_id": "patient-0",
        ...         "visit_id": "visit-0",
        ...         "conditions": ["cond-33", "cond-86"],
        ...         "procedures": ["proc-12", "proc-45"],
        ...         "label": 1,
        ...     },
        ...     {
        ...         "patient_id": "patient-1",
        ...         "visit_id": "visit-1",
        ...         "conditions": ["cond-12"],
        ...         "procedures": ["proc-23"],
        ...         "label": 0,
        ...     },
        ... ]
        >>> dataset = create_sample_dataset(
        ...     samples=samples,
        ...     input_schema={
        ...         "conditions": "sequence",
        ...         "procedures": "sequence",
        ...     },
        ...     output_schema={"label": "binary"},
        ...     dataset_name="test",
        ... )
        >>> from pyhealth.datasets import get_dataloader
        >>> loader = get_dataloader(dataset, batch_size=2, shuffle=True)
        >>> model = LMM(dataset=dataset, embedding_dim=32, hidden_dim=16)
        >>> batch = next(iter(loader))
        >>> ret = model(**batch)
        >>> sorted(ret.keys())
        ['logit', 'loss', 'y_prob', 'y_true']
    """

    def __init__(
        self,
        dataset: SampleDataset,
        embedding_dim: int = 128,
        hidden_dim: int = 128,
        **kwargs,
    ) -> None:
        super().__init__(dataset=dataset)
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim

        if "input_size" in kwargs:
            raise ValueError("input_size is determined by embedding_dim")
        if "hidden_size" in kwargs:
            raise ValueError(
                "hidden_size is determined by hidden_dim"
            )

        assert len(self.label_keys) == 1, (
            "Only one label key is supported"
        )
        self.label_key = self.label_keys[0]
        self.mode = self.dataset.output_schema[self.label_key]

        self.embedding_model = EmbeddingModel(dataset, embedding_dim)

        # One LMMLayer per feature
        self.lmm = nn.ModuleDict()
        for feature_key in self.dataset.input_processors.keys():
            self.lmm[feature_key] = LMMLayer(
                input_size=embedding_dim,
                hidden_size=hidden_dim,
                **kwargs,
            )

        output_size = self.get_output_size()
        self.fc = nn.Linear(
            len(self.feature_keys) * self.hidden_dim, output_size
        )

    def forward(self, **kwargs) -> Dict[str, torch.Tensor]:
        """Forward propagation.

        The label ``kwargs[self.label_key]`` is a list of labels for
        each patient.

        Args:
            **kwargs: keyword arguments for the model. The keys must
                contain all the feature keys and the label key.

        Returns:
            Dict[str, torch.Tensor]: A dictionary with the following
                keys:
                - loss: a scalar tensor representing the loss.
                - y_prob: a tensor representing the predicted
                    probabilities.
                - y_true: a tensor representing the true labels.
                - logit: a tensor representing the logits.
                - embed (optional): a tensor representing the patient
                    embeddings if requested.
        """
        patient_emb = []
        embedded = self.embedding_model(kwargs)

        for feature_key in self.feature_keys:
            x = embedded[feature_key]
            mask = (torch.abs(x).sum(dim=-1) != 0).int()
            _, x = self.lmm[feature_key](x, mask)
            patient_emb.append(x)

        patient_emb = torch.cat(patient_emb, dim=1)
        logits = self.fc(patient_emb)

        y_true = kwargs[self.label_key].to(self.device)
        loss = self.get_loss_function()(logits, y_true)
        y_prob = self.prepare_y_prob(logits)

        results = {
            "loss": loss,
            "y_prob": y_prob,
            "y_true": y_true,
            "logit": logits,
        }
        if kwargs.get("embed", False):
            results["embed"] = patient_emb
        return results


if __name__ == "__main__":
    from pyhealth.datasets import create_sample_dataset, get_dataloader

    samples = [
        {
            "patient_id": "patient-0",
            "visit_id": "visit-0",
            "conditions": ["cond-33", "cond-86"],
            "procedures": ["proc-12", "proc-45"],
            "label": 1,
        },
        {
            "patient_id": "patient-1",
            "visit_id": "visit-1",
            "conditions": ["cond-12"],
            "procedures": ["proc-23"],
            "label": 0,
        },
    ]

    dataset = create_sample_dataset(
        samples=samples,
        input_schema={
            "conditions": "sequence",
            "procedures": "sequence",
        },
        output_schema={"label": "binary"},
        dataset_name="test",
    )

    loader = get_dataloader(dataset, batch_size=2, shuffle=True)

    model = LMM(dataset=dataset, embedding_dim=32, hidden_dim=16)

    batch = next(iter(loader))
    out = model(**batch)
    print("keys:", sorted(out.keys()))
    out["loss"].backward()
    print("backward pass successful")
