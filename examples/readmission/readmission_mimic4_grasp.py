import tempfile

import torch
from pyhealth.datasets import MIMIC4Dataset
from pyhealth.datasets import get_dataloader, split_by_patient
from pyhealth.models import GRASP
from pyhealth.tasks import ReadmissionPredictionMIMIC4
from pyhealth.trainer import Trainer


if __name__ == "__main__":
    # Load MIMIC-III dataset
    base_dataset = MIMIC4Dataset(
        # ehr_root="https://physionet.org/files/mimic-iv-demo/2.2/",
        ehr_root="/home/cmbeard2",
        ehr_tables=["diagnoses_icd", "procedures_icd", "prescriptions"],
        cache_dir=tempfile.TemporaryDirectory().name,
        dev=False,
    )
    
    base_dataset.stats()

    # Define task without code mapping
    task = ReadmissionPredictionMIMIC4(exclude_minors=False)

    samples = base_dataset.set_task(task)

    print(f"Generated {len(samples)} samples")
    print(f"\nInput schema: {samples.input_schema}")
    print(f"Output schema: {samples.output_schema}")

    print("Sample structure:")
    print(samples[0])

    print("\n" + "=" * 50)
    print("Processor Vocabulary Sizes:")
    print("=" * 50)
    for key, proc in samples.input_processors.items():
        if hasattr(proc, "code_vocab"):
            print(f"{key}: {len(proc.code_vocab)} codes (including <pad>, <unk>)")

    readmission_count = sum(float(s.get("readmission", 0)) for s in samples)
    print(f"\nTotal samples: {len(samples)}")
    print(f"Readmission rate: {readmission_count / len(samples) * 100:.2f}%")
    print(f"Positive samples: {int(readmission_count)}")
    print(f"Negative samples: {len(samples) - int(readmission_count)}")

    # Split dataset
    train_dataset, val_dataset, test_dataset = split_by_patient(
        samples, [0.8, 0.1, 0.1], seed=42
    )

    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print(f"Test samples: {len(test_dataset)}")

    # Create dataloaders
    train_dataloader = get_dataloader(train_dataset, batch_size=32, shuffle=True)
    val_dataloader = get_dataloader(val_dataset, batch_size=32, shuffle=False)
    test_dataloader = get_dataloader(test_dataset, batch_size=32, shuffle=False)

    print(f"Training batches: {len(train_dataloader)}")
    print(f"Validation batches: {len(val_dataloader)}")
    print(f"Test batches: {len(test_dataloader)}")

    # Initialize model
    model = GRASP(
        dataset=samples,
        embedding_dim=100,
        hidden_dim=32,
        cluster_num=8,
        block="GRU",
        dropout=0.5,
    )

    print(f"Model initialized with {sum(p.numel() for p in model.parameters()):,} parameters")
    print("\nModel architecture:")
    print(model)

    # Train
    trainer = Trainer(
        model=model,
        metrics=["roc_auc", "pr_auc", "accuracy", "f1"],
    )

    trainer.train(
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        epochs=50,
        monitor="pr_auc",
        optimizer_params={"lr": 1e-3},
        weight_decay=1e-4,
        max_grad_norm=1.0,
    )

    # Evaluate
    test_results = trainer.evaluate(test_dataloader)

    print("\n" + "=" * 50)
    print("Test Set Performance (NO code_mapping)")
    print("=" * 50)
    for metric, value in test_results.items():
        print(f"{metric}: {value:.4f}")

    # Extract embeddings + sample predictions
    model.eval()
    test_batch = next(iter(test_dataloader))
    test_batch["embed"] = True

    with torch.no_grad():
        output = model(**test_batch)

    print(f"Embedding shape: {output['embed'].shape}")
    print(f"  - Batch size: {output['embed'].shape[0]}")
    print(f"  - Embedding dim: {output['embed'].shape[1]}")

    print("\n" + "=" * 50)
    print("Sample Predictions:")
    print("=" * 50)
    predictions = output["y_prob"].cpu().numpy()
    true_labels = output["y_true"].cpu().numpy()

    for i in range(min(5, len(predictions))):
        pred = predictions[i][0]
        true = int(true_labels[i][0])
        print(
            f"Patient {i + 1}: Predicted={pred:.3f}, True={true}, "
            f"Prediction={'Readmission' if pred > 0.5 else 'No Readmission'}"
        )
