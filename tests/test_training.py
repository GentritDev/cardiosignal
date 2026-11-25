
import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from cardiosignal.training.engine import (
    evaluate_one_epoch,
    train_one_epoch,
)


CLASS_NAMES = ("CLASS_A", "CLASS_B")


class SyntheticECGDataset(Dataset):
    def __init__(self) -> None:
        self.signals = torch.tensor(
            [
                [[1.0, 0.0]],
                [[0.0, 1.0]],
                [[1.0, 1.0]],
                [[0.0, 0.0]],
            ],
            dtype=torch.float32,
        )

        self.labels = torch.tensor(
            [
                [1.0, 0.0],
                [0.0, 1.0],
                [1.0, 1.0],
                [0.0, 0.0],
            ],
            dtype=torch.float32,
        )

    def __len__(self) -> int:
        return len(self.signals)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {
            "signal": self.signals[index],
            "labels": self.labels[index],
        }


def create_test_components():
    dataset = SyntheticECGDataset()
    dataloader = DataLoader(dataset, batch_size=2, shuffle=False)

    model = nn.Sequential(
        nn.Flatten(),
        nn.Linear(2, 2),
    )

    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    device = torch.device("cpu")

    return model, dataloader, loss_fn, optimizer, device


def test_train_one_epoch_updates_model_parameters() -> None:
    model, dataloader, loss_fn, optimizer, device = create_test_components()

    parameters_before = [
        parameter.detach().clone()
        for parameter in model.parameters()
    ]

    results = train_one_epoch(
        model=model,
        dataloader=dataloader,
        loss_fn=loss_fn,
        optimizer=optimizer,
        device=device,
        class_names=CLASS_NAMES,
    )

    parameters_after = list(model.parameters())

    assert any(
        not torch.equal(before, after)
        for before, after in zip(parameters_before, parameters_after)
    )

    assert results["num_samples"] == 4
    assert results["num_batches"] == 2
    assert torch.isfinite(torch.tensor(results["loss"]))
    assert "f1_macro" in results
    assert "auroc_macro" in results
    assert "auprc_macro" in results


def test_evaluate_one_epoch_does_not_update_parameters() -> None:
    model, dataloader, loss_fn, _, device = create_test_components()

    parameters_before = [
        parameter.detach().clone()
        for parameter in model.parameters()
    ]

    results = evaluate_one_epoch(
        model=model,
        dataloader=dataloader,
        loss_fn=loss_fn,
        device=device,
        class_names=CLASS_NAMES,
    )

    parameters_after = list(model.parameters())

    assert all(
        torch.equal(before, after)
        for before, after in zip(parameters_before, parameters_after)
    )

    assert model.training is False
    assert results["num_samples"] == 4


def test_training_respects_max_batches() -> None:
    model, dataloader, loss_fn, optimizer, device = create_test_components()

    results = train_one_epoch(
        model=model,
        dataloader=dataloader,
        loss_fn=loss_fn,
        optimizer=optimizer,
        device=device,
        class_names=CLASS_NAMES,
        max_batches=1,
    )

    assert results["num_batches"] == 1
    assert results["num_samples"] == 2


def test_training_rejects_invalid_threshold() -> None:
    model, dataloader, loss_fn, optimizer, device = create_test_components()

    with pytest.raises(ValueError, match="threshold"):
        train_one_epoch(
            model=model,
            dataloader=dataloader,
            loss_fn=loss_fn,
            optimizer=optimizer,
            device=device,
            class_names=CLASS_NAMES,
            threshold=1.0,
        )


def test_evaluation_rejects_empty_class_names() -> None:
    model, dataloader, loss_fn, _, device = create_test_components()

    with pytest.raises(ValueError, match="class_names"):
        evaluate_one_epoch(
            model=model,
            dataloader=dataloader,
            loss_fn=loss_fn,
            device=device,
            class_names=[],
        )
