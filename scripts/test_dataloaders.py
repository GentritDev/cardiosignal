"""Smoke test for the reusable CardioSignal DataLoaders."""

import torch

from cardiosignal.data.loader import create_dataloaders


def main() -> None:
    print("\n========== CARDIOSIGNAL DATALOADER TEST ==========\n")

    dataloaders = create_dataloaders(
        batch_size=32,
        num_workers=0,
    )

    for split_name, dataloader in dataloaders.items():
        dataset = dataloader.dataset
        batch = next(iter(dataloader))

        signals = batch["signal"]
        targets = batch["labels"]
        record_ids = batch["record_id"]

        expected_signal_shape = (
            min(32, len(dataset)),
            12,
            5000,
        )
        expected_label_shape = (
            min(32, len(dataset)),
            5,
        )

        assert tuple(signals.shape) == expected_signal_shape
        assert tuple(targets.shape) == expected_label_shape
        assert torch.isfinite(signals).all()
        assert torch.isfinite(targets).all()
        assert len(record_ids) == signals.shape[0]

        print(f"{split_name.upper()}")
        print(f"  Dataset records: {len(dataset):,}")
        print(f"  Batch signals:   {tuple(signals.shape)}")
        print(f"  Batch labels:    {tuple(targets.shape)}")
        print(f"  Signal dtype:    {signals.dtype}")
        print(f"  Labels dtype:    {targets.dtype}")
        print(f"  First record:    {record_ids[0]}")
        print()

    print("ALL DATALOADERS PASSED VALIDATION.")


if __name__ == "__main__":
    main()