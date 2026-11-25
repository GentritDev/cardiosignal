
import pytest
import torch

from cardiosignal.models.cnn1d import ConvBlock1D, ECG1DCNN


def test_model_output_shape() -> None:
    model = ECG1DCNN()
    inputs = torch.randn(4, 12, 5000)

    outputs = model(inputs)

    assert outputs.shape == (4, 5)


def test_model_output_is_finite() -> None:
    model = ECG1DCNN()
    inputs = torch.randn(2, 12, 5000)

    outputs = model(inputs)

    assert torch.isfinite(outputs).all()


def test_model_supports_different_batch_sizes() -> None:
    model = ECG1DCNN()

    for batch_size in (1, 2, 8):
        inputs = torch.randn(batch_size, 12, 5000)
        outputs = model(inputs)

        assert outputs.shape == (batch_size, 5)


def test_model_supports_custom_channels() -> None:
    model = ECG1DCNN(
        in_channels=12,
        num_classes=5,
        channels=(16, 32, 64),
    )

    outputs = model(torch.randn(2, 12, 5000))

    assert outputs.shape == (2, 5)


def test_model_rejects_wrong_input_dimensions() -> None:
    model = ECG1DCNN()

    with pytest.raises(ValueError, match="3D input"):
        model(torch.randn(2, 12, 5000, 1))


def test_model_rejects_wrong_number_of_channels() -> None:
    model = ECG1DCNN()

    with pytest.raises(ValueError, match="input channels"):
        model(torch.randn(2, 10, 5000))


def test_model_supports_backward_pass() -> None:
    model = ECG1DCNN()
    inputs = torch.randn(2, 12, 5000)
    targets = torch.randint(0, 2, (2, 5)).float()

    criterion = torch.nn.BCEWithLogitsLoss()
    outputs = model(inputs)
    loss = criterion(outputs, targets)

    loss.backward()

    assert loss.item() > 0
    assert model.classifier[-1].weight.grad is not None


def test_conv_block_rejects_even_kernel_size() -> None:
    with pytest.raises(ValueError, match="odd"):
        ConvBlock1D(12, 32, kernel_size=4)


from cardiosignal.models.resnet1d import ECGResNet1D, ResidualBlock1D


def test_resnet_output_shape() -> None:
    model = ECGResNet1D()
    outputs = model(torch.randn(2, 12, 5000))

    assert outputs.shape == (2, 5)


def test_resnet_output_is_finite() -> None:
    model = ECGResNet1D()
    outputs = model(torch.randn(2, 12, 5000))

    assert torch.isfinite(outputs).all()


def test_resnet_backward_pass() -> None:
    model = ECGResNet1D()
    inputs = torch.randn(2, 12, 5000)
    targets = torch.randint(0, 2, (2, 5)).float()

    loss = torch.nn.BCEWithLogitsLoss()(model(inputs), targets)
    loss.backward()

    assert loss.item() > 0
    assert model.classifier[-1].weight.grad is not None


def test_resnet_rejects_wrong_input_channels() -> None:
    model = ECGResNet1D()

    with pytest.raises(ValueError, match="input channels"):
        model(torch.randn(2, 10, 5000))


def test_residual_block_preserves_shape() -> None:
    block = ResidualBlock1D(32, 32)
    outputs = block(torch.randn(2, 32, 100))

    assert outputs.shape == (2, 32, 100)


def test_residual_block_projection_changes_channels() -> None:
    block = ResidualBlock1D(32, 64, stride=2)
    outputs = block(torch.randn(2, 32, 100))

    assert outputs.shape == (2, 64, 50)
