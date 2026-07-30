from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as functional
from torch.nn import init


class Linear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        bias: bool = True,
        activation: str = "relu",
    ) -> None:
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        if activation == "relu":
            self.activation: nn.Module = nn.ReLU(inplace=True)
        elif activation == "identity":
            self.activation = nn.Identity()
        else:
            raise ValueError(f"Unsupported activation: {activation}")

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.activation(self.linear(inputs))


class Conv3DConfigurable(nn.Module):
    def __init__(
        self,
        in_channels: int,
        filters: int,
        dilation_rate: int,
        *,
        mid_filter: int | None = None,
        sharable: bool = False,
        use_bias: bool = False,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList()
        temporal_channels = 2 * filters if mid_filter is None else mid_filter

        if not sharable:
            spatial = nn.Conv3d(
                in_channels=in_channels,
                out_channels=temporal_channels,
                kernel_size=(1, 3, 3),
                padding=(0, 1, 1),
                dilation=(1, 1, 1),
                bias=False,
            )
            init.kaiming_normal_(spatial.weight, mode="fan_in", nonlinearity="relu")
            self.layers.append(spatial)

        temporal = nn.Conv3d(
            in_channels=temporal_channels,
            out_channels=filters,
            kernel_size=(3, 1, 1),
            padding=(dilation_rate, 0, 0),
            dilation=(dilation_rate, 1, 1),
            bias=use_bias,
        )
        init.kaiming_normal_(temporal.weight, mode="fan_in", nonlinearity="relu")
        self.layers.append(temporal)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        output = inputs
        for layer in self.layers:
            output = layer(output)
        return output


class DilatedDCNNV2(nn.Module):
    def __init__(
        self,
        in_channels: int,
        filters: int,
        *,
        multiplier: int = 2,
        n_dilation: int = 4,
    ) -> None:
        super().__init__()
        self.conv_blocks = nn.ModuleList()
        filters_per_block = (filters * 4) // n_dilation

        for dilation in range(n_dilation):
            output_filters = (
                filters_per_block
                if dilation < n_dilation - 1
                else (filters * 4) - filters_per_block * (n_dilation - 1)
            )
            self.conv_blocks.append(
                Conv3DConfigurable(
                    in_channels,
                    output_filters,
                    2**dilation,
                    mid_filter=multiplier * filters,
                    use_bias=False,
                )
            )

        self.batch_norm = nn.BatchNorm3d(
            num_features=filters * 4,
            eps=1e-3,
            momentum=0.1,
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        output = torch.cat([block(inputs) for block in self.conv_blocks], dim=1)
        return functional.relu(self.batch_norm(output))


class DilatedDCNNV2ABC(nn.Module):
    def __init__(
        self,
        in_channels: int,
        filters: int,
        *,
        multiplier: int = 4,
        n_dilation: int = 4,
    ) -> None:
        super().__init__()
        shared_channels = multiplier * filters
        self.share = nn.Conv3d(
            in_channels=in_channels,
            out_channels=shared_channels,
            kernel_size=(1, 3, 3),
            padding=(0, 1, 1),
            dilation=(1, 1, 1),
            bias=False,
        )
        init.kaiming_normal_(self.share.weight, mode="fan_in", nonlinearity="relu")

        self.conv_blocks = nn.ModuleList()
        filters_per_block = (filters * 4) // n_dilation
        for dilation in range(n_dilation):
            output_filters = (
                filters_per_block
                if dilation < n_dilation - 1
                else (filters * 4) - filters_per_block * (n_dilation - 1)
            )
            self.conv_blocks.append(
                Conv3DConfigurable(
                    shared_channels,
                    output_filters,
                    2**dilation,
                    mid_filter=shared_channels,
                    sharable=True,
                    use_bias=False,
                )
            )

        self.batch_norm = nn.BatchNorm3d(
            num_features=filters * 4,
            eps=1e-3,
            momentum=0.1,
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        shared = self.share(inputs)
        output = torch.cat([block(shared) for block in self.conv_blocks], dim=1)
        return functional.relu(self.batch_norm(output))


def _centered_similarity_windows(
    similarities: torch.Tensor, width: int
) -> torch.Tensor:
    if width % 2 != 1:
        raise ValueError("Similarity window width must be odd")
    batch_size, frame_count, other_count = similarities.shape
    if frame_count != other_count:
        raise ValueError("Similarity tensor must be square in its last dimensions")

    radius = width // 2
    padded = functional.pad(similarities, (radius, radius))
    offsets = torch.arange(width, device=similarities.device)
    starts = torch.arange(frame_count, device=similarities.device).unsqueeze(1)
    indices = starts + offsets.unsqueeze(0)
    indices = indices.unsqueeze(0).expand(batch_size, -1, -1)
    return torch.gather(padded, 2, indices)


class FrameSimilarity(nn.Module):
    def __init__(
        self,
        *,
        in_channels: int = 448,
        similarity_dim: int = 128,
        lookup_window: int = 101,
        output_dim: int = 128,
    ) -> None:
        super().__init__()
        self.projection = Linear(
            in_channels,
            similarity_dim,
            activation="identity",
        )
        self.fc = Linear(lookup_window, output_dim, activation="relu")
        self.lookup_window = lookup_window

    def forward(self, inputs: list[torch.Tensor]) -> torch.Tensor:
        output = torch.cat(
            [torch.mean(block, dim=(3, 4)) for block in inputs],
            dim=1,
        )
        output = output.permute(0, 2, 1)
        batch_size, frame_count, channel_count = output.shape
        output = self.projection(
            output.reshape(batch_size * frame_count, channel_count)
        )
        output = functional.normalize(output, p=2, dim=1)
        output = output.reshape(batch_size, frame_count, -1)
        similarities = torch.matmul(output, output.permute(0, 2, 1))
        return self.fc(_centered_similarity_windows(similarities, self.lookup_window))


class ColorHistograms(nn.Module):
    def __init__(self, *, lookup_window: int = 101, output_dim: int = 128) -> None:
        super().__init__()
        self.fc = Linear(lookup_window, output_dim, activation="relu")
        self.lookup_window = lookup_window

    @staticmethod
    def _histograms(inputs: torch.Tensor) -> torch.Tensor:
        frames = inputs.to(dtype=torch.int64).permute(0, 2, 3, 4, 1)
        batch_size, frame_count, height, width, channels = frames.shape
        if channels != 3:
            raise ValueError("AutoShot color histograms expect RGB input")

        red = frames[..., 0] >> 5
        green = frames[..., 1] >> 5
        blue = frames[..., 2] >> 5
        bins = ((red << 6) + (green << 3) + blue).reshape(
            batch_size * frame_count,
            height * width,
        )

        histograms = torch.zeros(
            batch_size * frame_count,
            512,
            dtype=torch.float32,
            device=inputs.device,
        )
        histograms.scatter_add_(
            1,
            bins,
            torch.ones_like(bins, dtype=torch.float32),
        )
        histograms = functional.normalize(histograms, p=2, dim=1)
        return histograms.reshape(batch_size, frame_count, 512)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        histograms = self._histograms(inputs)
        similarities = torch.matmul(histograms, histograms.permute(0, 2, 1))
        return self.fc(_centered_similarity_windows(similarities, self.lookup_window))


class AutoShotSupernet(nn.Module):
    """AutoShot architecture compatible with the supplied ``checkpoint['net']``.

    The class is vendored in the worker package so production inference does not
    need to download Python source code when the container starts.
    """

    def __init__(self, feature_width: int = 1024) -> None:
        super().__init__()
        self.Layer_0_3 = DilatedDCNNV2(3, 16, multiplier=1)
        self.Layer_1_8 = DilatedDCNNV2ABC(
            16 * 4,
            16,
            multiplier=4,
            n_dilation=5,
        )
        self.Layer_2_8 = DilatedDCNNV2ABC(
            16 * 4,
            32,
            multiplier=4,
            n_dilation=5,
        )
        self.Layer_3_8 = DilatedDCNNV2ABC(
            32 * 4,
            32,
            multiplier=4,
            n_dilation=5,
        )
        self.Layer_4_13 = DilatedDCNNV2(
            32 * 4,
            64,
            multiplier=3,
            n_dilation=5,
        )
        self.Layer_5_12 = DilatedDCNNV2(
            64 * 4,
            64,
            multiplier=2,
            n_dilation=5,
        )
        self.pool = nn.AvgPool3d(kernel_size=(1, 2, 2))

        self.fc1_0 = Linear(4864, feature_width, activation="relu")
        self.fc1 = Linear(5888, feature_width, activation="relu")
        self.cls_layer1 = Linear(feature_width, 1, activation="identity")
        self.cls_layer2 = Linear(feature_width, 1, activation="identity")
        self.frame_sim_layer = FrameSimilarity()
        self.color_hist_layer = ColorHistograms()
        self.dropout = nn.Dropout(p=0.5)
        self._initialize_weights()

    def _initialize_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.BatchNorm3d):
                module.weight.data.fill_(1)
                module.bias.data.zero_()
            elif isinstance(module, nn.Linear):
                fan_out = module.weight.size(0)
                fan_in = module.weight.size(1)
                limit = math.sqrt(6.0 / (fan_in + fan_out))
                module.weight.data.uniform_(-limit, limit)
                if module.bias is not None:
                    module.bias.data.zero_()
            elif isinstance(module, nn.Conv3d):
                init.kaiming_normal_(module.weight, mode="fan_in", nonlinearity="relu")

    def forward(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        output = inputs / 255.0
        block_features: list[torch.Tensor] = []
        shortcut: torch.Tensor | None = None

        layers = (
            self.Layer_0_3,
            self.Layer_1_8,
            self.Layer_2_8,
            self.Layer_3_8,
            self.Layer_4_13,
            self.Layer_5_12,
        )
        for index, layer in enumerate(layers):
            output = layer(output)
            if index in (0, 2, 4):
                shortcut = output
                continue
            if shortcut is None:
                raise RuntimeError("AutoShot residual shortcut was not initialized")
            output = self.pool(shortcut + output)
            block_features.append(output)

        output = output.permute(0, 2, 3, 4, 1)
        output = output.reshape(output.shape[0], output.shape[1], -1)
        output = torch.cat(
            [
                self.frame_sim_layer(block_features),
                self.color_hist_layer(inputs),
                output,
            ],
            dim=2,
        )
        output = self.fc1_0(output)
        output = self.dropout(output)
        return self.cls_layer1(output), self.cls_layer2(output)
