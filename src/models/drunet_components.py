from typing import Literal, Optional

import torch
import torch.nn.functional as F
from torch import nn

class LabelEmbedder(nn.Module):
    """
    Embeds class labels into vector representations. Also handles label dropout for classifier-free guidance.
    """

    def __init__(self, num_classes, hidden_size, dropout_prob):
        super().__init__()
        use_cfg_embedding = dropout_prob > 0
        self.embedding_table = nn.Embedding(num_classes + use_cfg_embedding, hidden_size)
        self.num_classes = num_classes
        self.dropout_prob = dropout_prob

    def token_drop(self, labels, force_drop_ids=None):
        """
        Drops labels to enable classifier-free guidance.
        """
        if force_drop_ids is None:
            drop_ids = torch.rand(labels.shape[0], device=labels.device) < self.dropout_prob
        else:
            drop_ids = force_drop_ids == 1
        labels = torch.where(drop_ids, self.num_classes, labels)
        return labels

    def forward(self, labels, train, force_drop_ids=None):
        use_dropout = self.dropout_prob > 0
        if (train and use_dropout) or (force_drop_ids is not None):
            labels = self.token_drop(labels, force_drop_ids)
        embeddings = self.embedding_table(labels)
        return embeddings


class CondSequential(torch.nn.Module):
    def __init__(self, *args):
        super().__init__()
        self.m = torch.nn.ModuleList(args)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        for m in self.m:
            x = m(x, y)
        return x


class ResBlock(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        norm_type: Optional[Literal["instance", "group"]] = "instance",
        kernel_size: int = 3,
        padding: int = 1,
        bias: bool = False,
        num_cond_features: int = 0,
        dim: int = 3,
    ):
        super().__init__()

        conv = torch.nn.Conv3d if dim == 3 else torch.nn.Conv2d

        if norm_type == "instance":
            Normalization = lambda num_channels: (
                nn.InstanceNorm3d(num_channels, affine=True)
                if dim == 3
                else nn.InstanceNorm2d(num_channels, affine=True)
            )
        elif norm_type == "group":
            Normalization = lambda num_channels: nn.GroupNorm(32, num_channels)
        elif norm_type is None:
            Normalization = lambda num_channels: nn.Identity()
        else:
            raise ValueError(f"Unsupported norm_type: {norm_type}")

        self.c1 = conv(
            in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, padding=padding, bias=bias
        )
        self.norm1 = Normalization(out_channels) if norm_type is not None else None

        if num_cond_features:
            self.affine = ConditionalAffine(
                num_features=out_channels, num_cond_features=num_cond_features, bias=bias, dim=dim
            )
        else:
            self.affine = Identity()
        self.act = torch.nn.SiLU()
        self.c2 = conv(
            in_channels=out_channels, out_channels=in_channels, kernel_size=kernel_size, padding=padding, bias=bias
        )
        self.norm2 = Normalization(in_channels) if norm_type is not None else None

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        h = self.c1(x)

        if self.norm1 is not None:
            h = self.norm1(h)

        h = self.affine(h, y)
        h = self.act(h)
        h = self.c2(h)

        if self.norm2 is not None:
            h = self.norm2(h)

        return x + h


class DownBlock(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        mode: str = "conv",
        kernel_size: int = 3,
        stride: int = 2,
        bias: bool = True,
        dim: int = 3,
    ):
        super().__init__()

        if mode == "conv":
            conv = torch.nn.Conv3d if dim == 3 else torch.nn.Conv2d
            self.m = conv(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=kernel_size // 2,
                bias=bias,
            )
        elif mode == "avgpool":
            pool = torch.nn.AvgPool3d if dim == 3 else torch.nn.AvgPool2d
            self.m = pool(
                kernel_size=kernel_size,
                stride=stride,
                padding=kernel_size // 2,
            )
        elif mode == "maxpool":
            pool = torch.nn.MaxPool3d if dim == 3 else torch.nn.MaxPool2d
            self.m = pool(
                kernel_size=kernel_size,
                stride=stride,
                padding=kernel_size // 2,
            )
        else:
            raise RuntimeError("unkown mode")

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        return self.m(x)


class UpBlock(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        mode: str = "conv",
        kernel_size: int = 3,
        stride: int = 2,
        bias: bool = True,
        dim: int = 3,
    ):
        super().__init__()

        if mode == "conv":
            conv = torch.nn.ConvTranspose3d if dim == 3 else torch.nn.ConvTranspose2d
            self.m = conv(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=kernel_size // 2,
                bias=bias,
            )
        elif mode == "interpolate":
            pass
            # TODO!
        else:
            raise RuntimeError("unkown mode")

    def forward(self, x: torch.Tensor, shape: tuple) -> torch.Tensor:
        return self.m(x, output_size=shape)


class Identity(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return x


class ConditionalAffine(torch.nn.Module):
    def __init__(self, num_features: int, num_cond_features: int, bias: bool = False, dim: int = 3):
        super().__init__()

        self.num_features = num_features
        self.bias = bias
        self.dim = dim
        self.linear = torch.nn.Linear(in_features=num_cond_features, out_features=(2 if bias else 1) * num_features)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        if self.bias:
            gamma, beta = self.linear(y).chunk(2, dim=1)
        else:
            gamma = self.linear(y)
            beta = torch.zeros_like(gamma)

        gamma = F.softplus(gamma)

        if self.dim == 3:
            return gamma.view(-1, self.num_features, 1, 1, 1) * x + beta.view(-1, self.num_features, 1, 1, 1)
        else:
            return gamma.view(-1, self.num_features, 1, 1) * x + beta.view(-1, self.num_features, 1, 1)