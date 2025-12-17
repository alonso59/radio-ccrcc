from typing import Any, Dict, List, Literal, Optional

import numpy as np
import torch
from torch import nn
import drunet_components as B
from torchinfo import summary


class GaussianFourierProjection(nn.Module):
    """Gaussian random features for encoding time steps."""
    def __init__(self, embed_dim, scale=30.0):
        super().__init__()
        self.W = nn.Parameter(torch.randn(embed_dim // 2) * scale, requires_grad=False)

    def forward(self, x):
        x_proj = x[:, None] * self.W[None, :] * 2 * np.pi
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)


class FeedForward(torch.nn.Module):
    def __init__(self, in_features: int, hidden_features: int, out_features: int) -> None:
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(in_features=in_features, out_features=hidden_features),
            torch.nn.SiLU(),
            torch.nn.Linear(in_features=hidden_features, out_features=out_features),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DRUNet(torch.nn.Module):
    def __init__(
        self,
        dim: int,
        nc: List[int],
        nb: int,
        bias: bool,
        num_cond_features: int,
        in_channels: int,
        out_channels: int,
        label_dim: int,  # Number of class labels, 0 = unconditional
        label_dropout: float,  # Dropout probability for classifier-free guidance
        embedding: Dict[str, Any],
        norm_type: Optional[Literal["instance", "group"]] = "instance",
    ):
        """
        nc: list of channel widths for each scale, length >= 2.
            Example: [16,32,64,128]  (3 encoder/decoder scales + 1 body at 128)
                     [32,64,128]    (2 encoder/decoder scales + 1 body at 128)
        """
        super().__init__()

        assert dim in {2, 3}, "dim must be either 2 or 3"
        assert len(nc) >= 2, "nc must contain at least two levels (encoder scale(s) + body)"
        self.dim = dim
        self.nc = nc
        self.nb = nb
        self.num_scales = len(nc)  # number of levels including body
        self.cond_channels = num_cond_features

        # Embedders
        self.t_embedder = FeedForward(**embedding)
        self.y_embedder = B.LabelEmbedder(label_dim, num_cond_features, label_dropout) if label_dim else None

        conv = torch.nn.Conv3d if dim == 3 else torch.nn.Conv2d

        # Head conv: in -> nc[0]
        self.m_head = conv(in_channels=in_channels, out_channels=nc[0], kernel_size=3, padding=1, bias=bias)

        # Build encoders (one per scale except the deepest body). Encoder count = len(nc)-1
        encoders = []
        for ch in nc[:-1]:
            encoders.append(
                B.CondSequential(
                    *[
                        B.ResBlock(ch, ch, norm_type, bias=bias, num_cond_features=num_cond_features, dim=dim)
                        for _ in range(nb)
                    ]
                )
            )
        self.m_encoders = torch.nn.ModuleList(encoders)

        # Build down blocks between encoder levels (maps nc[i] -> nc[i+1])
        self.m_down = torch.nn.ModuleList(
            [B.DownBlock(nc[i], nc[i + 1], bias=bias, mode="conv", dim=dim) for i in range(len(nc) - 1)]
        )

        # Body at deepest nc[-1]
        self.m_body = B.CondSequential(
            *[
                B.ResBlock(nc[-1], nc[-1], norm_type, bias=bias, num_cond_features=num_cond_features, dim=dim)
                for _ in range(nb)
            ]
        )

        # Build up blocks (reverse of down blocks)
        self.m_up = torch.nn.ModuleList(
            [B.UpBlock(nc[i + 1], nc[i], bias=bias, mode="conv", dim=dim) for i in range(len(nc) - 1)]
        )

        # Build decoders — one per encoder level (in reverse order they are applied)
        decoders = []
        for ch in nc[:-1]:
            decoders.append(
                B.CondSequential(
                    *[
                        B.ResBlock(ch, ch, norm_type, bias=bias, num_cond_features=num_cond_features, dim=dim)
                        for _ in range(nb)
                    ]
                )
            )
        # Note: decoder order in the list corresponds to encoder order.
        self.m_decoders = torch.nn.ModuleList(decoders)

        # Tail conv: nc[0] -> out_channels
        self.m_tail = conv(in_channels=nc[0], out_channels=out_channels, kernel_size=3, padding=1, bias=bias)

    def forward(self, x: torch.Tensor, t: torch.Tensor, y: Optional[torch.Tensor] = None) -> torch.Tensor:
        emb = self.t_embedder(t.view(x.shape[0], -1))  # (B, D)

        if self.y_embedder is not None and y is not None:
            y = self.y_embedder(y.flatten(), self.training)  # (B, D)
            emb = emb + y

        # Head
        x_head = self.m_head(x)

        # Encoder path
        encs: List[torch.Tensor] = []         # outputs of each encoder (before down)
        down_outs: List[torch.Tensor] = []    # outputs of each down (inputs to next encoder / body)
        x_cur = x_head
        n_enc = len(self.m_encoders)  # = len(nc) - 1

        for i in range(n_enc):
            x_enc = self.m_encoders[i](x_cur, emb)  # encode at level i
            encs.append(x_enc)
            # apply down to get input for next level (or body)
            x_down = self.m_down[i](x_enc)
            down_outs.append(x_down)
            x_cur = x_down

        # Body (deepest)
        x = self.m_body(x_cur, emb)

        # Decoder / up path
        # iterate from deepest encoder index down to 0
        for i in reversed(range(n_enc)):
            # up block expects (input_tensor, target_shape_of_skip)
            # add residual from down_outs[i] (the input to body for the first iteration)
            x = self.m_up[i](x + down_outs[i], encs[i].shape)
            x = self.m_decoders[i](x, emb)

        # Final tail: add skip from head-level encoder (encs[0]) like original pattern
        x = self.m_tail(x + encs[0])

        return x


if __name__ == "__main__":
    # Example usage with 4-level net:
    model = DRUNet(
        dim=3,
        nc=[16, 32, 64, 128],
        nb=2,
        bias=True,
        num_cond_features=128,
        in_channels=1,
        out_channels=1,
        label_dim=0,
        label_dropout=0.0,
        embedding={
            "in_features": 1,
            "hidden_features": 256,
            "out_features": 128,
        },
        norm_type="instance",
    )

    x = torch.randn(2, 1, 128, 128, 128)
    t = torch.randn(2, 1)
    out = model(x, t)
    print("output shape:", out.shape)  # Expect (2, 1, 128, 128, 128)

    # Example usage with 3-level net (fewer scales):
    model_small = DRUNet(
        dim=3,
        nc=[32, 64, 128],
        nb=2,
        bias=True,
        num_cond_features=128,
        in_channels=1,
        out_channels=1,
        label_dim=0,
        label_dropout=0.0,
        embedding={
            "in_features": 1,
            "hidden_features": 256,
            "out_features": 128,
        },
        norm_type="instance",
    )
    x2 = torch.randn(2, 1, 128, 128, 128)
    out2 = model_small(x2, t)
    print("output shape small:", out2.shape)

    summary(model, input_data=(x, t), col_names=["input_size", "output_size", "num_params", "trainable"], depth=4)
