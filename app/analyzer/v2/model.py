import math
import torch
import torch.nn as nn
from torchvision import models
from torch import Tensor

from analyzer.utils import to_best_device


class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


class ViewFeatures(nn.Module):
    def __init__(self):
        super().__init__()

    def __repr__(self):
        return f'ViewFeatures'

    def forward(self, x):
        return x.view(x.shape[0], -1)


class ViewOfShape(nn.Module):
    def __init__(self, shape: torch.Size):
        super().__init__()
        self.shape = shape

    def __repr__(self):
        return f'View {self.shape}'

    def forward(self, x):
        # print(f"ViewOfShape before is {x.shape}")
        return x.view(self.shape)


class View(nn.Module):
    def __init__(self):
        super().__init__()

    def __repr__(self):
        return f'View'

    def forward(self, x):
        # print(f"View before is {x.shape}")
        return x.view(x.shape[0], -1, x.shape[3])


class Permute(nn.Module):
    def __init__(self, *permutations):
        super().__init__()
        self.permutations = permutations

    def __repr__(self):
        return f'Permute {self.permutations}'

    def forward(self, x):
        # print(x.shape)
        # print(f"{self.permutations}")
        return x.permute(*self.permutations)


class PostAttention(nn.Module):
    def __init__(self, seq_len: int):
        super().__init__()
        self.selector = to_best_device(torch.tensor([0, 1, 2, 3]))

        final_mask = torch.full([seq_len], 0.)
        final_mask = to_best_device(final_mask)
        final_mask.requires_grad_(False)
        final_mask[:4] = 1.
        self.final_mask = final_mask

    def __repr__(self):
        return f'Post Attention'

    def forward(self, x):
        x = x.squeeze(dim=2)
        x = torch.index_select(x, 1, self.selector)
        return x


class AttentionSubsBoxerModel(nn.Module):

    def __init__(self, num_encoder_layers: int = 3, dropout: int = 0.25):
        super(AttentionSubsBoxerModel, self).__init__()
        seq_len = 16 * 16
        d_model = 512  # Depends on backbone model
        resnet = models.resnet34(weights='ResNet34_Weights.DEFAULT')
        for param in resnet.parameters():
            param.requires_grad_(True)
        layers = list(resnet.children())[:8]
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model,
                                                   nhead=8,
                                                   batch_first=True)
        features_extractor: nn.Module = nn.Sequential(
            nn.Sequential(*layers),
            # nn.AdaptiveAvgPool2d((1, 1)),
            # ViewFeatures(),
            # nn.Linear(512, 1024),
            # nn.Sigmoid()
        )
        pre_attention = nn.Sequential(
            Permute(0, 2, 3, 1),
            View(),
            PositionalEncoding(d_model, dropout, max_len=seq_len)
        )
        attention = nn.Sequential(
            nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers),
            ViewOfShape(torch.Size((-1, seq_len * d_model))),
            nn.Linear(seq_len * d_model, 5)
        )
        self.model = nn.Sequential(
            features_extractor,
            pre_attention,
            attention
        )

    def forward(self, x):
        # print(f"In the beginning x.shape = {x.shape}")
        return self.model(x)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=1.0)
            if module.bias is not None:
                module.bias.data.zero_()

    def initialize_weights(self):
        # self.final_layer.apply(self._init_weights)
        pass