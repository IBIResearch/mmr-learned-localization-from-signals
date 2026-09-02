import torch.nn as nn

from . import model_factory


@model_factory.add_model("deep-conv-transformer")
class DeepConvTransformer(nn.Module):
    def __init__(
        self, n_coils=16, coord_dim=3, embedding_dim=128, num_heads=8, kernel_size=5
    ):
        super().__init__()

        self.conv1 = nn.Conv1d(n_coils, embedding_dim // 2, kernel_size=kernel_size)
        self.bn1 = nn.BatchNorm1d(embedding_dim // 2)
        self.conv2 = nn.Conv1d(
            embedding_dim // 2, embedding_dim, kernel_size=kernel_size
        )
        self.bn2 = nn.BatchNorm1d(embedding_dim)
        self.mha = nn.MultiheadAttention(
            embedding_dim, num_heads=num_heads, batch_first=True
        )
        self.bn3 = nn.BatchNorm1d(embedding_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim * 4),
            nn.GELU(),
            nn.Linear(embedding_dim * 4, embedding_dim),
        )
        self.bn4 = nn.BatchNorm1d(embedding_dim)
        self.output_layer = nn.Linear(embedding_dim, coord_dim)
        self.activation = nn.GELU()

    def forward(self, inputs):
        x = inputs.transpose(1, 2)
        x = self.activation(self.bn1(self.conv1(x)))
        x = self.activation(self.bn2(self.conv2(x)))
        x = x.transpose(1, 2)
        out, _ = self.mha(x, x, x)
        x = x + out
        x = self.bn3(x.transpose(1, 2)).transpose(1, 2)
        out = self.ffn(x)
        x = x + out
        x = self.bn4(x.transpose(1, 2)).transpose(1, 2)
        x = x.mean(dim=1)
        return self.output_layer(x)
