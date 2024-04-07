import torch
import torch.nn.functional as F  # noqa: N812
from torch import nn


class BenchmarkCNN(nn.Module):
    """BenchmarkCNN."""

    def __init__(self, number_of_classes, vocab_size, num_filters, filter_sizes, embedding_dim=100):
        """Genomics Benchmark CNN model.

        `embedding_dim` = 100 comes from:
        https://github.com/ML-Bioinfo-CEITEC/genomic_benchmarks/tree/main/experiments/torch_cnn_experiments
        """
        super().__init__()
        self.number_of_classes = number_of_classes
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)

        self.qual_linear1 = nn.Sequential(
            nn.Linear(1, embedding_dim),
        )

        layers = []
        in_channels = embedding_dim
        for idx, fs in enumerate(filter_sizes):
            layers.append(
                nn.Conv1d(
                    in_channels=in_channels,
                    out_channels=num_filters[idx],
                    kernel_size=fs,
                    padding="same",
                )
            )
            layers.append(nn.BatchNorm1d(num_filters[idx]))
            layers.append(nn.ReLU())
            in_channels = num_filters[idx]

        self.model = nn.Sequential(*layers)

        # use number of kernel same as the length of the sequence and average pooling
        # then flatten and use dense layers
        self.dense_model = nn.Sequential(
            nn.Linear(in_channels, number_of_classes),
        )

    def forward(
        self, input_ids: torch.Tensor, input_quals: torch.Tensor
    ):  # Adding `state` to be consistent with other models
        x = self.embeddings(input_ids)
        x = F.relu(x + self.qual_linear1(input_quals.unsqueeze(-1)))
        x = x.transpose(1, 2)  # [batch_size, embedding_dim, input_len]
        x = self.model(x)
        x = x.transpose(1, 2)
        return self.dense_model(x)  # [batch_size, input_len, num_filters]
