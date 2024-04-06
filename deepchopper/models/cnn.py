import torch
import torch.nn.functional as F  # noqa: N812
from torch import nn


class BenchmarkCNN(nn.Module):
    def __init__(self, number_of_classes, vocab_size, num_filters, filter_sizes, embedding_dim=100):
        """Genomics Benchmark CNN model.

        `embedding_dim` = 100 comes from:
        https://github.com/ML-Bioinfo-CEITEC/genomic_benchmarks/tree/main/experiments/torch_cnn_experiments
        """
        super().__init__()
        self.number_of_classes = number_of_classes
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()

        for fs in filter_sizes:
            self.convs.append(
                nn.Conv1d(
                    in_channels=embedding_dim,
                    out_channels=num_filters,
                    kernel_size=fs,
                    padding="same",
                )
            )
            self.batch_norms.append(nn.BatchNorm1d(num_filters))

            # Final convolutional layer to get the number_of_classes predictions per token
        self.final_conv = nn.Conv1d(
            in_channels=num_filters * len(filter_sizes),
            out_channels=number_of_classes,
            kernel_size=1,
        )  # this layer doesn't alter sequence length

        # use number of kernel same as the length of the sequence and average pooling
        # then flatten and use dense layers
        self.dense_model = nn.Sequential(
            nn.Linear(num_filters * len(filter_sizes), number_of_classes),
        )

    def forward(self, x):  # Adding `state` to be consistent with other models
        x = self.embeddings(x)
        x = x.transpose(1, 2)  # [batch_size, embedding_dim, input_len]

        x = [F.relu(bn(conv(x))) for conv, bn in zip(self.convs, self.batch_norms, strict=True)]
        x = torch.cat(x, dim=1) if len(x) > 1 else x[0]
        x = self.final_conv(x)
        return x.transpose(1, 2)  # [batch_size, input_len, num_filters]
