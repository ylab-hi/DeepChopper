import torch
from torch import nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer


class TransformerEncoderModel(nn.Module):
    def __init__(
        self,
        vocab_size,
        d_model,
        nhead,
        num_encoder_layers,
        dim_feedforward,
        num_classes,
        max_seq_length,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.position_encoding = nn.Parameter(torch.zeros(1, max_seq_length, d_model))

        encoder_layer = TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward
        )

        self.transformer_encoder = TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, input_ids: torch.Tensor, input_quals: torch.Tensor):
        src = self.embedding(input_ids) + self.position_encoding[:, : input_ids.size(1), :]
        src = src.permute(1, 0, 2)  # Transformer expects [seq_len, batch_size, d_model]
        output = self.transformer_encoder(src)
        output = output.permute(1, 0, 2)  # Convert back to [batch_size, seq_len, d_model]
        logits = self.classifier(output)
        return logits
