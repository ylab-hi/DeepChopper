import torch
from torch import nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer


class TokenClassificationModel(nn.Module):
    def __init__(
        self, vocab_size, d_model, nhead, num_encoder_layers, dim_feedforward, num_classes
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        encoder_layer = TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward
        )
        self.transformer_encoder = TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
        self.classifier = nn.Linear(d_model, num_classes)
        self.d_model = d_model

    def forward(self, src, src_mask=None):
        seq_length = src.size(1)
        position = torch.arange(seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, self.d_model, 2).float()
            * (-torch.log(torch.tensor(10000.0)) / self.d_model)
        )
        pos_embedding = torch.zeros((seq_length, 1, self.d_model))
        pos_embedding[:, 0, 0::2] = torch.sin(position * div_term)
        pos_embedding[:, 0, 1::2] = torch.cos(position * div_term)

        src = self.embedding(src) + pos_embedding.to(src.device)
        src = src.permute(1, 0, 2)  # Transformer expects [seq_len, batch_size, d_model]

        if src_mask is not None:
            src_mask = src_mask.to(dtype=torch.bool)

        output = self.transformer_encoder(src, src_key_padding_mask=src_mask)
        output = output.permute(1, 0, 2)  # Convert back to [batch_size, seq_len, d_model]
        return self.classifier(output)
