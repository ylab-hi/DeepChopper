import torch
import torch.nn.functional as F  # noqa: N812
from torch import nn
from torch.nn import (
    TransformerDecoder,
    TransformerDecoderLayer,
    TransformerEncoder,
    TransformerEncoderLayer,
)


class TokenClassificationModule(nn.Module):
    def __init__(self, vocab_size, d_model, nhead, num_encoder_layers, dim_feedforward, number_of_classes):
        super().__init__()
        self.number_of_classes = number_of_classes

        self.qual_linear1 = nn.Sequential(
            nn.Linear(1, d_model),
        )

        self.embedding = nn.Embedding(vocab_size, d_model)
        encoder_layer = TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, batch_first=True
        )

        self.transformer_encoder = TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)

        self.classifier = nn.Linear(d_model, number_of_classes)
        self.d_model = d_model

    def forward(self, src: torch.Tensor, input_quals: torch.Tensor, src_mask=None):
        seq_length = src.size(1)

        # Calculate positional embeddings
        position = torch.arange(seq_length, dtype=torch.float, device=src.device).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, self.d_model, 2, device=src.device).float()
            * (-torch.log(torch.tensor(10000.0)) / self.d_model)
        )

        pos_embedding = torch.zeros((seq_length, self.d_model), device=src.device)
        pos_embedding[:, 0::2] = torch.sin(position * div_term)
        pos_embedding[:, 1::2] = torch.cos(position * div_term)

        src = F.relu(self.embedding(src) + self.qual_linear1(input_quals.unsqueeze(-1)))
        src = src + pos_embedding.unsqueeze(0)

        if src_mask is not None:
            src_mask = src_mask.to(dtype=torch.bool)

        output = self.transformer_encoder(src, src_key_padding_mask=src_mask)
        return self.classifier(output)


class Seq2SeqTokenClassifier(nn.Module):
    def __init__(
        self,
        vocab_size,
        d_model,
        nhead,
        num_encoder_layers,
        num_decoder_layers,
        dim_feedforward,
        number_of_classes,
    ):
        super().__init__()
        self.d_model = d_model

        self.embedding = nn.Embedding(vocab_size, d_model)
        self.qual_linear1 = nn.Linear(1, d_model)

        encoder_layer = TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, batch_first=True
        )
        self.transformer_encoder = TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)

        decoder_layer = TransformerDecoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, batch_first=True
        )
        self.transformer_decoder = TransformerDecoder(decoder_layer, num_layers=num_decoder_layers)

        self.classifier = nn.Linear(d_model, number_of_classes)

    def forward(
        self,
        src: torch.Tensor,
        input_quals: torch.Tensor,
        tgt: torch.Tensor,
        src_mask=None,
        tgt_mask=None,
    ):
        src_embeddings = self.embedding(src)
        qual_embeddings = self.qual_linear1(input_quals.unsqueeze(-1))

        encoder_input = src_embeddings + qual_embeddings

        if src_mask is not None:
            src_mask = src_mask.to(dtype=torch.bool)

        encoder_output = self.transformer_encoder(encoder_input, src_key_padding_mask=src_mask)

        tgt_embeddings = self.embedding(tgt)
        decoder_output = self.transformer_decoder(
            tgt_embeddings, encoder_output, memory_key_padding_mask=src_mask, tgt_mask=tgt_mask
        )

        return self.classifier(decoder_output)
