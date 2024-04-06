import torch
from torch import nn
from transformers import AutoModel


class TokenClassificationModule(nn.Module):
    """Token classification model."""

    def __init__(
        self,
        number_of_classes: int,
        head: nn.Module,
        hyenadna_model: str = "hyenadna-small-32k-seqlen",
    ):
        super().__init__()
        self.number_of_classes = number_of_classes
        self.hyenadna_model_name = hyenadna_model
        self.hyenadna = AutoModel.from_pretrained(
            f"LongSafari/{hyenadna_model}-hf", trust_remote_code=True
        )
        self.head = head

    def forward(
        self,
        input_ids: torch.Tensor,
        input_quals: torch.Tensor,
    ):
        transformer_outputs = self.hyenadna(
            input_ids,
            inputs_embeds=None,
            output_hidden_states=None,
            return_dict=None,
        )
        hidden_states = transformer_outputs[0]
        return self.head(hidden_states, input_quals)
