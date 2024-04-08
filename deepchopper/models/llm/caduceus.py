import torch
from torch import nn
from transformers import AutoModel

BACKBONES = [
    "hyenadna-tiny-1k-seqlen",
    "hyenadna-small-32k-seqlen",
    "hyenadna-medium-160k-seqlen",
    "hyenadna-medium-450k-seqlen",
    "hyenadna-large-1m-seqlen",
    "caduceus-ph_seqlen-131k_d_model-256_n_layer-16",
    "caduceus-ps_seqlen-131k_d_model-256_n_layer-16",
]

# https://github.com/kuleshov-group/caduceus


class TokenClassificationModule(nn.Module):
    """Token classification model."""

    def __init__(
        self,
        number_of_classes: int,
        head: nn.Module,
        backbone_name: str = "caduceus-ph_seqlen-131k_d_model-256_n_layer-16",
    ):
        super().__init__()
        self.number_of_classes = number_of_classes
        self.backbone_name = backbone_name

        if "hyenadna" in backbone_name:
            model_name = f"LongSafari/{backbone_name}-hf"
        elif "caduceus" in backbone_name:
            model_name = f"kuleshov-group/{backbone_name}"
        else:
            msg = f"Unknown backbone model: {backbone_name}"
            raise ValueError(msg)

        self.backbone = AutoModel.from_pretrained(model_name, trust_remote_code=True)
        self.head = head

    def forward(
        self,
        input_ids: torch.Tensor,
        input_quals: torch.Tensor,
    ):
        transformer_outputs = self.backbone(
            input_ids,
            inputs_embeds=None,
            output_hidden_states=None,
            return_dict=None,
        )

        hidden_states = transformer_outputs[0]
        return self.head(hidden_states, input_quals)
