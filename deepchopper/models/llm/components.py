import torch
from torch import nn
from transformers import AutoModel, PretrainedConfig, PreTrainedModel
from transformers.modeling_outputs import TokenClassifierOutput

from .head import TokenClassificationHead

HyenadnaMaxLengths = {
    "hyenadna-tiny-1k-seqlen": 1024,
    "hyenadna-small-32k-seqlen": 32768,
    "hyenadna-medium-160k-seqlen": 160000,
    "hyenadna-medium-450k-seqlen": 450000,  # T4 up to here
    "hyenadna-large-1m-seqlen": 1_000_000,  # only A100 (paid tier)
}


class TokenClassificationConfig(PretrainedConfig):
    """Configuration class to store the model's hyperparameters."""

    model_type = "token-classification"

    def __init__(
        self,
        input_size: int = 256,
        lin1_size: int = 1024,
        lin2_size: int = 1024,
        num_class: int = 2,
        *,
        use_identity_layer_for_qual: bool = True,
        use_qual: bool = True,
        **kwargs,
    ):
        self.input_size = input_size
        self.lin1_size = lin1_size
        self.lin2_size = lin2_size
        self.num_class = num_class
        self.use_identity_layer_for_qual = use_identity_layer_for_qual
        self.use_qual = use_qual
        super().__init__(**kwargs)


class TokenClassification(PreTrainedModel):
    """Token classification model."""

    config_class = TokenClassificationConfig

    def __init__(
        self,
        config,
        hyenadna_model: str = "hyenadna-small-32k-seqlen",
        **kwargs,
    ):
        super().__init__(config, **kwargs)
        self.num_class = config.num_class
        self.hyenadna_model_name = hyenadna_model
        self.hyenadna = AutoModel.from_pretrained(f"LongSafari/{hyenadna_model}-hf", trust_remote_code=True)

        self.head = TokenClassificationHead(
            input_size=config.input_size,
            lin1_size=config.lin1_size,
            lin2_size=config.lin2_size,
            num_class=config.num_class,
            use_identity_layer_for_qual=config.use_identity_layer_for_qual,
            use_qual=config.use_qual,
        )

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor,
        input_quals: torch.Tensor,
        inputs_embeds: torch.FloatTensor | None = None,
        output_hidden_states: bool | None = None,
        return_dict: bool | None = None,
    ):
        transformer_outputs = self.hyenadna(
            input_ids,
            inputs_embeds=inputs_embeds,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        _batch_size = input_ids.shape[0]
        hidden_states = transformer_outputs[0]

        logits = self.head(hidden_states, input_quals)
        labels = labels.to(logits.device)
        loss_fct = nn.CrossEntropyLoss()

        loss = loss_fct(logits.view(-1, self.num_class), labels.view(-1))

        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=transformer_outputs.hidden_states,
        )
