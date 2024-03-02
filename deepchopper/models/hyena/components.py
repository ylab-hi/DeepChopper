import torch
from torch import nn
from transformers import AutoModel, PretrainedConfig, PreTrainedModel
from transformers.modeling_outputs import TokenClassifierOutput
from transformers.utils import logging

logging.set_verbosity_info()
logger = logging.get_logger("transformers")


class TokenClassificationHead(nn.Module):
    def __init__(
        self,
        input_size: int,
        lin1_size: int,
        lin2_size: int,
        num_class: int,
        *,
        use_identity_layer_for_qual: bool,
    ):
        super().__init__()
        self.activation = nn.ReLU()
        self.linear1 = nn.Linear(input_size, lin1_size)
        self.linear2 = nn.Linear(lin1_size, lin2_size)
        self.linear3 = nn.Linear(lin2_size, num_class)
        self.qual_linear1 = (
            nn.Identity() if use_identity_layer_for_qual else nn.Linear(1, lin1_size)
        )

    def forward(self, x: torch.Tensor, input_quals: torch.Tensor) -> torch.Tensor:
        output = self.activation(self.linear1(x))
        residual = output + self.qual_linear1(input_quals.unsqueeze(-1))  # may add activation
        output = self.activation(self.linear2(residual) + residual)
        return self.linear3(output)


class TokenClassificationConfig(PretrainedConfig):
    model_type = "token-classification"

    def __init__(
        self,
        input_size: int = 256,
        lin1_size: int = 1024,
        lin2_size: int = 1024,
        num_class: int = 2,
        *,
        use_identity_layer_for_qual: bool = True,
        **kwargs,
    ):
        self.input_size = input_size
        self.lin1_size = lin1_size
        self.lin2_size = lin2_size
        self.num_class = num_class
        self.use_identity_layer_for_qual = use_identity_layer_for_qual
        super().__init__(**kwargs)


class TokenClassification(PreTrainedModel):
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
        self.hyenadna = AutoModel.from_pretrained(
            f"LongSafari/{hyenadna_model}-hf", trust_remote_code=True
        )

        self.head = TokenClassificationHead(
            input_size=config.input_size,
            lin1_size=config.lin1_size,
            lin2_size=config.lin2_size,
            num_class=config.num_class,
            use_identity_layer_for_qual=config.use_identity_layer_for_qual,
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
    ) -> torch.Tensor:
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
