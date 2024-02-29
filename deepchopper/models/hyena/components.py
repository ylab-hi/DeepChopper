import torch
from torch import nn
from transformers import AutoModel, PreTrainedModel
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
    ):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, lin1_size),
            nn.ReLU(),
            nn.Linear(lin1_size, lin2_size),
            nn.ReLU(),
            nn.Linear(lin2_size, num_class),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class HyenaDNAForTokenClassification(PreTrainedModel):
    def __init__(
        self,
        config,
        input_size: int = 256,
        lin1_size: int = 2048,
        lin2_size: int = 1024,
        num_class: int = 2,
        **kwargs,
    ):
        super().__init__(config, trust_remote_code=True, **kwargs)
        self.num_class = num_class
        self.backbone_model_name = config.name_or_path
        self.backbone = AutoModel.from_config(config)

        self.head = TokenClassificationHead(
            input_size=input_size,
            lin1_size=lin1_size,
            lin2_size=lin2_size,
            num_class=num_class,
        )

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor,
        # input_quals: torch.Tensor,
        inputs_embeds: torch.FloatTensor | None = None,
        output_hidden_states: bool | None = None,
        return_dict: bool | None = None,
    ) -> torch.Tensor:
        # logger.info(f"{input_ids.shape=}")
        # logger.info(f"{labels.shape=}")
        # logger.info(f"{input_quals.shape=}")

        transformer_outputs = self.backbone(
            input_ids,
            inputs_embeds=inputs_embeds,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        input_ids.shape[0]
        hidden_states = transformer_outputs[0]
        # logger.info(f"{hidden_states.shape=}")
        logits = self.head(hidden_states)

        (torch.eq(input_ids, self.backbone.config.pad_token_id).long().argmax(-1) - 1).to(
            logits.device
        )
        labels = labels.to(logits.device)
        loss_fct = nn.CrossEntropyLoss()

        loss = loss_fct(logits.view(-1, self.num_class), labels.view(-1))
        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=transformer_outputs.hidden_states,
        )
