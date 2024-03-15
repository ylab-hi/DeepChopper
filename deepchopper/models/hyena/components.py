import torch
from torch import nn
from transformers import AutoModel, PretrainedConfig, PreTrainedModel
from transformers.modeling_outputs import TokenClassifierOutput
from transformers.utils import logging

logging.set_verbosity_info()
logger = logging.get_logger("transformers")

HyenadnaMaxLengths = {
    "hyenadna-tiny-1k-seqlen": 1024,
    "hyenadna-small-32k-seqlen": 32768,
    "hyenadna-medium-160k-seqlen": 160000,
    "hyenadna-medium-450k-seqlen": 450000,  # T4 up to here
    "hyenadna-large-1m-seqlen": 1_000_000,  # only A100 (paid tier)
}


class TokenClassificationHead(nn.Module):
    """Token classification head for the model."""

    def __init__(
        self,
        input_size: int,
        lin1_size: int,
        lin2_size: int,
        num_class: int,
        *,
        use_identity_layer_for_qual: bool,
        use_qual: bool,
    ):
        """Initialize the neural network model.

        Parameters:
            input_size (int): The size of the input features.
            lin1_size (int): The size of the first linear layer.
            lin2_size (int): The size of the second linear layer.
            num_class (int): The number of output classes.
            use_identity_layer_for_qual (bool): Whether to use an identity layer for quality.
            use_qual (bool): Whether to use quality in the model.
        """
        super().__init__()
        self.use_qual = use_qual
        self.activation = nn.ReLU()
        self.linear1 = nn.Linear(input_size, lin1_size)
        self.linear2 = nn.Linear(lin1_size, lin2_size)
        self.linear3 = nn.Linear(lin2_size, num_class)

        self.qual_linear1 = (
            nn.Identity() if use_identity_layer_for_qual else nn.Linear(1, lin1_size)
        )

    def forward(self, x: torch.Tensor, input_quals: torch.Tensor) -> torch.Tensor:
        """Forward pass through the neural network model.

        Parameters:
            x (torch.Tensor): Input tensor to the model.
            input_quals (torch.Tensor): Input tensor representing qualities.

        Returns:
            torch.Tensor: Output tensor from the model.

        This method performs a forward pass through the neural network model.
        It takes in two input tensors, x and input_quals, and processes them through the model layers.
        The output tensor is returned from the model after passing through the linear and activation layers.
        If the 'use_qual' flag is set to True, the input_quals tensor is used to calculate a residual value that is added to the output tensor before passing through the linear and activation layers again.
        This helps incorporate qualities into the model's predictions.
        If the 'use_qual' flag is set to False, the input_quals tensor is not used and the output tensor from the first linear layer is directly passed through the second linear and activation layers.
        The final output tensor is returned from the model after passing through the third linear layer.
        Note: The activation function used in the model is specified by self.activation and should be set during model initialization.
        """
        output = self.activation(self.linear1(x))

        if self.use_qual:
            residual = output + self.qual_linear1(input_quals.unsqueeze(-1))  # may add activation
            output = self.activation(self.linear2(residual) + residual)
        else:
            output = self.activation(self.linear2(output))

        return self.linear3(output)


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
        self.hyenadna = AutoModel.from_pretrained(
            f"LongSafari/{hyenadna_model}-hf", trust_remote_code=True
        )

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
