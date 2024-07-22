import torch
import torch.nn.functional as F  # noqa: N812
from torch import nn


class TokenClassificationCnnHead(nn.Module):
    def __init__(
        self,
        input_size,
        number_of_classes,
        num_filters,
        filter_sizes,
    ):
        super().__init__()
        self.number_of_classes = number_of_classes

        self.qual_linear1 = nn.Linear(num_filters, number_of_classes)

        layers = []
        in_channels = input_size

        for idx, fs in enumerate(filter_sizes):
            layers.append(nn.Conv1d(in_channels=in_channels, out_channels=num_filters, kernel_size=fs, padding="same"))
            layers.append(nn.BatchNorm1d(num_filters[idx]))
            layers.append(nn.ReLU())
            in_channels = num_filters[idx]

        self.model = nn.Sequential(*layers)
        self.dense = nn.Linear(in_channels, number_of_classes)

    def forward(self, x: torch.Tensor, input_quals: torch.Tensor):
        x = F.relu(x + self.qual_linear1(input_quals.unsqueeze(-1)))
        x = x.transpose(1, 2)  # (batch, num_filters, seq_len)
        x = self.model(x)  # (batch, num_filters, seq_len)
        x = x.transpose(1, 2)  # (batch, seq_len, num_filters)
        return self.dense(x)


class TokenClassificationHead(nn.Module):
    """Token classification head for the model."""

    def __init__(
        self,
        input_size: int,
        num_class: int,
        lin1_size: int,
        lin2_size: int,
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
        if lin1_size != lin2_size:
            msg = f"{lin1_size=} and {lin2_size=} must be equal"
            raise ValueError(msg)

        super().__init__()
        self.use_qual = use_qual
        self.activation = nn.ReLU()
        self.linear1 = nn.Linear(input_size, lin1_size)
        self.linear2 = nn.Linear(lin1_size, lin2_size)
        self.linear3 = nn.Linear(lin2_size, num_class)

        self.qual_linear1 = nn.Identity() if use_identity_layer_for_qual else nn.Linear(1, lin1_size)

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
