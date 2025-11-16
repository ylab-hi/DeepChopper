from functools import partial

import torch

from . import basic_module, llm
from .basic_module import TokenClassificationLit


class DeepChopper:
    """DeepChopper: A genomic language model for chimera artifact detection.

    This class provides convenient methods to load DeepChopper models from checkpoints or
    from pretrained models on the Hugging Face Hub, and to push trained models to the Hub.

    Example:
        Load a pretrained model:
        >>> model = DeepChopper.from_pretrained("yangliz5/deepchopper")

        Load from a local checkpoint:
        >>> model = DeepChopper.from_checkpoint("path/to/checkpoint.ckpt")

        Push a model to Hugging Face Hub:
        >>> model = DeepChopper.to_hub("username/model-name", "path/to/checkpoint.ckpt")
    """

    @staticmethod
    def to_hub(
        model_name: str,
        checkpoint_path: str,
        *,
        commit_message: str = "Upload DeepChopper model",
        private: bool = False,
        token: str | None = None,
    ):
        """Load a model from a checkpoint and push it to the Hugging Face Hub.

        Args:
            model_name: The repository ID on Hugging Face Hub (format: username/model-name)
            checkpoint_path: Path to the local checkpoint file (.ckpt)
            commit_message: Commit message for the upload (default: "Upload DeepChopper model")
            private: Whether to create a private repository (default: False)
            token: Hugging Face API token. If None, uses the stored token from `huggingface-cli login`

        Returns:
            The loaded TokenClassificationLit model

        Example:
            >>> model = DeepChopper.to_hub(
            ...     "username/deepchopper-v1",
            ...     "epoch_012_f1_0.9947.ckpt",
            ...     commit_message="Upload DeepChopper v1.0",
            ...     private=False
            ... )
        """
        model = DeepChopper.from_checkpoint(checkpoint_path)

        # Prepare kwargs for push_to_hub
        push_kwargs = {
            "repo_id": model_name,
            "commit_message": commit_message,
            "private": private,
        }

        if token is not None:
            push_kwargs["token"] = token

        model.push_to_hub(**push_kwargs)
        return model

    @staticmethod
    def from_checkpoint(checkpoint_path: str):
        """Load a DeepChopper model from a local checkpoint file.

        This method creates a new TokenClassificationLit model with the HyenaDNA backbone
        and loads the weights from the specified checkpoint file.

        Args:
            checkpoint_path: Path to the checkpoint file (.ckpt) containing model weights

        Returns:
            A TokenClassificationLit model loaded with the checkpoint weights

        Example:
            >>> model = DeepChopper.from_checkpoint("epoch_012_f1_0.9947.ckpt")

        Note:
            This function loads checkpoints with weights_only=False to support PyTorch Lightning
            checkpoints containing optimizer/scheduler configs. Only load checkpoints from trusted sources.
        """
        model = TokenClassificationLit(
            net=llm.hyena.TokenClassificationModule(
                number_of_classes=2,
                backbone_name="hyenadna-small-32k-seqlen",
                freeze_backbone=False,
                head=llm.TokenClassificationHead(
                    input_size=256,
                    lin1_size=1024,
                    lin2_size=1024,
                    num_class=2,
                    use_identity_layer_for_qual=True,
                    use_qual=True,
                ),
            ),
            optimizer=partial(  # type: ignore[arg-type]
                torch.optim.Adam,
                lr=0.0002,
                weight_decay=0,
            ),
            scheduler=partial(torch.optim.lr_scheduler.ReduceLROnPlateau, mode="min", factor=0.1, patience=10),  # type: ignore[arg-type]
            criterion=basic_module.ContinuousIntervalLoss(lambda_penalty=0),
            compile=False,
        )
        # weights_only=False is required for PyTorch Lightning checkpoints that contain
        # optimizer/scheduler configs (functools.partial). Only use with trusted checkpoints.
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint["state_dict"])
        return model

    @staticmethod
    def from_pretrained(model_name: str):
        """Load a pretrained DeepChopper model from the Hugging Face Hub.

        This method downloads and loads a pretrained model from the Hugging Face Hub.
        The model architecture is automatically configured to match the expected
        HyenaDNA-based token classification setup.

        Args:
            model_name: The repository ID on Hugging Face Hub (e.g., "yangliz5/deepchopper")

        Returns:
            A TokenClassificationLit model loaded with pretrained weights

        Example:
            >>> model = DeepChopper.from_pretrained("yangliz5/deepchopper")

        Note:
            This requires an internet connection to download the model from Hugging Face Hub.
            The model will be cached locally after the first download.
        """
        return TokenClassificationLit.from_pretrained(
            model_name,
            net=llm.hyena.TokenClassificationModule(
                number_of_classes=2,
                backbone_name="hyenadna-small-32k-seqlen",
                freeze_backbone=False,
                head=llm.TokenClassificationHead(
                    input_size=256,
                    lin1_size=1024,
                    lin2_size=1024,
                    num_class=2,
                    use_identity_layer_for_qual=True,
                    use_qual=True,
                ),
            ),
            optimizer=partial(  # type: ignore[arg-type]
                torch.optim.Adam,
                lr=0.0002,
                weight_decay=0,
            ),
            scheduler=partial(torch.optim.lr_scheduler.ReduceLROnPlateau, mode="min", factor=0.1, patience=10),  # type: ignore[arg-type]
            criterion=basic_module.ContinuousIntervalLoss(lambda_penalty=0),
            compile=False,
        )
