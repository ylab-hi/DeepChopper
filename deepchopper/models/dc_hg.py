from functools import partial

import torch

from . import basic_module, llm
from .basic_module import TokenClassificationLit


class DeepChopper:
    @classmethod
    def from_pretrained(cls, model_name: str, **kwargs) -> "DeepChopper":
        """Load a pretrained model."""
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
            optimizer=partial(
                torch.optim.Adam,
                lr=0.0002,
                weight_decay=0,
            ),
            scheduler=partial(torch.optim.lr_scheduler.ReduceLROnPlateau, mode="min", factor=0.1, patience=10),
            criterion=basic_module.ContinuousIntervalLoss(lambda_penalty=0),
            compile=False,
        )
