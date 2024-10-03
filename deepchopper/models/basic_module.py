from typing import Any

import torch
from huggingface_hub import PyTorchModelHubMixin
from lightning import LightningModule
from torch import nn
from torchmetrics import MaxMetric, MeanMetric
from torchmetrics.classification import F1Score, Precision, Recall


class ContinuousIntervalLoss(nn.Module):
    """A custom loss function that penalizes the model for predicting different classes in consecutive positions."""

    def __init__(self, lambda_penalty: float = 0, **kwargs):
        super().__init__()
        self.base = torch.nn.CrossEntropyLoss(**kwargs)
        self.lambda_penalty = lambda_penalty

    @property
    def ignore_index(self):
        return self.base.ignore_index

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        loss = self.base(pred, target)
        if self.lambda_penalty == 0:
            return loss
        valid_mask = target != self.ignore_index
        true_pred = pred.argmax(-1)[valid_mask]
        true_target = target[valid_mask]
        penalty = self.lambda_penalty * (true_pred[1:] != true_target[:-1]).float().mean()
        return loss + penalty


class TokenClassificationLit(LightningModule, PyTorchModelHubMixin):
    """A PyTorch Lightning module for training a token classification model."""

    def __init__(
        self,
        net: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler,
        criterion: nn.Module,
        *,
        compile: bool,
    ):
        """Genomics Benchmark CNN model for PyTorch Lightning.

        :param net: The CNN model.
        :param scheduler: The learning rate scheduler to use for training.
        """
        super().__init__()

        self.example_input_array = {
            "input_ids": torch.randint(0, 11, (1, 1000)),
            "input_quals": torch.rand(1, 1000),
        }  # [batch, seq_len]

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False, ignore=["net", "criterion"])
        self.net = net
        # loss function
        self.criterion = criterion

        # metric objects for calculating and averaging accuracy across batches
        self.train_acc = F1Score(
            task="binary", num_classes=net.number_of_classes, ignore_index=self.criterion.ignore_index
        )
        self.val_acc = F1Score(
            task="binary", num_classes=net.number_of_classes, ignore_index=self.criterion.ignore_index
        )
        self.test_acc = F1Score(
            task="binary", num_classes=net.number_of_classes, ignore_index=self.criterion.ignore_index
        )

        self.test_precision = Precision(
            task="binary", num_classes=net.number_of_classes, ignore_index=self.criterion.ignore_index
        )
        self.test_recall = Recall(
            task="binary", num_classes=net.number_of_classes, ignore_index=self.criterion.ignore_index
        )

        # for averaging loss across batches
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()
        # for tracking best so far validation accuracy
        self.val_acc_best = MaxMetric()

    def forward(
        self,
        input_ids: torch.Tensor,
        input_quals: torch.Tensor,
    ) -> torch.Tensor:
        """Perform a forward pass through the model `self.net`.

        :param x: A tensor of images.
        :return: A tensor of logits.
        """
        return self.net(input_ids, input_quals)

    def on_train_start(self) -> None:
        """Lightning hook that is called when training begins."""
        # by default lightning executes validation step sanity checks before training starts,
        # so it's worth to make sure validation metrics don't store results from these checks
        self.val_loss.reset()
        self.val_acc.reset()
        self.val_acc_best.reset()

    def model_step(self, batch: tuple[torch.Tensor, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Perform a single model step on a batch of data.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target labels.
        :return: A tuple containing (in order):
            - A tensor of losses.
            - A tensor of predictions.
            - A tensor of target labels.
        """
        input_ids = batch["input_ids"]
        input_quals = batch["input_quals"]
        logits = self.forward(input_ids, input_quals)
        loss = self.criterion(logits.reshape(-1, logits.size(-1)), batch["labels"].long().view(-1))
        preds = torch.argmax(logits, dim=-1)
        return loss, preds, batch["labels"]

    def training_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Perform a single training step on a batch of data from the training set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        :return: A tensor of losses between model predictions and targets.
        """
        loss, preds, targets = self.model_step(batch)

        # update and log metrics
        self.train_loss(loss)
        self.train_acc(preds, targets)

        self.log("train/loss", self.train_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/f1", self.train_acc, on_step=False, on_epoch=True, prog_bar=True)

        # return loss or backpropagation will fail
        return loss

    def on_train_epoch_end(self) -> None:
        """Lightning hook that is called when a training epoch ends."""

    def validation_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        """Perform a single validation step on a batch of data from the validation set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        """
        loss, preds, targets = self.model_step(batch)

        # update and log metrics
        self.val_loss(loss)
        self.val_acc(preds, targets)

        self.log("val/loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/f1", self.val_acc, on_step=False, on_epoch=True, prog_bar=True)

    def on_validation_epoch_end(self) -> None:
        """Lightning hook that is called when a validation epoch ends."""
        acc = self.val_acc.compute()  # get current val acc
        self.val_acc_best(acc)  # update best so far val acc
        # log `val_acc_best` as a value through `.compute()` method, instead of as a metric object
        # otherwise metric would be reset by lightning after each epoch
        self.log("val/f1_best", self.val_acc_best.compute(), sync_dist=True, prog_bar=True)

    def test_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        """Perform a single test step on a batch of data from the test set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        """
        loss, preds, targets = self.model_step(batch)

        # update and log metrics
        self.test_loss(loss)
        self.test_acc(preds, targets)

        self.test_precision(preds, targets)
        self.test_recall(preds, targets)

        self.log("test/loss", self.test_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/f1", self.test_acc, on_step=False, on_epoch=True, prog_bar=True)

    def on_test_epoch_end(self) -> None:
        """Lightning hook that is called when a test epoch ends."""
        self.log("test/precision", self.test_precision)
        self.log("test/recall", self.test_recall)

    def predict_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Perform a single prediction step on a batch of data from the test set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        """
        input_ids = batch["input_ids"]
        input_quals = batch["input_quals"]
        logits = self.forward(input_ids, input_quals)
        return logits, batch["labels"]

    def setup(self, stage: str) -> None:
        """Lightning hook that is called at the beginning of fit (train + validate), validate, test, or predict.

        This is a good hook when you need to build models dynamically or adjust something about
        them. This hook is called on every process when using DDP.

        :param stage: Either `"fit"`, `"validate"`, `"test"`, or `"predict"`.
        """
        if self.hparams.compile and stage == "fit":
            self.net = torch.compile(self.net)

    def configure_optimizers(self) -> dict[str, Any]:
        """Choose what optimizers and learning-rate schedulers to use in your optimization.

        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        Examples:
            https://lightning.ai/docs/pytorch/latest/common/lightning_module.html#configure-optimizers

        :return: A dict containing the configured optimizers and learning-rate schedulers to be used for training.
        """
        optimizer = self.hparams.optimizer(params=self.trainer.model.parameters())

        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/loss",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}
