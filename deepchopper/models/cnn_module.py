"""Genomics Benchmark CNN model.

Adapted from https://github.com/ML-Bioinfo-CEITEC/genomic_benchmarks/blob/main/src/genomic_benchmarks/models/torch.py
"""

import lightning as L  # noqa: N812
import torch
import torch.nn.functional as F  # noqa: N812
from torch import nn


class BenchmarkCNN(nn.Module):
    def __init__(self, number_of_classes, vocab_size, num_filters, filter_sizes, embedding_dim=100):
        """Genomics Benchmark CNN model.

        `embedding_dim` = 100 comes from:
        https://github.com/ML-Bioinfo-CEITEC/genomic_benchmarks/tree/main/experiments/torch_cnn_experiments
        """
        super().__init__()
        self.number_of_classes = number_of_classes
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()

        for fs in filter_sizes:
            self.convs.append(
                nn.Conv1d(in_channels=embedding_dim, out_channels=num_filters, kernel_size=fs)
            )
            self.batch_norms.append(nn.BatchNorm1d(num_filters))

        # use number of kernel same as the length of the sequence and average pooling
        # then flatten and use dense layers
        self.dense_model = nn.Sequential(
            nn.Linear(num_filters * len(filter_sizes), number_of_classes),
        )

    def forward(self, x):  # Adding `state` to be consistent with other models
        x = self.embeddings(x)
        x = x.transpose(1, 2)  # [batch_size, embedding_dim, input_len]
        x = [F.relu(conv(x)) for conv, bn in zip(self.convs, self.batch_norms, strict=True)]
        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]
        x = torch.cat(x, 1)
        return self.dense_model(x)


class LitBenchmarkCNN(L.LightningModule):
    def __init__(
        self,
        net: BenchmarkCNN,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        *,
        compile: bool,
    ):
        """Genomics Benchmark CNN model for PyTorch Lightning.

        :param net: The CNN model.
        :param scheduler: The learning rate scheduler to use for training.
        """
        super().__init__()
        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)
        self.net = net
        # loss function
        self.criterion = torch.nn.CrossEntropyLoss()
        # metric objects for calculating and averaging accuracy across batches
        self.train_acc = L.Accuracy(task="multiclass", num_classes=net.number_of_classes)
        self.val_acc = L.Accuracy(task="multiclass", num_classes=net.number_of_classes)
        self.test_acc = L.Accuracy(task="multiclass", num_classes=net.number_of_classes)
        # for averaging loss across batches
        self.train_loss = L.MeanMetric()
        self.val_loss = L.MeanMetric()
        self.test_loss = L.MeanMetric()
        # for tracking best so far validation accuracy
        self.val_acc_best = L.MaxMetric()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Perform a forward pass through the model `self.net`.

        :param x: A tensor of images.
        :return: A tensor of logits.
        """
        return self.net(x)

    def on_train_start(self) -> None:
        """Lightning hook that is called when training begins."""
        # by default lightning executes validation step sanity checks before training starts,
        # so it's worth to make sure validation metrics don't store results from these checks
        self.val_loss.reset()
        self.val_acc.reset()
        self.val_acc_best.reset()

    def model_step(
        self, batch: tuple[torch.Tensor, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Perform a single model step on a batch of data.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target labels.

        :return: A tuple containing (in order):
            - A tensor of losses.
            - A tensor of predictions.
            - A tensor of target labels.
        """
        x, y = batch
        logits = self.forward(x)
        loss = self.criterion(logits, y)
        preds = torch.argmax(logits, dim=1)
        return loss, preds, y

    def training_step(
        self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
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
        self.log("train/acc", self.train_acc, on_step=False, on_epoch=True, prog_bar=True)

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
        self.log("val/acc", self.val_acc, on_step=False, on_epoch=True, prog_bar=True)

    def on_validation_epoch_end(self) -> None:
        """Lightning hook that is called when a validation epoch ends."""
        acc = self.val_acc.compute()  # get current val acc
        self.val_acc_best(acc)  # update best so far val acc
        # log `val_acc_best` as a value through `.compute()` method, instead of as a metric object
        # otherwise metric would be reset by lightning after each epoch
        self.log("val/acc_best", self.val_acc_best.compute(), sync_dist=True, prog_bar=True)

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
        self.log("test/loss", self.test_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/acc", self.test_acc, on_step=False, on_epoch=True, prog_bar=True)

    def on_test_epoch_end(self) -> None:
        """Lightning hook that is called when a test epoch ends."""

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
