import pytorch_lightning as L
import torch
import torch.nn.functional as F
from torch import nn


class DummyTokenClassifier(L.LightningModule):
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        vocab_size=10000,
        embedding_dim=256,
        number_of_classes=2,
        scheduler: torch.optim.lr_scheduler = None,
        *,
        compile: bool,
    ):
        super().__init__()
        self.save_hyperparameters()

        # Simple model architecture
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.conv1 = nn.Conv1d(embedding_dim, 128, kernel_size=3)
        self.fc = nn.Linear(128, number_of_classes)

        self.train_acc = L.Accuracy(task="multiclass", num_classes=number_of_classes)
        self.val_acc = L.Accuracy(task="multiclass", num_classes=number_of_classes)
        self.test_acc = L.Accuracy(task="multiclass", num_classes=number_of_classes)
        # for averaging loss across batches
        self.train_loss = L.MeanMetric()
        self.val_loss = L.MeanMetric()
        self.test_loss = L.MeanMetric()
        # for tracking best so far validation accuracy
        self.val_acc_best = L.MaxMetric()

    def forward(self, x):
        # x: [batch_size, seq_length]
        x = self.embedding(x)  # [batch_size, seq_length, embedding_dim]
        x = x.permute(0, 2, 1)  # [batch_size, embedding_dim, seq_length]
        x = F.relu(self.conv1(x))
        x = F.max_pool1d(x, x.size(2)).squeeze(2)
        return self.fc(x)

    def on_train_start(self) -> None:
        """Lightning hook that is called when training begins."""
        # by default lightning executes validation step sanity checks before training starts,
        # so it's worth to make sure validation metrics don't store results from these checks
        self.val_loss.reset()
        self.val_acc.reset()
        self.val_acc_best.reset()

    def model_step(self, batch: tuple[torch.Tensor, torch.Tensor]):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        loss = F.cross_entropy(logits, y)
        preds = torch.argmax(logits, dim=1)
        return loss, preds, y

    # Implement training, validation, and test steps using model_step
    def training_step(self, batch, batch_idx):
        loss, preds, y = self.model_step(batch)
        acc = self.train_acc(preds, y)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train_acc", acc, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, preds, y = self.model_step(batch)
        acc = self.val_acc(preds, y)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_acc", acc, on_step=False, on_epoch=True, prog_bar=True)

    def test_step(self, batch, batch_idx):
        loss, preds, y = self.model_step(batch)
        acc = self.test_acc(preds, y)
        self.log("test_loss", loss, on_step=False, on_epoch=True)
        self.log("test_acc", acc, on_step=False, on_epoch=True)

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
