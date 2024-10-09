from pathlib import Path

import torch
from lightning.pytorch.callbacks import BasePredictionWriter


class CustomWriter(BasePredictionWriter):
    def __init__(self, output_dir, write_interval="epoch"):
        super().__init__(write_interval)
        self.output_dir = Path(output_dir)

    def write_on_batch_end(self, trainer, pl_module, prediction, batch_indices, batch, batch_idx, dataloader_idx):
        folder = self.output_dir / str(dataloader_idx)
        if not folder.exists():
            folder.mkdir(parents=True, exist_ok=True)

        save_prediction = {
            "prediction": prediction[0].cpu(),
            "target": prediction[1].to(torch.int64).cpu(),
            "seq": batch["input_ids"].cpu(),
            "qual": batch["input_quals"].cpu(),
            "id": batch["id"].to(torch.int64).cpu(),
        }

        torch.save(save_prediction, folder / f"{trainer.global_rank}_{batch_idx}.pt")

    def write_on_epoch_end(self, trainer, pl_module, predictions, batch_indices):
        # WARN: This is a simple implementation that saves all predictions in a single file
        if not self.output_dir.exists():
            self.output_dir.mkdir(parents=False, exist_ok=True)

        torch.save(predictions, self.output_dir / "predictions.pt")
