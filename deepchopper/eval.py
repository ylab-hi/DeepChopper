import os
from typing import TYPE_CHECKING, Any

import hydra
from omegaconf import DictConfig

from .utils import (
    RankedLogger,
    extras,
    instantiate_callbacks,
    instantiate_loggers,
    log_hyperparameters,
    task_wrapper,
)

if TYPE_CHECKING:
    from lightning import Callback, LightningDataModule, LightningModule, Trainer
    from lightning.pytorch.loggers import Logger

log = RankedLogger(__name__, rank_zero_only=True)


@task_wrapper
def evaluate(cfg: DictConfig) -> tuple[dict[str, Any], dict[str, Any]]:
    """Evaluates given checkpoint on a datamodule testset.

    This method is wrapped in optional @task_wrapper decorator, that controls the behavior during
    failure. Useful for multiruns, saving info about the crash, etc.

    :param cfg: DictConfig configuration composed by Hydra.
    :return: Tuple[dict, dict] with metrics and dict with all instantiated objects.
    """
    assert cfg.ckpt_path

    log.info(f"Instantiating datamodule <{cfg.data._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.data)

    log.info(f"Instantiating model <{cfg.model._target_}>")
    model: LightningModule = hydra.utils.instantiate(cfg.model)

    log.info("Instantiating callbacks...")
    callbacks: list[Callback] = instantiate_callbacks(cfg.get("callbacks"))

    log.info("Instantiating loggers...")
    logger: list[Logger] = instantiate_loggers(cfg.get("logger"))

    log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(cfg.trainer, logger=logger, callbacks=callbacks)

    object_dict = {
        "cfg": cfg,
        "datamodule": datamodule,
        "model": model,
        "logger": logger,
        "trainer": trainer,
    }

    if logger:
        log.info("Logging hyperparameters!")
        log_hyperparameters(object_dict)

    if datamodule.hparams.predict_data_path is None:
        log.info("Starting testing!")
        trainer.test(model=model, datamodule=datamodule, ckpt_path=cfg.ckpt_path)
    else:
        # for predictions use trainer.predict(...)
        import multiprocess.context as ctx

        ctx._force_start_method("spawn")
        trainer.predict(model=model, dataloaders=datamodule, ckpt_path=cfg.ckpt_path, return_predictions=False)

    metric_dict = trainer.callback_metrics
    return metric_dict, object_dict


@hydra.main(version_base="1.3", config_path=os.getenv("DC_CONFIG_PATH", "configs"), config_name="eval.yaml")
def main(cfg: DictConfig) -> None:
    """Main entry point for evaluation.

    :param cfg: DictConfig configuration composed by Hydra.
    """
    # apply extra utilities
    # (e.g. ask for tags if none are provided in cfg, print cfg tree, etc.)
    extras(cfg)

    evaluate(cfg)
