import hydra
import lightning as L
import torch
import numpy as np
from hydra.utils import instantiate
from lightning.pytorch import loggers
from lightning.pytorch.tuner import Tuner
# from monai.networks.nets.efficientnet import get_efficientnet_image_size
from lightning.pytorch.callbacks import ModelCheckpoint
from datamodule import KvasirSEGDataset
from network_module import Net
import warnings
import os
from hyperparameter_tuning import tune_hyperparameters
from visualize_results import save_visualization_grid


L.seed_everything(42, workers=True)
torch.set_float32_matmul_precision("medium")
os.environ["NO_ALBUMENTATIONS_UPDATE"]="1"
os.environ["HYDRA_FULL_ERROR"]="1"
warnings.filterwarnings("ignore")

@hydra.main(config_path="config", config_name="config_unet", version_base=None)
def main(cfg, load_existing=False):
    logger = loggers.TensorBoardLogger("../logs/", name=str(cfg.run_name))
    model = instantiate(cfg.model.object)
    
    dataset = KvasirSEGDataset(batch_size=cfg.batch_size, img_size=cfg.img_size)

    net = Net(
        model=model,
        criterion=instantiate(cfg.criterion),
        optimizer=cfg.optimizer,
        lr=cfg.lr,
        scheduler=cfg.scheduler,
    )
    if load_existing:
        print("Loading existing model weights...")
        checkpoint_path = os.path.join(logger.log_dir, "model_checkpoint.ckpt")
        net = net.load_from_checkpoint(checkpoint_path)

    checkpoint_callback = ModelCheckpoint(
        dirpath=logger.log_dir,
        filename="model_checkpoint",
        save_top_k=1
    )

    trainer = instantiate(cfg.trainer, logger=logger, callbacks=[checkpoint_callback])
    cfg = tune_hyperparameters(cfg, trainer, net, dataset, logger)
    net.lr = cfg.lr
    # Recreate the net with updated scheduler config
    net = Net(
        model=model,
        criterion=instantiate(cfg.criterion),
        optimizer=cfg.optimizer,
        lr=cfg.lr,
        scheduler=cfg.scheduler,
    )

    trainer.fit(net, dataset)
    trainer.test(net, dataset)
    prediction_outputs = trainer.predict(net, dataset)
    
    # Create and log visualization grid
    save_visualization_grid(prediction_outputs, logger.log_dir)

    # save the configuration file in the logs directory
    config_path = os.path.join(logger.log_dir, cfg.config_name)
    with open(config_path, "w") as f:
        f.write(cfg.pretty())


if __name__ == "__main__":
    main()
