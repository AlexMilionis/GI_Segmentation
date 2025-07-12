import hydra
import lightning as L
import torch
from hydra.utils import instantiate
from lightning.pytorch import loggers
from lightning.pytorch.tuner import Tuner
from monai.networks.nets.efficientnet import get_efficientnet_image_size
from datamodule import KvasirSEGDataset
from network_module import Net
import warnings
import os
from visualize_results import visualization_grid

L.seed_everything(42, workers=True)
torch.set_float32_matmul_precision("medium")
os.environ["NO_ALBUMENTATIONS_UPDATE"]="1"
os.environ["HYDRA_FULL_ERROR"]="1"
warnings.filterwarnings("ignore")

@hydra.main(config_path="config", config_name="overfit_config", version_base=None)
def main(cfg):
    logger = loggers.TensorBoardLogger("logs/", name=str(cfg.run_name))
    model = instantiate(cfg.model.object)

    if cfg.img_size == "derived":
        img_size = get_efficientnet_image_size(model.model_name)
    else:
        img_size = cfg.img_size

    dataset = KvasirSEGDataset(batch_size=cfg.batch_size, img_size=img_size)

    net = Net(
        model=model,
        criterion=instantiate(cfg.criterion),
        optimizer=cfg.optimizer,
        lr=cfg.lr,
        scheduler=cfg.scheduler,
    )

    trainer = instantiate(cfg.trainer, logger=logger)

    # if efficientnetb5, b6, or b7, use binsearch to find the largest batch size
    if cfg.model.object.model_name in ["efficientnet-b5", "efficientnet-b6", "efficientnet-b7"]:
        tuner = Tuner(trainer)
        tuner.scale_batch_size(net, dataset, mode="binsearch")
    
    trainer.fit(net, dataset)
    trainer.test(net, dataset)
    images, masks, predicted_masks = trainer.predict()
    
    # Create and log visualization grid
    N = 5 
    fig = visualization_grid(images[:N], masks[:N], predicted_masks[:N])
    logger.experiment.add_figure("Predictions Grid", fig, global_step=trainer.global_step)



if __name__ == "__main__":
    main()
