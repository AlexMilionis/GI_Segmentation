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

    # # if efficientnetb5, b6, or b7, use binsearch to find the largest batch size
    # if cfg.model.object.model_name in ["efficientnet-b5", "efficientnet-b6", "efficientnet-b7"]:
    #     tuner = Tuner(trainer)
    #     tuner.scale_batch_size(net, dataset, mode="binsearch")

    tuner = Tuner(trainer)

    # Find the optimal learning rate
    lr_finder = tuner.lr_find(model=net, datamodule=dataset)
    suggested_lr = lr_finder.suggestion()
    print(f"Suggested learning rate: {suggested_lr}")
    fig = lr_finder.plot(suggest=True)
    logger.experiment.add_figure("LR Finder", fig, global_step=0)

    # Find optimal batch size
    batch_size_finder = tuner.scale_batch_size(model=net, datamodule=dataset, mode="binsearch")
    print(f"Optimal batch size: {batch_size_finder}")
    
    # trainer.fit(net, dataset)
    # trainer.test(net, dataset)
    # prediction_outputs = trainer.predict(net, dataset)
    
    # # Create and log visualization grid
    # fig = visualization_grid(prediction_outputs, samples_to_visualize=5)
    # logger.experiment.add_figure("Prediction Grid", fig, global_step=trainer.global_step)

if __name__ == "__main__":
    main()
