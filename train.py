import hydra
import lightning as L
import torch
import numpy as np
from hydra.utils import instantiate
from lightning.pytorch import loggers
from lightning.pytorch.tuner import Tuner
from monai.networks.nets.efficientnet import get_efficientnet_image_size
from lightning.pytorch.callbacks import ModelCheckpoint
from datamodule import KvasirSEGDataset
from network_module import Net
import warnings
import os
from hyperparameter_tuning import tune_hyperparameters
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

    checkpoint_callback = ModelCheckpoint(
        dirpath=logger.log_dir,
        filename="lr_finder_checkpoint",
        save_top_k=1
    )

    trainer = instantiate(cfg.trainer, logger=logger, callbacks=[checkpoint_callback])

    # # if efficientnetb5, b6, or b7, use binsearch to find the largest batch size
    # if cfg.model.object.model_name in ["efficientnet-b5", "efficientnet-b6", "efficientnet-b7"]:
    #     tuner = Tuner(trainer)
    #     tuner.scale_batch_size(net, dataset, mode="binsearch")

    # tuner = Tuner(trainer)

    # # Find the optimal learning rate
    # suggested_lrs = []
    # for i in range(3):
    #     lr_finder = tuner.lr_find(model=net, datamodule=dataset)
    #     suggested_lrs.append(lr_finder.suggestion())
    #     print(f"Run {i+1} suggested LR: {suggested_lrs[-1]}")
    # optimal_lr = np.exp(np.mean(np.log(suggested_lrs)))
    # print(f"Average suggested learning rate: {optimal_lr}")

    # # lr_finder = tuner.lr_find(model=net, datamodule=dataset, ckpt_path=os.path.join(logger.log_dir, "lr_finder.ckpt"))
    # # suggested_lr = lr_finder.suggestion()
    # # print(f"Suggested learning rate: {suggested_lr}")
    # # fig = lr_finder.plot(suggest=True)
    # # logger.experiment.add_figure("LR Finder", fig, global_step=0)

    # # Find optimal batch size
    # # batch_size_finder = tuner.scale_batch_size(model=net, datamodule=dataset, mode="binsearch", ckpt_path=os.path.join(logger.log_dir, "batch_size_finder.ckpt"))
    # # print(f"Optimal batch size: {batch_size_finder}")
    
    # net.lr = optimal_lr
    
    # # Update scheduler eta_min to 0.1 * optimal_lr if using cosine scheduler
    # cfg.scheduler.eta_min = 0.1 * optimal_lr
    # print(f"Updated scheduler eta_min to: {cfg.scheduler.eta_min}")
    
    # # Recreate the net with updated scheduler config
    # net = Net(
    #     model=model,
    #     criterion=instantiate(cfg.criterion),
    #     optimizer=cfg.optimizer,
    #     lr=optimal_lr,
    #     scheduler=cfg.scheduler,
    # )

    # net, optimal_lr = tune_hyperparameters(cfg, trainer, net, dataset, model)

    trainer.fit(net, dataset)
    trainer.test(net, dataset)
    prediction_outputs = trainer.predict(net, dataset)
    
    # Create and log visualization grid
    total_samples = sum(batch['images'].shape[0] for batch in prediction_outputs)
    print(f"Total samples to visualize: {total_samples}")
    fig = visualization_grid(prediction_outputs)
    logger.experiment.add_figure("Prediction Grid", fig, global_step=trainer.global_step)

if __name__ == "__main__":
    main()
