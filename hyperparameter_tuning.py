from lightning.pytorch.tuner import Tuner
from network_module import Net
import numpy as np
from hydra.utils import instantiate
import os

def tune_hyperparameters(cfg, trainer, net, dataset, logger):
    
    tuner = Tuner(trainer)

    # Find the optimal learning rate
    # suggested_lrs = []
    # for i in range(3):
    #     lr_finder = tuner.lr_find(model=net, datamodule=dataset)
    #     suggested_lrs.append(lr_finder.suggestion())
    #     # print(f"Run {i+1} suggested LR: {suggested_lrs[-1]}")
    # optimal_lr = np.exp(np.mean(np.log(suggested_lrs)))
    # print(f"Average suggested learning rate: {optimal_lr}")
    lr_finder = tuner.lr_find(model=net, datamodule=dataset)
    optimal_lr = lr_finder.suggestion()
    logger.experiment.add_scalar("Calculated Learning Rate", optimal_lr, 0)  # Logs at step 0

    cfg.lr = optimal_lr
    # Update scheduler eta_min to 0.1 * optimal_lr if using cosine scheduler
    cfg.scheduler.eta_min = 0.1 * float(optimal_lr)
    # print(f"Updated scheduler eta_min to: {cfg.scheduler.eta_min}")

    # # Recreate the net with updated scheduler config
    # net = Net(
    #     model=model,
    #     criterion=instantiate(cfg.criterion),
    #     optimizer=cfg.optimizer,
    #     lr=optimal_lr,
    #     scheduler=cfg.scheduler,
    # )

    # fig = lr_finder.plot(suggest=True)
    

    # Find optimal batch size
    # batch_size_finder = tuner.scale_batch_size(model=net, datamodule=dataset, mode="binsearch", ckpt_path=os.path.join(logger.log_dir, "batch_size_finder.ckpt"))
    # print(f"Optimal batch size: {batch_size_finder}")

    return cfg