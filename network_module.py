import lightning as L
import torch
from hydra.utils import instantiate
from monai import metrics as mm


class Net(L.LightningModule):
    def __init__(self, model, criterion, optimizer, lr, scheduler=None):
        super().__init__()
        
        self.model = model

        self.get_dice = mm.DiceMetric(include_background=False, reduction="mean")
        self.get_iou = mm.MeanIoU(include_background=False, reduction="mean")
        self.get_accuracy = mm.ConfusionMatrixMetric(include_background=False, metric_name="accuracy")
        self.get_recall = mm.ConfusionMatrixMetric(include_background=False, metric_name="sensitivity")
        self.get_precision = mm.ConfusionMatrixMetric(include_background=False, metric_name="precision")
        # self.get_hausdorff = mm.HausdorffDistanceMetric(include_background=False, reduction="mean")


        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.lr = lr

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        optimizer = instantiate(self.optimizer, self.parameters(), lr=self.lr)
        if self.scheduler:
            return {
                "optimizer": optimizer,
                "lr_scheduler": instantiate(self.scheduler, optimizer=optimizer),
                "monitor": "val_loss",
            }
        return optimizer

    def training_step(self, batch, batch_idx):
        x, y = batch

        logits = self(x)
        loss = self.criterion(logits, y)
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch

        logits = self(x)
        loss = self.criterion(logits, y)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        preds = (torch.sigmoid(logits) > 0.5).long()
        self.get_dice(preds, y)
        self.get_iou(preds, y)
        self.get_accuracy(preds, y)
        self.get_recall(preds, y)
        self.get_precision(preds, y)

        return loss
    
    def test_step(self, batch, batch_idx):
        x, y = batch

        logits = self(x)
        loss = self.criterion(logits, y)
        self.log("test_loss", loss, prog_bar=True, logger=True)

        preds = (torch.sigmoid(logits) > 0.5).long()
        self.get_dice(preds, y)
        self.get_iou(preds, y)
        self.get_accuracy(preds, y)
        self.get_recall(preds, y)
        self.get_precision(preds, y)
        # self.get_hausdorff(preds, y)

        return loss
    
    def predict_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        preds = (torch.sigmoid(logits) > 0.5).long()
        return {
            'images': x,
            'masks': y,
            'predictions': preds
        }

    def on_validation_epoch_end(self):
        dice = self.get_dice.aggregate()[0].item()
        iou = self.get_iou.aggregate()[0].item()
        accuracy = self.get_accuracy.aggregate()[0].item()
        recall = self.get_recall.aggregate()[0].item()
        precision = self.get_precision.aggregate()[0].item()
        f1 = 2 * (precision * recall) / (precision + recall + 1e-8)
        f2 = 5 * (precision * recall) / (4 * precision + recall + 1e-8)
        

        self.log("val_dice", dice, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log("val_iou", iou, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log("val_accuracy", accuracy, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log("val_recall", recall, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log("val_precision", precision, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log("val_f1", f1, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log("val_f2", f2, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        self.get_dice.reset()
        self.get_iou.reset()
        self.get_accuracy.reset()
        self.get_recall.reset()
        self.get_precision.reset()
    
    def on_test_epoch_end(self):
        dice = self.get_dice.aggregate()[0].item()
        iou = self.get_iou.aggregate()[0].item()
        accuracy = self.get_accuracy.aggregate()[0].item()
        recall = self.get_recall.aggregate()[0].item()
        precision = self.get_precision.aggregate()[0].item()
        f1 = 2 * (precision * recall) / (precision + recall + 1e-8)
        f2 = 5 * (precision * recall) / (4 * precision + recall + 1e-8)
        # hausdorff = self.get_hausdorff.aggregate()[0].item()

        self.log("test_dice", dice, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log("test_iou", iou, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log("test_accuracy", accuracy, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log("test_recall", recall, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log("test_precision", precision, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log("test_f1", f1, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log("test_f2", f2, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        # self.log("test_hausdorff", hausdorff, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        self.get_dice.reset()
        self.get_iou.reset()
        self.get_accuracy.reset()
        self.get_recall.reset()
        self.get_precision.reset()
        # self.get_hausdorff.reset()
