import monai.networks.nets as monai_nets
import torch.nn as nn


class AttentionUnet(nn.Module):
    def __init__(self, model_name="attention_unet", spatial_dims=2, in_channels=3, classes=1, 
                 channels=[64, 128, 256, 512, 1024], strides=[2, 2, 2, 2], dropout=0):
        super().__init__()
        self.model_name = model_name
        self.spatial_dims = spatial_dims
        self.in_channels = in_channels
        self.classes = classes
        self.channels = channels
        self.strides = strides
        self.dropout = dropout
        self.model = self._build_model()


    def _build_model(self):
        return monai_nets.AttentionUnet(
            spatial_dims=self.spatial_dims,  # <-- 2D mode
            in_channels=self.in_channels,
            out_channels=self.classes,
            channels=self.channels,  # Example channel progression
            strides=self.strides,  # Downsampling steps
            dropout=self.dropout
        )

    def forward(self, x):
        return self.model(x)