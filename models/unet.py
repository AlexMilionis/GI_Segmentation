import monai.networks.nets as monai_nets
import torch.nn as nn


class UNet(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.spatial_dims = 2
        self.in_channels = 3
        self.classes = 1
        self.channels = [64, 128, 256, 512, 1024]
        self.strides = [2, 2, 2, 2] 
        self.num_res_units = 0
        self.norm = 'batch'
        self.activation = 'relu'
        self.dropout = 0 #0.1
        self.bias = True
        self.deep_supervision = False
        self.model = self._build_model()


    def _build_model(self):
        return monai_nets.UNet(
            spatial_dims=self.spatial_dims,  # <-- 2D mode
            in_channels=self.in_channels,
            out_channels=self.classes,
            channels=self.channels,  # Example channel progression
            strides=self.strides,  # Downsampling steps
            num_res_units=self.num_res_units,  # Residual blocks per stage
            norm=self.norm,  # BatchNorm by default
            bias = self.bias,
            act=self.activation,  # Activation function
            dropout=self.dropout
        )

    def forward(self, x):
        return self.model(x)