import monai.networks.nets as monai_nets
import torch.nn as nn


class UNet(nn.Module):
    def __init__(self, model_name="unet", spatial_dims=2, in_channels=3, classes=1, 
                 channels=[64, 128, 256, 512, 1024], strides=[2, 2, 2, 2], 
                 num_res_units=0, norm='batch', activation='relu', dropout=0, 
                 bias=True, deep_supervision=False):
        super().__init__()
        self.model_name = model_name
        self.spatial_dims = spatial_dims
        self.in_channels = in_channels
        self.classes = classes
        self.channels = channels
        self.strides = strides
        self.num_res_units = num_res_units
        self.norm = norm
        self.activation = activation
        self.dropout = dropout
        self.bias = bias
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