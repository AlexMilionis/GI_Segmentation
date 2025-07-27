import monai.networks.nets as monai_nets
import torch.nn as nn

class SegResNet(nn.Module):
    def __init__(self, model_name="segresnet", spatial_dims=2, in_channels=3, classes=1, 
                 init_filters=32, dropout_prob=0.0, norm="batch"):
        super().__init__()
        self.model_name = model_name
        self.spatial_dims = spatial_dims
        self.in_channels = in_channels
        self.classes = classes
        self.init_filters = init_filters
        self.dropout_prob = dropout_prob
        self.norm = norm
        self.model = self._build_model()

    def _build_model(self):
        return monai_nets.SegResNet(
            spatial_dims=self.spatial_dims,
            in_channels=self.in_channels,
            out_channels=self.classes,
            init_filters=self.init_filters,
            dropout_prob=self.dropout_prob,
            norm=self.norm
        )

    def forward(self, x):
        return self.model(x)