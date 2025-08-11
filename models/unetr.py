import monai.networks.nets as monai_nets
import torch.nn as nn


class UNETR(nn.Module):
    def __init__(self, 
                 model_name="unetr", 
                 spatial_dims=2,
                 in_channels=3,
                 classes=1,
                 feature_size=48,
                 dropout_rate=0.1,
                 norm_name='batch', 
                 image_size =(512, 512)
                 ):
        super().__init__()
        self.model_name = model_name
        self.spatial_dims = spatial_dims
        self.in_channels = in_channels
        self.classes = classes
        self.feature_size = feature_size
        self.dropout_rate = dropout_rate
        self.norm_name = norm_name
        self.image_size = image_size
        self.model = self._build_model()


    def _build_model(self):
        return monai_nets.UNETR(
            spatial_dims=self.spatial_dims,
            in_channels=self.in_channels,
            out_channels=self.classes,
            img_size=self.image_size,
            feature_size=self.feature_size,
            dropout_rate=self.dropout_rate,
            norm_name=self.norm_name,
        )


    def forward(self, x):
        return self.model(x)