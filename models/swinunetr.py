import monai.networks.nets as monai_nets
import torch.nn as nn


class SwinUNETR(nn.Module):
    def __init__(self, 
                 model_name="swinunetr", 
                 spatial_dims=2,
                 in_channels=3, 
                 classes=1,
                 feature_size=48, 
                 drop_rate=0.1,
                 norm_name='batch', 
                 use_checkpoint=True,
                 ):
        super().__init__()
        self.model_name = model_name
        self.spatial_dims = spatial_dims
        self.in_channels = in_channels
        self.classes = classes
        self.feature_size = feature_size
        self.drop_rate = drop_rate
        self.norm_name = norm_name
        self.use_checkpoint = use_checkpoint
        self.model = self._build_model()


    def _build_model(self):
        return monai_nets.SwinUNETR(
            spatial_dims=self.spatial_dims,  # <-- 2D mode
            in_channels=self.in_channels,
            out_channels=self.classes,
            feature_size=self.feature_size,
            drop_rate=self.drop_rate,
            norm_name=self.norm_name,
            use_checkpoint=self.use_checkpoint
        )


    def forward(self, x):
        return self.model(x)