from enum import Enum

import torch.nn as nn
from models.tipsv2 import TIPSv2


class BACKBONES(str, Enum):
    RADIO = "radio"
    DINOV3 = "dinov3"
    DINOV2 = "dinov2"
    SIGLIP2 = "siglip2"
    CLIP = "clip"
    TIPSV2 = "tipsv2"

    def __str__(self):
        return self.value  # For prettier argparse help output

class FeatureExtractor(nn.Module):

    def __init__(self, model_name, height=768, use_local=False):
        super().__init__()
        self.model = TIPSv2(height=height)

    def forward(self, x):
        return self.model(x)
