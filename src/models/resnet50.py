"""src/models/resnet50.py - ResNet-50 builder for HAM10000."""

import torch.nn as nn
import torchvision.models as models
from src.models import register_model


@register_model("resnet50")
def build_resnet50(num_classes: int = 7, pretrained: bool = True, **kwargs) -> nn.Module:
    weights = models.ResNet50_Weights.DEFAULT if pretrained else None
    model = models.resnet50(weights=weights, **kwargs)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model
