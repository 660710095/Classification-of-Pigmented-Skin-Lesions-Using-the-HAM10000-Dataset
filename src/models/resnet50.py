"""src/models/resnet50.py - ResNet-50 builder with customizable Dense head, Dropout, and Backbone freezing."""

from typing import Optional
import torch.nn as nn
import torchvision.models as models
from src.models import register_model


@register_model("resnet50")
def build_resnet50(
    num_classes: int = 7,
    pretrained: bool = True,
    dropout_rate: float = 0.3,
    hidden_dim: Optional[int] = 512,
    freeze_backbone: bool = False,
    **kwargs
) -> nn.Module:
    """
    สร้างโมเดล ResNet-50 พร้อม Classifier Head ที่ปรับแต่งได้

    พารามิเตอร์:
        num_classes: จำนวนคลาสเอาต์พุต (7 คลาส)
        pretrained: โหลด ImageNet pretrained weights
        dropout_rate: สัดส่วน Dropout เพื่อลด Overfitting (เช่น 0.0, 0.3, 0.5)
        hidden_dim: จำนวนโหนดในชั้น Dense คั่นกลาง (ถ้าใส่ None จะใช้ Dense ชั้นเดียวแบบคลาสสิก)
        freeze_backbone: ล็อกชั้น Convolution ล่างไม่ให้อัปเดต เทรนเฉพาะชั้นหัว
    """
    weights = models.ResNet50_Weights.DEFAULT if pretrained else None
    model = models.resnet50(weights=weights, **kwargs)

    # หากต้องการ Freeze backbone เพื่อเทรนเฉพาะ Classifier head
    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False

    in_features = model.fc.in_features

    # ปรับแต่ง Dense layer (Classifier head)
    if hidden_dim is not None and hidden_dim > 0:
        # Multi-layer Head: Linear -> BatchNorm -> ReLU -> Dropout -> Linear
        model.fc = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate),
            nn.Linear(hidden_dim, num_classes),
        )
    else:
        # Single-layer Head พร้อม Dropout
        if dropout_rate > 0.0:
            model.fc = nn.Sequential(
                nn.Dropout(p=dropout_rate),
                nn.Linear(in_features, num_classes),
            )
        else:
            model.fc = nn.Linear(in_features, num_classes)

    return model
