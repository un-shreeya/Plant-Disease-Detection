import torch
import torch.nn as nn
from torchvision import models

class PlantDiseaseModel(nn.Module):
    def __init__(self, num_classes: int = 38, pretrained: bool = True):
        super(PlantDiseaseModel, self).__init__()
        self.num_classes = num_classes
        
        if pretrained:
            weights = models.EfficientNet_B3_Weights.IMAGENET1K_V1
            self.backbone = models.efficientnet_b3(weights=weights)
        else:
            self.backbone = models.efficientnet_b3(weights=None)

        # Extract output feature dimension (1536 for EfficientNet-B3)
        backbone_out_features = self.backbone.classifier[1].in_features
        # Replace default classifier with Identity
        self.backbone.classifier = nn.Identity()
        # Custom Classification Head
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(backbone_out_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(256, num_classes),
        )
        self._initialize_weights()

    def _initialize_weights(self):
        for module in self.classifier.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(
                    module.weight, mode="fan_out", nonlinearity="relu"
                )
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.BatchNorm1d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        logits = self.classifier(features)
        return logits

    def freeze_backbone(self):
        """Freezes backbone parameters for Phase 1 head training."""
        for param in self.backbone.parameters():
            param.requires_grad = False
        print("Backbone frozen — training custom head only.")

    def unfreeze_backbone(self):
        """Unfreezes backbone parameters for Phase 2 fine-tuning."""
        for param in self.backbone.parameters():
            param.requires_grad = True
        print("Backbone unfrozen — training all layers.")

    def get_grad_cam_target_layer(self) -> nn.Module:
        """Returns the target convolutional layer for Grad-CAM generation."""
        return self.backbone.features[-1]

    def get_model_info(self) -> dict:
        """Returns metadata about the architecture and parameter counts."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(
            p.numel() for p in self.parameters() if p.requires_grad
        )
        model_size_mb = (total_params * 4) / (1024 * 1024)

        return {
            "architecture": "EfficientNet-B3",
            "num_classes": self.num_classes,
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "model_size_mb": round(model_size_mb, 2),
            "input_size": "224x224x3",
            "backbone": "EfficientNet-B3 (ImageNet pre-trained)",
            "classifier_head": "1536→512→256→38",
            "activation": "ReLU + Swish (backbone)",
            "regularization": "Dropout(0.3, 0.2, 0.2) + BatchNorm",
        }


def create_model(
    num_classes: int = 38, pretrained: bool = True
) -> PlantDiseaseModel:
    model = PlantDiseaseModel(num_classes=num_classes, pretrained=pretrained)

    info = model.get_model_info()
    print(" MODEL SUMMARY")
    print(f"  Architecture:     {info['architecture']}")
    print(f"  Input Size:       {info['input_size']}")
    print(f"  Output Classes:   {info['num_classes']}")
    print(f"  Total Params:     {info['total_parameters']:,}")
    print(f"  Trainable Params: {info['trainable_parameters']:,}")
    print(f"  Model Size:       {info['model_size_mb']} MB")
    print(f"  Classifier:       {info['classifier_head']}")
    return model


if __name__ == "__main__":
    print(" Testing PlantDiseaseModel...")
    model = create_model(num_classes=38, pretrained=True)
    dummy_input = torch.randn(4, 3, 224, 224)

    with torch.no_grad():
        output = model(dummy_input)
    print(f"\n  Input shape:  {dummy_input.shape}")
    print(f"  Output shape: {output.shape}")
    assert output.shape == (4, 38), f"Shape mismatch: {output.shape}"
    print("  ✅ Architecture verification passed!")
