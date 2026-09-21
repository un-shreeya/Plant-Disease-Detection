"""
=============================================================================
 EfficientNet-B0 Plant Disease Classification Model
 File: models/efficientnet_model.py
=============================================================================
 PURPOSE:
   Defines the deep learning model architecture for plant disease
   classification using EfficientNet-B0 with transfer learning.

 ARCHITECTURE OVERVIEW:
   ┌──────────────────────────────────────────────────┐
   │  Input Image (224 × 224 × 3)                     │
   ├──────────────────────────────────────────────────┤
   │  EfficientNet-B0 Backbone (ImageNet Pre-trained) │
   │  ├── Stem: Conv3×3 → BN → Swish                 │
   │  ├── MBConv Blocks (16 blocks)                   │
   │  │   ├── Depthwise Separable Convolution         │
   │  │   ├── Squeeze-and-Excitation (SE) Block       │
   │  │   └── Skip Connections                        │
   │  └── Head: Conv1×1 → BN → Swish                 │
   ├──────────────────────────────────────────────────┤
   │  Custom Classification Head                      │
   │  ├── Adaptive Average Pooling (1×1)              │
   │  ├── Flatten                                     │
   │  ├── Dropout (p=0.3)                             │
   │  ├── Linear(1280, 512) → ReLU → BatchNorm       │
   │  ├── Dropout (p=0.2)                             │
   │  ├── Linear(512, 256) → ReLU → BatchNorm        │
   │  ├── Dropout (p=0.2)                             │
   │  └── Linear(256, num_classes)                    │
   └──────────────────────────────────────────────────┘

 KEY EQUATIONS:
   Compound Scaling (EfficientNet):
     depth  = α^φ  (where α = 1.2 for B0)
     width  = β^φ  (where β = 1.1 for B0)
     resolution = γ^φ  (where γ = 1.15 for B0)
     Constraint: α · β² · γ² ≈ 2

   Swish Activation:
     f(x) = x · σ(x) = x / (1 + e^(-x))

   Squeeze-and-Excitation:
     s = σ(W₂ · ReLU(W₁ · GAP(F)))
     F̃ = s ⊙ F  (channel-wise scaling)

 USAGE:
   from models.efficientnet_model import PlantDiseaseModel

   model = PlantDiseaseModel(num_classes=38, pretrained=True)
   output = model(image_tensor)  # shape: (batch, 38)
=============================================================================
"""

import torch
import torch.nn as nn
from torchvision import models


class PlantDiseaseModel(nn.Module):
    """
    EfficientNet-B0 based plant disease classifier with custom head.

    This model uses transfer learning from ImageNet pre-trained weights.
    The backbone extracts hierarchical visual features (edges → textures →
    patterns → disease-specific features), and the custom head maps these
    to disease class probabilities.

    ATTRIBUTES:
      backbone (nn.Module): EfficientNet-B0 feature extractor
      classifier (nn.Sequential): Custom classification head
      num_classes (int): Number of output disease classes

    PARAMETERS:
      Total:    ~5.7M parameters
      Backbone: ~5.3M (pre-trained, can be frozen)
      Head:     ~0.4M (trained from scratch)

    MODEL SIZE:
      ~22 MB (FP32), ~11 MB (FP16)
    """

    def __init__(self, num_classes: int = 38, pretrained: bool = True):
        """
        Initialize the PlantDiseaseModel.

        PARAMETERS:
          num_classes (int): Number of disease classes to predict.
                            Default is 38 for PlantVillage dataset.
          pretrained (bool): If True, loads ImageNet pre-trained weights
                            for the EfficientNet-B0 backbone.
                            Set False for training from scratch.

        WHAT HAPPENS:
          1. Loads EfficientNet-B0 backbone with/without ImageNet weights
          2. Removes the original classification head
          3. Attaches our custom head with dropout regularization
          4. The backbone's feature dimension is 1280 for EfficientNet-B0
        """
        super(PlantDiseaseModel, self).__init__()

        self.num_classes = num_classes

        # ── Load EfficientNet-B0 Backbone ──────────────────────────────────
        # EfficientNet-B0 uses compound scaling to balance depth, width,
        # and resolution. It achieves ImageNet top-1 accuracy of 77.1%
        # with only 5.3M parameters — much smaller than ResNet-50 (25.6M).
        if pretrained:
            weights = models.EfficientNet_B3_Weights.IMAGENET1K_V1
            self.backbone = models.efficientnet_b3(weights=weights)
        else:
            self.backbone = models.efficientnet_b3(weights=None)

        # ── Get backbone feature dimension ─────────────────────────────────
        # EfficientNet-B3 outputs 1536-dimensional feature vectors
        # after global average pooling
        backbone_out_features = self.backbone.classifier[1].in_features  # 1536

        # ── Remove original classifier head ────────────────────────────────
        # Replace with identity so forward pass returns raw features
        self.backbone.classifier = nn.Identity()

        # ── Custom Classification Head ─────────────────────────────────────
        # We add a deeper head than default to learn disease-specific
        # feature combinations. Dropout layers prevent overfitting on
        # the relatively small PlantVillage dataset.
        #
        # Architecture rationale:
        #   1280 → 512: First compression — reduces features while keeping
        #               important disease indicators
        #   512 → 256:  Second compression — learns abstract disease patterns
        #   256 → 38:   Final classification layer
        #
        # BatchNorm after each linear stabilizes training and allows
        # higher learning rates on the custom head.
        self.classifier = nn.Sequential(
            # Dropout after backbone prevents co-adaptation of features
            nn.Dropout(p=0.3),

            # First fully connected block
            nn.Linear(backbone_out_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),

            # Second fully connected block
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),

            # Output layer — raw logits (no softmax, handled by CrossEntropyLoss)
            nn.Linear(256, num_classes),
        )

        # ── Initialize classifier weights ──────────────────────────────────
        # Kaiming initialization is optimal for ReLU networks.
        # Mathematical basis: Var(w) = 2/n_in prevents signal
        # vanishing/exploding through deep layers.
        self._initialize_weights()

    def _initialize_weights(self):
        """
        Initialize the custom classifier head weights using Kaiming Normal.

        MATHEMATICAL BASIS (He et al., 2015):
          For ReLU networks, the variance of weights should be:
            Var(w) = 2 / n_input

          This prevents the signal from vanishing (too small) or exploding
          (too large) as it passes through deep layers.

          Kaiming Normal: w ~ N(0, √(2/n_input))

        Biases are initialized to zero, which is standard practice.
        BatchNorm weights are initialized to 1 (scale) and 0 (shift).
        """
        for module in self.classifier.modules():
            if isinstance(module, nn.Linear):
                # Kaiming init for layers followed by ReLU
                nn.init.kaiming_normal_(
                    module.weight,
                    mode='fan_out',      # Based on output dimension
                    nonlinearity='relu'  # Accounts for ReLU's zero-mean truncation
                )
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.BatchNorm1d):
                # BatchNorm: γ=1, β=0 (identity transform initially)
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the model.

        FLOW:
          Input (B, 3, 224, 224)
            → EfficientNet-B0 backbone
            → Feature vector (B, 1280)
            → Custom classifier head
            → Logits (B, num_classes)

        PARAMETERS:
          x (torch.Tensor): Batch of images, shape (B, 3, 224, 224)
                           where B is batch size.
                           Pixel values should be normalized to [0,1]
                           then standardized with ImageNet mean/std.

        RETURNS:
          torch.Tensor: Raw logits of shape (B, num_classes).
                       Apply softmax to get probabilities:
                       probs = F.softmax(output, dim=1)
        """
        # Extract features through the backbone
        # Output shape: (batch_size, 1280) after global avg pooling
        features = self.backbone(x)

        # Classify through our custom head
        # Output shape: (batch_size, num_classes)
        logits = self.classifier(features)

        return logits

    def freeze_backbone(self):
        """
        Freeze all backbone parameters to prevent updating during training.

        USE CASE:
          During the first few epochs, we freeze the pre-trained backbone
          and only train the custom classifier head. This prevents the
          random head gradients from corrupting the learned backbone features.

        TRAINING STRATEGY (Two-Phase):
          Phase 1 (epochs 1-5):  Freeze backbone, train head only
          Phase 2 (epochs 6+):   Unfreeze all, fine-tune with lower LR

        MATHEMATICAL IMPACT:
          Frozen parameters have requires_grad=False, so:
            ∂L/∂w_backbone = 0  (no gradient computation)
            w_backbone_new = w_backbone_old  (no weight update)
        """
        for param in self.backbone.parameters():
            param.requires_grad = False
        print("   🔒 Backbone frozen — only classifier head will be trained")

    def unfreeze_backbone(self):
        """
        Unfreeze all backbone parameters for fine-tuning.

        Called after the classifier head has converged (Phase 2).
        Use a smaller learning rate (e.g., 10× smaller) for the backbone
        to avoid destroying pre-trained features.

        After unfreezing:
          ∂L/∂w_backbone ≠ 0  (gradients flow through backbone)
          w_backbone_new = w_backbone_old - η · ∂L/∂w  (updates occur)
        """
        for param in self.backbone.parameters():
            param.requires_grad = True
        print("   🔓 Backbone unfrozen — all parameters will be trained")

    def get_grad_cam_target_layer(self):
        """
        Returns the target layer for Grad-CAM visualization.

        For EfficientNet-B0, the last convolutional layer is in the
        features block. This layer captures the highest-level spatial
        features before global average pooling.

        The last convolutional block produces 1280 feature maps at
        7×7 spatial resolution (for 224×224 input), which provides
        good localization granularity for disease regions.

        RETURNS:
          nn.Module: The target convolutional layer for Grad-CAM
        """
        # EfficientNet-B0's last feature block (before classifier)
        return self.backbone.features[-1]

    def get_model_info(self) -> dict:
        """
        Returns model metadata as a dictionary.

        Useful for the /api/model-info endpoint and evaluation reports.

        RETURNS:
          dict: Contains parameter count, model size, architecture details
        """
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        # Estimate model size in MB (FP32: 4 bytes per parameter)
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


def create_model(num_classes: int = 38, pretrained: bool = True) -> PlantDiseaseModel:
    """
    Factory function to create a PlantDiseaseModel.

    This is the recommended way to create the model — it handles
    device placement and prints model summary.

    PARAMETERS:
      num_classes (int): Number of disease classes
      pretrained (bool): Whether to use ImageNet pre-trained weights

    RETURNS:
      PlantDiseaseModel: Initialized model on appropriate device

    EXAMPLE:
      model = create_model(num_classes=38, pretrained=True)
      model.freeze_backbone()  # For Phase 1 training
    """
    model = PlantDiseaseModel(num_classes=num_classes, pretrained=pretrained)

    # Print model summary
    info = model.get_model_info()
    print("\n" + "=" * 50)
    print("  🧠 MODEL SUMMARY")
    print("=" * 50)
    print(f"  Architecture:     {info['architecture']}")
    print(f"  Input Size:       {info['input_size']}")
    print(f"  Output Classes:   {info['num_classes']}")
    print(f"  Total Params:     {info['total_parameters']:,}")
    print(f"  Trainable Params: {info['trainable_parameters']:,}")
    print(f"  Model Size:       {info['model_size_mb']} MB")
    print(f"  Classifier:       {info['classifier_head']}")
    print("=" * 50)

    return model


# =============================================================================
# STANDALONE TEST
# =============================================================================
if __name__ == "__main__":
    """
    Quick test to verify the model architecture works correctly.

    Creates a model, passes a dummy batch through it, and verifies
    the output shape matches expectations.
    """
    print("🧪 Testing PlantDiseaseModel...")

    # Create model
    model = create_model(num_classes=38, pretrained=True)

    # Create dummy input (batch of 4 images, 3 channels, 224x224)
    dummy_input = torch.randn(4, 3, 224, 224)

    # Forward pass
    with torch.no_grad():
        output = model(dummy_input)

    print(f"\n  Input shape:  {dummy_input.shape}")
    print(f"  Output shape: {output.shape}")
    print(f"  Expected:     torch.Size([4, 38])")

    assert output.shape == (4, 38), f"Shape mismatch: {output.shape}"
    print("\n  ✅ Model architecture test PASSED!")

    # Test freeze/unfreeze
    model.freeze_backbone()
    trainable_after_freeze = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Trainable after freeze: {trainable_after_freeze:,}")

    model.unfreeze_backbone()
    trainable_after_unfreeze = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Trainable after unfreeze: {trainable_after_unfreeze:,}")

    # Test Grad-CAM target layer
    target_layer = model.get_grad_cam_target_layer()
    print(f"  Grad-CAM target: {type(target_layer).__name__}")

    print("\n  ✅ All tests PASSED!")
