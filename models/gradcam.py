"""
=============================================================================
 Grad-CAM Explainability Module
 File: models/gradcam.py
=============================================================================
 PURPOSE:
   Generates Grad-CAM (Gradient-weighted Class Activation Mapping) heatmaps
   to explain which regions of a leaf image the model focused on to make
   its disease prediction.

 WHAT IS GRAD-CAM?
   Grad-CAM is a visual explainability technique that produces a heatmap
   showing the "important" regions in an input image for a particular
   class prediction. It uses the gradients flowing into the final
   convolutional layer to produce a coarse localization map.

 MATHEMATICAL FOUNDATION:
   Given a convolutional feature map A^k (k-th channel) and target class c:

   Step 1: Compute importance weights (global average pooling of gradients)
     α_k^c = (1/Z) × Σᵢ Σⱼ (∂y^c / ∂A_{ij}^k)

     where Z = width × height of the feature map
     This tells us "how important is channel k for class c?"

   Step 2: Weighted combination of feature maps + ReLU
     L_Grad-CAM^c = ReLU(Σ_k α_k^c × A^k)

     ReLU keeps only features with POSITIVE influence on class c.
     Negative values indicate features for OTHER classes.

   Step 3: Upsample to input image size and overlay

 WHY IS THIS IMPORTANT FOR THE RESEARCH PAPER?
   - Proves the model is looking at disease-relevant regions (lesions, spots)
   - Not just memorizing background or irrelevant features
   - Builds trust for farmers: "The AI focused on THIS brown spot"
   - Reviewers LOVE explainable AI — it strengthens the paper significantly

 USAGE:
   from models.gradcam import GradCAMExplainer

   explainer = GradCAMExplainer(model, device)
   heatmap, overlay = explainer.generate(image_tensor, original_image)
=============================================================================
"""

import os
import sys
import base64
import io
import numpy as np

import torch
import torch.nn.functional as F
from PIL import Image

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class GradCAMExplainer:
    """
    Generates Grad-CAM heatmaps for the PlantDiseaseModel.

    This class hooks into the last convolutional layer of EfficientNet-B0
    to capture:
      1. Forward activations (feature maps A^k)
      2. Backward gradients (∂y^c / ∂A^k)

    Then combines them to produce a heatmap showing which spatial regions
    were most important for the predicted class.

    ATTRIBUTES:
      model (nn.Module): The trained PlantDiseaseModel
      device (torch.device): CPU or CUDA device
      target_layer (nn.Module): The convolutional layer to explain
      activations (torch.Tensor): Captured forward-pass feature maps
      gradients (torch.Tensor): Captured backward-pass gradients
    """

    def __init__(self, model, device):
        """
        Initialize the Grad-CAM explainer.

        PARAMETERS:
          model: Trained PlantDiseaseModel instance
          device: torch.device (CPU or CUDA)

        WHAT HAPPENS:
          1. Identifies the target layer (last conv block of EfficientNet-B0)
          2. Registers forward hook to capture activations
          3. Registers backward hook to capture gradients
        """
        self.model = model
        self.device = device
        self.model.eval()  # Must be in eval mode for consistent results

        # Storage for hooks
        self.activations = None
        self.gradients = None

        # Get the target layer for Grad-CAM
        # For EfficientNet-B0, this is the last feature block
        # It outputs 1280 channels at 7×7 spatial resolution (for 224×224 input)
        self.target_layer = model.get_grad_cam_target_layer()

        # ── Register Hooks ──────────────────────────────────────────────
        # Hooks are PyTorch's mechanism to intercept forward/backward passes
        # without modifying the model code.

        # Forward hook: captures the output (activations) of the target layer
        # during the forward pass. Shape: (batch, 1280, 7, 7)
        self.target_layer.register_forward_hook(self._forward_hook)

        # Backward hook: captures the gradients flowing back through the
        # target layer during backpropagation. Same shape as activations.
        self.target_layer.register_full_backward_hook(self._backward_hook)

    def _forward_hook(self, module, input, output):
        """
        Forward hook — captures activations during forward pass.

        Called automatically by PyTorch when data flows through target_layer.
        Stores the output feature maps for later use in Grad-CAM computation.

        PARAMETERS:
          module: The target layer module
          input: Input to the layer (not used)
          output: Output activations from the layer, shape (B, C, H, W)
        """
        self.activations = output.detach()

    def _backward_hook(self, module, grad_input, grad_output):
        """
        Backward hook — captures gradients during backward pass.

        Called automatically by PyTorch when gradients flow back through
        the target layer. Stores gradients for importance weight computation.

        PARAMETERS:
          module: The target layer module
          grad_input: Gradients w.r.t. layer input (not used)
          grad_output: Tuple of gradients w.r.t. layer output
        """
        self.gradients = grad_output[0].detach()

    def generate(
        self,
        input_tensor: torch.Tensor,
        original_image: np.ndarray = None,
        target_class: int = None,
    ) -> dict:
        """
        Generate Grad-CAM heatmap for an input image.

        ALGORITHM:
          1. Forward pass through model → get prediction and activations
          2. Select target class (predicted class if not specified)
          3. Backward pass from target class score → get gradients
          4. Global average pool gradients → importance weights α_k
          5. Weighted sum of activations → L = Σ_k α_k × A^k
          6. Apply ReLU → keep only positive influences
          7. Normalize to [0, 1] range
          8. Resize to original image dimensions
          9. Overlay on original image with colormap

        PARAMETERS:
          input_tensor (torch.Tensor): Preprocessed image, shape (1, 3, 224, 224)
          original_image (np.ndarray): Original image (H, W, 3), values [0, 255]
                                      Used for overlay visualization.
          target_class (int): Class to explain. If None, uses predicted class.

        RETURNS:
          dict with keys:
            'heatmap': np.ndarray — Raw heatmap (H, W), values [0, 1]
            'overlay': np.ndarray — Heatmap overlaid on original image
            'overlay_base64': str — Base64-encoded PNG of overlay (for web API)
            'predicted_class': int — The class index the model predicted
            'confidence': float — Prediction confidence (softmax probability)
        """
        # Ensure model is in eval mode
        self.model.eval()

        # Move input to device
        input_tensor = input_tensor.to(self.device)
        if input_tensor.dim() == 3:
            input_tensor = input_tensor.unsqueeze(0)  # Add batch dimension

        # ── Step 1: Forward Pass ───────────────────────────────────────
        # This triggers the forward hook, capturing activations
        input_tensor.requires_grad_(True)
        output = self.model(input_tensor)

        # ── Step 2: Select Target Class ────────────────────────────────
        if target_class is None:
            target_class = output.argmax(dim=1).item()

        # Get prediction confidence
        probabilities = F.softmax(output, dim=1)
        confidence = probabilities[0, target_class].item()

        # ── Step 3: Backward Pass ──────────────────────────────────────
        # Zero all previous gradients
        self.model.zero_grad()

        # Create one-hot target for the predicted class
        # This tells PyTorch: "compute gradients w.r.t. this class's score"
        one_hot = torch.zeros_like(output)
        one_hot[0, target_class] = 1.0

        # Backward pass — triggers backward hook, capturing gradients
        output.backward(gradient=one_hot, retain_graph=True)

        # ── Step 4: Compute Importance Weights ─────────────────────────
        # α_k = (1/Z) × Σᵢ Σⱼ (∂y^c / ∂A_{ij}^k)
        # Global average pooling of gradients across spatial dimensions
        # Shape: (1, 1280) — one weight per channel
        weights = torch.mean(self.gradients, dim=[2, 3], keepdim=True)

        # ── Step 5: Weighted Combination ───────────────────────────────
        # L = Σ_k α_k × A^k
        # Multiply each activation channel by its importance weight
        # Shape: (1, 1280, 7, 7) × (1, 1280, 1, 1) → sum → (1, 1, 7, 7)
        cam = torch.sum(weights * self.activations, dim=1, keepdim=True)

        # ── Step 6: ReLU ───────────────────────────────────────────────
        # Keep only positive influences (features FOR this class)
        cam = F.relu(cam)

        # ── Step 7: Normalize to [0, 1] ───────────────────────────────
        cam = cam.squeeze()  # Shape: (7, 7)
        if cam.max() > 0:
            cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        cam = cam.cpu().numpy()

        # ── Step 8: Resize to Original Image Dimensions ────────────────
        if original_image is not None:
            h, w = original_image.shape[:2]
        else:
            h, w = 224, 224

        # Use PIL for high-quality resizing (bilinear interpolation)
        cam_pil = Image.fromarray((cam * 255).astype(np.uint8))
        cam_resized = cam_pil.resize((w, h), Image.BILINEAR)
        heatmap = np.array(cam_resized).astype(np.float32) / 255.0

        # ── Step 9: Create Overlay ─────────────────────────────────────
        overlay = None
        overlay_base64 = None

        if original_image is not None:
            overlay = self._create_overlay(original_image, heatmap)
            overlay_base64 = self._to_base64(overlay)

        return {
            'heatmap': heatmap,
            'overlay': overlay,
            'overlay_base64': overlay_base64,
            'predicted_class': target_class,
            'confidence': confidence,
        }

    def _create_overlay(
        self,
        original_image: np.ndarray,
        heatmap: np.ndarray,
        alpha: float = 0.4,
    ) -> np.ndarray:
        """
        Overlays the Grad-CAM heatmap on the original image.

        Uses a JET colormap to convert the grayscale heatmap to a
        red-yellow-green-blue color visualization, then blends it
        with the original image.

        COLOR INTERPRETATION:
          Red/Yellow: High activation — model focused strongly here
          Green/Blue: Low activation — model paid less attention
          Dark Blue:  Near-zero activation — not relevant for prediction

        PARAMETERS:
          original_image: Original leaf image (H, W, 3), uint8 [0, 255]
          heatmap: Grad-CAM heatmap (H, W), float32 [0, 1]
          alpha: Blending factor. 0.4 means 40% heatmap + 60% original.

        RETURNS:
          np.ndarray: Blended overlay image (H, W, 3), uint8 [0, 255]
        """
        try:
            import cv2

            # Apply JET colormap to heatmap
            heatmap_colored = cv2.applyColorMap(
                (heatmap * 255).astype(np.uint8),
                cv2.COLORMAP_JET
            )
            # OpenCV uses BGR, convert to RGB
            heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)

            # Ensure original image is uint8
            if original_image.dtype != np.uint8:
                original_image = (original_image * 255).astype(np.uint8)

            # Resize heatmap to match original image if needed
            if heatmap_colored.shape[:2] != original_image.shape[:2]:
                heatmap_colored = cv2.resize(
                    heatmap_colored,
                    (original_image.shape[1], original_image.shape[0])
                )

            # Blend: overlay = α × heatmap + (1 - α) × original
            overlay = cv2.addWeighted(
                heatmap_colored, alpha,
                original_image, 1 - alpha,
                0
            )
            return overlay

        except ImportError:
            # Fallback without OpenCV — simple overlay
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.cm as cm

            # Apply JET colormap using matplotlib
            cmap = cm.get_cmap('jet')
            heatmap_colored = (cmap(heatmap)[:, :, :3] * 255).astype(np.uint8)

            if original_image.dtype != np.uint8:
                original_image = (original_image * 255).astype(np.uint8)

            overlay = (
                alpha * heatmap_colored.astype(np.float32) +
                (1 - alpha) * original_image.astype(np.float32)
            ).astype(np.uint8)
            return overlay

    def _to_base64(self, image: np.ndarray) -> str:
        """
        Converts a numpy image array to a base64-encoded PNG string.

        Used by the web API to send the overlay image directly in
        the JSON response without saving to disk.

        PARAMETERS:
          image: numpy array (H, W, 3), uint8 [0, 255]

        RETURNS:
          str: Base64-encoded PNG image string, prefixed with data URI
        """
        pil_image = Image.fromarray(image)
        buffer = io.BytesIO()
        pil_image.save(buffer, format='PNG', quality=95)
        buffer.seek(0)
        img_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        return f"data:image/png;base64,{img_base64}"


def generate_gradcam_for_image(
    model,
    image_path: str,
    device: torch.device,
    class_names: list = None,
) -> dict:
    """
    Convenience function to generate Grad-CAM for a single image file.

    Handles the complete pipeline: load image → preprocess → generate
    Grad-CAM → return results with class name.

    PARAMETERS:
      model: Trained PlantDiseaseModel
      image_path: Path to the leaf image file
      device: CPU or CUDA device
      class_names: List of class names (for human-readable output)

    RETURNS:
      dict: Same as GradCAMExplainer.generate() plus 'class_name' key
    """
    from torchvision import transforms

    # ImageNet normalization (same as training)
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    # Load original image (for overlay)
    original_image = np.array(Image.open(image_path).convert('RGB'))

    # Preprocess for model
    pil_image = Image.open(image_path).convert('RGB')
    input_tensor = transform(pil_image).unsqueeze(0)

    # Generate Grad-CAM
    explainer = GradCAMExplainer(model, device)
    result = explainer.generate(input_tensor, original_image)

    # Add class name if available
    if class_names and result['predicted_class'] < len(class_names):
        result['class_name'] = class_names[result['predicted_class']]

    return result


# =============================================================================
# STANDALONE TEST
# =============================================================================
if __name__ == "__main__":
    """
    Test the Grad-CAM explainer with a dummy input.
    """
    from models.efficientnet_model import create_model

    print("🧪 Testing Grad-CAM Explainer...")

    device = torch.device('cpu')
    model = create_model(num_classes=38, pretrained=True)
    model = model.to(device)

    # Create dummy image
    dummy_input = torch.randn(1, 3, 224, 224)
    dummy_original = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)

    # Generate Grad-CAM
    explainer = GradCAMExplainer(model, device)
    result = explainer.generate(dummy_input, dummy_original)

    print(f"\n  Predicted Class: {result['predicted_class']}")
    print(f"  Confidence:      {result['confidence']:.4f}")
    print(f"  Heatmap Shape:   {result['heatmap'].shape}")
    print(f"  Overlay Shape:   {result['overlay'].shape}")
    print(f"  Base64 Length:   {len(result['overlay_base64'])} chars")

    print("\n  ✅ Grad-CAM test PASSED!")
