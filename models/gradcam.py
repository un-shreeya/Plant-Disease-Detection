import base64
import io
import os
import sys
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class GradCAMExplainer:
    def __init__(self, model, device):
        self.model = model
        self.device = device
        self.model.eval()

        self.activations = None
        self.gradients = None

        self.target_layer = model.get_grad_cam_target_layer()
        self.target_layer.register_forward_hook(self._forward_hook)
        self.target_layer.register_full_backward_hook(self._backward_hook)

    def _forward_hook(self, module, input, output):
        self.activations = output.detach()

    def _backward_hook(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(
        self,
        input_tensor: torch.Tensor,
        original_image: np.ndarray = None,
        target_class: int = None,
    ) -> dict:
        self.model.eval()

        input_tensor = input_tensor.to(self.device)
        if input_tensor.dim() == 3:
            input_tensor = input_tensor.unsqueeze(0)

        input_tensor.requires_grad_(True)
        output = self.model(input_tensor)

        if target_class is None:
            target_class = output.argmax(dim=1).item()

        probabilities = F.softmax(output, dim=1)
        confidence = probabilities[0, target_class].item()

        self.model.zero_grad()
        one_hot = torch.zeros_like(output)
        one_hot[0, target_class] = 1.0

        output.backward(gradient=one_hot, retain_graph=True)

        weights = torch.mean(self.gradients, dim=[2, 3], keepdim=True)
        cam = torch.sum(weights * self.activations, dim=1, keepdim=True)
        cam = F.relu(cam)

        cam = cam.squeeze()
        if cam.max() > 0:
            cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        cam = cam.cpu().numpy()

        if original_image is not None:
            h, w = original_image.shape[:2]
        else:
            h, w = 224, 224

        cam_pil = Image.fromarray((cam * 255).astype(np.uint8))
        cam_resized = cam_pil.resize((w, h), Image.BILINEAR)
        heatmap = np.array(cam_resized).astype(np.float32) / 255.0

        overlay = None
        overlay_base64 = None

        if original_image is not None:
            overlay = self._create_overlay(original_image, heatmap)
            overlay_base64 = self._to_base64(overlay)

        return {
            "heatmap": heatmap,
            "overlay": overlay,
            "overlay_base64": overlay_base64,
            "predicted_class": target_class,
            "confidence": confidence,
        }

    def _create_overlay(
        self,
        original_image: np.ndarray,
        heatmap: np.ndarray,
        alpha: float = 0.4,
    ) -> np.ndarray:
        try:
            import cv2

            heatmap_colored = cv2.applyColorMap(
                (heatmap * 255).astype(np.uint8), cv2.COLORMAP_JET
            )
            heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)

            if original_image.dtype != np.uint8:
                original_image = (original_image * 255).astype(np.uint8)

            if heatmap_colored.shape[:2] != original_image.shape[:2]:
                heatmap_colored = cv2.resize(
                    heatmap_colored,
                    (original_image.shape[1], original_image.shape[0]),
                )

            overlay = cv2.addWeighted(
                heatmap_colored, alpha, original_image, 1 - alpha, 0
            )
            return overlay

        except ImportError:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.cm as cm

            cmap = cm.get_cmap("jet")
            heatmap_colored = (cmap(heatmap)[:, :, :3] * 255).astype(np.uint8)

            if original_image.dtype != np.uint8:
                original_image = (original_image * 255).astype(np.uint8)

            overlay = (
                alpha * heatmap_colored.astype(np.float32)
                + (1 - alpha) * original_image.astype(np.float32)
            ).astype(np.uint8)
            return overlay

    def _to_base64(self, image: np.ndarray) -> str:
        pil_image = Image.fromarray(image)
        buffer = io.BytesIO()
        pil_image.save(buffer, format="PNG", quality=95)
        buffer.seek(0)
        img_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
        return f"data:image/png;base64,{img_base64}"


def generate_gradcam_for_image(
    model,
    image_path: str,
    device: torch.device,
    class_names: list = None,
) -> dict:
    from torchvision import transforms

    transform = transforms.Compose(
        [
            transforms.Resize((256, 256)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
            ),
        ]
    )

    original_image = np.array(Image.open(image_path).convert("RGB"))
    pil_image = Image.open(image_path).convert("RGB")
    input_tensor = transform(pil_image).unsqueeze(0)

    explainer = GradCAMExplainer(model, device)
    result = explainer.generate(input_tensor, original_image)

    if class_names and result["predicted_class"] < len(class_names):
        result["class_name"] = class_names[result["predicted_class"]]

    return result


if __name__ == "__main__":
    from models.efficientnet_model import create_model

    print("🧪 Testing Grad-CAM Explainer...")

    device = torch.device("cpu")
    model = create_model(num_classes=38, pretrained=True)
    model = model.to(device)

    dummy_input = torch.randn(1, 3, 224, 224)
    dummy_original = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)

    explainer = GradCAMExplainer(model, device)
    result = explainer.generate(dummy_input, dummy_original)

    print(f"\n  Predicted Class: {result['predicted_class']}")
    print(f"  Confidence:      {result['confidence']:.4f}")
    print(f"  Heatmap Shape:   {result['heatmap'].shape}")
    print(f"  Overlay Shape:   {result['overlay'].shape}")
    print(f"  Base64 Length:   {len(result['overlay_base64'])} chars")

    print("\n  ✅ Grad-CAM test PASSED!")
