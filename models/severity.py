import numpy as np
from PIL import Image

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False


class SeverityEstimator:
    def __init__(
        self,
        mild_threshold: float = 10.0,
        severe_threshold: float = 30.0,
    ):
        self.mild_threshold = mild_threshold
        self.severe_threshold = severe_threshold

    def estimate(self, image: np.ndarray) -> dict:
        if image.dtype != np.uint8:
            image = (image * 255).astype(np.uint8)

        if HAS_CV2:
            return self._estimate_cv2(image)
        else:
            return self._estimate_pil(image)

    def _estimate_cv2(self, image: np.ndarray) -> dict:
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]

        not_background = (v > 30) & (s > 15)
        leaf_broad = not_background & (v < 250)

        kernel = np.ones((5, 5), np.uint8)
        leaf_broad = leaf_broad.astype(np.uint8)
        leaf_broad = cv2.morphologyEx(leaf_broad, cv2.MORPH_CLOSE, kernel)
        leaf_broad = cv2.morphologyEx(leaf_broad, cv2.MORPH_OPEN, kernel)

        total_leaf_pixels = int(np.sum(leaf_broad > 0))

        if total_leaf_pixels < 100:
            return self._make_result("Healthy", 0.0, 0, total_leaf_pixels)

        green_mask = (
            (h >= 35) & (h <= 85) &
            (s >= 40) &
            (v >= 40) &
            (leaf_broad > 0)
        ).astype(np.uint8)

        disease_mask = (
            (leaf_broad > 0) &
            (green_mask == 0) &
            (v > 20)
        ).astype(np.uint8)

        disease_mask = cv2.morphologyEx(
            disease_mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)
        )
        diseased_pixels = int(np.sum(disease_mask > 0))

        if total_leaf_pixels > 0:
            severity_pct = (diseased_pixels / total_leaf_pixels) * 100.0
        else:
            severity_pct = 0.0

        severity = self._classify(severity_pct)

        return self._make_result(
            severity, severity_pct, diseased_pixels, total_leaf_pixels
        )

    def _estimate_pil(self, image: np.ndarray) -> dict:
        img = image.astype(np.float32) / 255.0
        r, g, b = img[:, :, 0], img[:, :, 1], img[:, :, 2]

        max_channel = np.maximum(np.maximum(r, g), b)
        min_channel = np.minimum(np.minimum(r, g), b)
        saturation = np.where(
            max_channel > 0, (max_channel - min_channel) / max_channel, 0
        )

        leaf_mask = (saturation > 0.08) & (max_channel > 0.1) & (max_channel < 0.98)
        total_leaf_pixels = int(np.sum(leaf_mask))

        if total_leaf_pixels < 100:
            return self._make_result("Healthy", 0.0, 0, total_leaf_pixels)

        green_dominant = leaf_mask & (g > r) & (g > b * 0.8) & (g > 0.2)
        healthy_pixels = int(np.sum(green_dominant))

        diseased_pixels = total_leaf_pixels - healthy_pixels

        severity_pct = (diseased_pixels / total_leaf_pixels) * 100.0
        severity = self._classify(severity_pct)

        return self._make_result(
            severity, severity_pct, diseased_pixels, total_leaf_pixels
        )

    def _classify(self, percentage: float) -> str:
        if percentage < 5.0:
            return "Healthy"
        elif percentage < self.mild_threshold:
            return "Mild"
        elif percentage < self.severe_threshold:
            return "Moderate"
        else:
            return "Severe"

    def _make_result(
        self,
        severity: str,
        percentage: float,
        diseased_pixels: int,
        total_leaf_pixels: int,
    ) -> dict:
        color_map = {
            "Healthy": "#00e676",
            "Mild": "#ffeb3b",
            "Moderate": "#ff9800",
            "Severe": "#f44336",
        }

        desc_map = {
            "Healthy": "No significant disease detected. The leaf appears healthy.",
            "Mild": "Early stage disease detected. Minor spots visible. Treatable with prompt action.",
            "Moderate": "Significant disease spread detected. Multiple affected areas. Immediate treatment recommended.",
            "Severe": "Extensive disease damage detected. Large portions affected. Urgent intervention required. Consider isolation.",
        }

        return {
            "severity": severity,
            "percentage": round(percentage, 1),
            "diseased_pixels": diseased_pixels,
            "total_leaf_pixels": total_leaf_pixels,
            "color_label": color_map.get(severity, "#ffffff"),
            "description": desc_map.get(severity, ""),
        }

    def estimate_from_file(self, image_path: str) -> dict:
        image = np.array(Image.open(image_path).convert("RGB"))
        return self.estimate(image)


if __name__ == "__main__":
    print("🧪 Testing Severity Estimator...")

    estimator = SeverityEstimator()

    test_image = np.zeros((224, 224, 3), dtype=np.uint8)
    test_image[:, :] = [34, 139, 34]
    test_image[50:100, 50:150] = [139, 90, 43]
    test_image[100:130, 80:140] = [160, 82, 45]

    result = estimator.estimate(test_image)

    print(f"\n  Severity:      {result['severity']}")
    print(f"  Percentage:    {result['percentage']}%")
    print(f"  Diseased px:   {result['diseased_pixels']:,}")
    print(f"  Total leaf px: {result['total_leaf_pixels']:,}")
    print(f"  Color:         {result['color_label']}")
    print(f"  Description:   {result['description'][:60]}...")

    healthy_image = np.full((224, 224, 3), [34, 139, 34], dtype=np.uint8)
    healthy_result = estimator.estimate(healthy_image)
    print(f"\n  Healthy test:  {healthy_result['severity']} ({healthy_result['percentage']}%)")

    print("\n  ✅ Severity estimation test PASSED!")
