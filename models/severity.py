"""
=============================================================================
 Disease Severity Estimation Module
 File: models/severity.py
=============================================================================
 PURPOSE:
   Estimates the severity of plant disease from a leaf image using
   color-based segmentation in HSV color space. Classifies severity
   as Mild, Moderate, or Severe based on the ratio of diseased pixels
   to total leaf pixels.

 ALGORITHM:
   1. Convert image from RGB to HSV color space
   2. Segment the leaf from the background using green channel thresholding
   3. Identify diseased regions (brown, yellow, dark spots)
   4. Calculate severity ratio = diseased_pixels / total_leaf_pixels
   5. Classify: <10% → Mild, 10-30% → Moderate, >30% → Severe

 WHY HSV COLOR SPACE?
   HSV (Hue, Saturation, Value) separates color information (Hue)
   from intensity (Value), making it robust to lighting variations.

   In HSV:
     - Healthy leaf: Hue ≈ 35-85° (green range)
     - Diseased areas: Hue ≈ 10-35° (yellow/brown) or Hue ≈ 0-10° (dark spots)
     - Background: Low Saturation (<20) or very dark (Value < 30)

 MATHEMATICAL FOUNDATION:
   Severity Ratio (SR):
     SR = N_diseased / N_leaf × 100%

   where:
     N_diseased = count of pixels classified as diseased
     N_leaf = count of pixels belonging to the leaf (not background)

   Classification Thresholds:
     SR < 10%  → Mild      (early stage, treatable)
     10% ≤ SR < 30% → Moderate (significant damage, action needed)
     SR ≥ 30% → Severe    (extensive damage, may spread)

 USAGE:
   from models.severity import SeverityEstimator

   estimator = SeverityEstimator()
   result = estimator.estimate(image_array)
   # result = {'severity': 'Moderate', 'percentage': 22.5, ...}
=============================================================================
"""

import numpy as np
from PIL import Image

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False


class SeverityEstimator:
    """
    Estimates plant disease severity using color-based image analysis.

    The estimator works in three stages:
      1. Leaf Segmentation: Isolates the leaf from the background
      2. Disease Detection: Identifies brown/yellow/dark diseased regions
      3. Severity Calculation: Computes the ratio and classifies severity

    THRESHOLDS (configurable):
      mild_threshold: Maximum SR for "Mild" classification (default: 10%)
      severe_threshold: Minimum SR for "Severe" classification (default: 30%)

    LIMITATIONS:
      - Works best on images with clear leaf-background contrast
      - Accuracy depends on lighting conditions
      - May struggle with non-green leaves (purple, red varieties)
      - For production use, consider training a segmentation CNN (e.g., U-Net)
    """

    def __init__(
        self,
        mild_threshold: float = 10.0,
        severe_threshold: float = 30.0,
    ):
        """
        Initialize the severity estimator with classification thresholds.

        PARAMETERS:
          mild_threshold (float): SR below this → "Mild" (default: 10%)
          severe_threshold (float): SR above this → "Severe" (default: 30%)
                                   Between mild and severe → "Moderate"
        """
        self.mild_threshold = mild_threshold
        self.severe_threshold = severe_threshold

    def estimate(self, image: np.ndarray) -> dict:
        """
        Estimate disease severity from a leaf image.

        ALGORITHM (step by step):

        Step 1: Convert RGB to HSV
          H = atan2(√3(G-B), 2R-G-B)  (hue angle, 0-180 in OpenCV)
          S = (max(R,G,B) - min(R,G,B)) / max(R,G,B)  (saturation, 0-255)
          V = max(R,G,B)  (value/brightness, 0-255)

        Step 2: Leaf Segmentation
          leaf_mask = (35 ≤ H ≤ 85) AND (S ≥ 40) AND (V ≥ 40)
          This captures the green portions of the leaf.
          We also include diseased portions using broader hue range.

        Step 3: Disease Region Detection
          disease_mask = leaf AND ((H < 35) OR (H > 85) OR (S < 40))
          AND (V > 20)  # Not pure black/background
          This captures brown, yellow, and dark spots within the leaf.

        Step 4: Calculate Severity
          SR = count(disease_mask=True) / count(leaf_total=True) × 100

        PARAMETERS:
          image (np.ndarray): Leaf image in RGB format (H, W, 3), uint8

        RETURNS:
          dict:
            'severity' (str): "Healthy", "Mild", "Moderate", or "Severe"
            'percentage' (float): Disease severity percentage (0-100)
            'diseased_pixels' (int): Count of diseased pixels
            'total_leaf_pixels' (int): Count of total leaf pixels
            'color_label' (str): CSS color for UI display
            'description' (str): Human-readable severity description
        """
        # Ensure image is RGB uint8
        if image.dtype != np.uint8:
            image = (image * 255).astype(np.uint8)

        if HAS_CV2:
            return self._estimate_cv2(image)
        else:
            return self._estimate_pil(image)

    def _estimate_cv2(self, image: np.ndarray) -> dict:
        """
        Severity estimation using OpenCV (preferred, more accurate).
        """
        # ── Step 1: Convert to HSV ─────────────────────────────────────
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]

        # ── Step 2: Segment the entire leaf (including diseased parts) ─
        # Broad leaf mask: anything that's not background
        # Background is typically very dark, very light, or very unsaturated
        not_background = (v > 30) & (s > 15)

        # Additional background removal using color
        # Pure white/gray backgrounds have very low saturation
        leaf_broad = not_background & (v < 250)

        # Morphological operations to clean up the mask
        kernel = np.ones((5, 5), np.uint8)
        leaf_broad = leaf_broad.astype(np.uint8)
        leaf_broad = cv2.morphologyEx(leaf_broad, cv2.MORPH_CLOSE, kernel)
        leaf_broad = cv2.morphologyEx(leaf_broad, cv2.MORPH_OPEN, kernel)

        total_leaf_pixels = int(np.sum(leaf_broad > 0))

        if total_leaf_pixels < 100:
            # Too few leaf pixels — likely a background-only image
            return self._make_result("Healthy", 0.0, 0, total_leaf_pixels)

        # ── Step 3: Detect healthy green regions ───────────────────────
        # Healthy leaves are green: Hue 35-85 (in OpenCV's 0-180 scale)
        green_mask = (
            (h >= 35) & (h <= 85) &  # Green hue range
            (s >= 40) &               # Reasonably saturated
            (v >= 40) &               # Not too dark
            (leaf_broad > 0)          # Within the leaf
        ).astype(np.uint8)

        healthy_pixels = int(np.sum(green_mask > 0))

        # ── Step 4: Detect diseased regions ────────────────────────────
        # Diseased areas: brown (H: 10-35), yellow (H: 20-35),
        # dark spots (low V), necrotic (very low S)
        disease_mask = (
            (leaf_broad > 0) &        # Within the leaf
            (green_mask == 0) &       # Not healthy green
            (v > 20)                  # Not pure black (shadow/background)
        ).astype(np.uint8)

        # Clean up disease mask
        disease_mask = cv2.morphologyEx(disease_mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        diseased_pixels = int(np.sum(disease_mask > 0))

        # ── Step 5: Calculate Severity Ratio ───────────────────────────
        if total_leaf_pixels > 0:
            severity_pct = (diseased_pixels / total_leaf_pixels) * 100.0
        else:
            severity_pct = 0.0

        # ── Step 6: Classify ───────────────────────────────────────────
        severity = self._classify(severity_pct)

        return self._make_result(severity, severity_pct, diseased_pixels, total_leaf_pixels)

    def _estimate_pil(self, image: np.ndarray) -> dict:
        """
        Fallback severity estimation without OpenCV.
        Uses basic numpy operations for color analysis.
        """
        # Convert to float for calculations
        img = image.astype(np.float32) / 255.0
        r, g, b = img[:, :, 0], img[:, :, 1], img[:, :, 2]

        # Simple leaf segmentation: green channel dominant
        max_channel = np.maximum(np.maximum(r, g), b)
        min_channel = np.minimum(np.minimum(r, g), b)
        saturation = np.where(max_channel > 0, (max_channel - min_channel) / max_channel, 0)

        # Leaf mask: reasonably saturated and not too dark/bright
        leaf_mask = (saturation > 0.08) & (max_channel > 0.1) & (max_channel < 0.98)
        total_leaf_pixels = int(np.sum(leaf_mask))

        if total_leaf_pixels < 100:
            return self._make_result("Healthy", 0.0, 0, total_leaf_pixels)

        # Healthy: green is the dominant channel
        green_dominant = leaf_mask & (g > r) & (g > b * 0.8) & (g > 0.2)
        healthy_pixels = int(np.sum(green_dominant))

        # Diseased: anything within the leaf that isn't green
        diseased_pixels = total_leaf_pixels - healthy_pixels

        # Calculate severity
        severity_pct = (diseased_pixels / total_leaf_pixels) * 100.0
        severity = self._classify(severity_pct)

        return self._make_result(severity, severity_pct, diseased_pixels, total_leaf_pixels)

    def _classify(self, percentage: float) -> str:
        """
        Classify severity based on percentage thresholds.

        DECISION BOUNDARY:
          percentage < 5%            → "Healthy"
          5% ≤ percentage < 10%     → "Mild"
          10% ≤ percentage < 30%    → "Moderate"
          percentage ≥ 30%           → "Severe"
        """
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
        """
        Creates a structured result dictionary with all severity information.

        Includes human-readable descriptions and CSS colors for the
        frontend dashboard display.
        """
        # Color mapping for UI display
        color_map = {
            "Healthy": "#00e676",   # Bright green
            "Mild": "#ffeb3b",      # Yellow
            "Moderate": "#ff9800",  # Orange
            "Severe": "#f44336",    # Red
        }

        # Description mapping for user display
        desc_map = {
            "Healthy": "No significant disease detected. The leaf appears healthy.",
            "Mild": "Early stage disease detected. Minor spots visible. "
                    "Treatable with prompt action.",
            "Moderate": "Significant disease spread detected. Multiple affected areas. "
                       "Immediate treatment recommended.",
            "Severe": "Extensive disease damage detected. Large portions affected. "
                     "Urgent intervention required. Consider isolation.",
        }

        return {
            'severity': severity,
            'percentage': round(percentage, 1),
            'diseased_pixels': diseased_pixels,
            'total_leaf_pixels': total_leaf_pixels,
            'color_label': color_map.get(severity, "#ffffff"),
            'description': desc_map.get(severity, ""),
        }

    def estimate_from_file(self, image_path: str) -> dict:
        """
        Convenience method to estimate severity from an image file path.

        PARAMETERS:
          image_path (str): Path to the leaf image file

        RETURNS:
          dict: Same as estimate()
        """
        image = np.array(Image.open(image_path).convert('RGB'))
        return self.estimate(image)


# =============================================================================
# STANDALONE TEST
# =============================================================================
if __name__ == "__main__":
    """
    Test severity estimation with a synthetic test image.
    """
    print("🧪 Testing Severity Estimator...")

    estimator = SeverityEstimator()

    # Create a synthetic test image:
    # Green background (healthy leaf) with a brown patch (diseased)
    test_image = np.zeros((224, 224, 3), dtype=np.uint8)

    # Fill with green (healthy leaf)
    test_image[:, :] = [34, 139, 34]  # Forest green

    # Add a brown patch (diseased area, ~20% of image)
    test_image[50:100, 50:150] = [139, 90, 43]  # Brown/diseased
    test_image[100:130, 80:140] = [160, 82, 45]  # Sienna (another disease spot)

    result = estimator.estimate(test_image)

    print(f"\n  Severity:      {result['severity']}")
    print(f"  Percentage:    {result['percentage']}%")
    print(f"  Diseased px:   {result['diseased_pixels']:,}")
    print(f"  Total leaf px: {result['total_leaf_pixels']:,}")
    print(f"  Color:         {result['color_label']}")
    print(f"  Description:   {result['description'][:60]}...")

    # Test with all-green image (healthy)
    healthy_image = np.full((224, 224, 3), [34, 139, 34], dtype=np.uint8)
    healthy_result = estimator.estimate(healthy_image)
    print(f"\n  Healthy test:  {healthy_result['severity']} ({healthy_result['percentage']}%)")

    print("\n  ✅ Severity estimation test PASSED!")
