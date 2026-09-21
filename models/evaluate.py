"""
=============================================================================
 Model Evaluation & Metrics Module
 File: models/evaluate.py
=============================================================================
 PURPOSE:
   Comprehensive evaluation of the trained plant disease model, generating
   all metrics and visualizations needed for the research paper.

 GENERATES:
   evaluation/outputs/
     ├── confusion_matrix.png       # 38×38 heatmap
     ├── classification_report.txt  # Per-class precision/recall/F1
     ├── roc_curves.png             # One-vs-Rest ROC curves
     ├── per_class_accuracy.png     # Bar chart of per-class accuracy
     ├── sample_predictions.png     # Grid of sample predictions
     ├── robustness_results.json    # Blur/dark/noise test results
     └── evaluation_summary.json    # Overall metrics summary

 METRICS COMPUTED:
   - Overall Accuracy
   - Weighted Precision, Recall, F1-Score
   - Per-class Precision, Recall, F1-Score
   - Confusion Matrix
   - ROC-AUC (One-vs-Rest)
   - Inference Time (ms per image)
   - Model Size (MB)
   - Robustness Tests (blur, dark, noise, resolution)

 USAGE:
   python models/evaluate.py
   python models/evaluate.py --model_path models/saved/best_model.pth
=============================================================================
"""

import os
import sys
import json
import time
import argparse
import numpy as np

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.efficientnet_model import PlantDiseaseModel


# ImageNet normalization constants
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def load_model(model_path: str, device: torch.device) -> tuple:
    """
    Loads a trained model from checkpoint.

    PARAMETERS:
      model_path: Path to saved .pth checkpoint
      device: CPU or CUDA device

    RETURNS:
      tuple: (model, class_names, checkpoint_info)
    """
    if not os.path.exists(model_path):
        print(f"❌ Model not found: {model_path}")
        print("   Train first: python models/train.py")
        sys.exit(1)

    checkpoint = torch.load(model_path, map_location=device, weights_only=False)

    num_classes = checkpoint.get('num_classes', 38)
    class_names = checkpoint.get('class_names', [f'Class_{i}' for i in range(num_classes)])

    model = PlantDiseaseModel(num_classes=num_classes, pretrained=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    return model, class_names, checkpoint


def evaluate_model(
    model,
    dataloader: DataLoader,
    device: torch.device,
    class_names: list,
    output_dir: str = 'evaluation/outputs',
):
    """
    Complete model evaluation — computes all metrics and generates plots.

    METRICS:
      Accuracy = (TP + TN) / (TP + TN + FP + FN)

      Precision = TP / (TP + FP)
        "Of all predicted positives, how many were actually positive?"

      Recall (Sensitivity) = TP / (TP + FN)
        "Of all actual positives, how many were correctly predicted?"

      F1-Score = 2 × (Precision × Recall) / (Precision + Recall)
        Harmonic mean of precision and recall. Balanced metric.

      ROC-AUC:
        Area under the Receiver Operating Characteristic curve.
        Measures how well the model distinguishes between classes.
        AUC = 1.0 → perfect, AUC = 0.5 → random guessing.
    """
    os.makedirs(output_dir, exist_ok=True)

    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    inference_times = []

    print("\n  📊 Running evaluation on test set...")

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)

            # Measure inference time
            start = time.time()
            outputs = model(images)
            inference_times.append((time.time() - start) / images.size(0) * 1000)

            probs = F.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_probs.extend(probs.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)

    # ── Compute Metrics ────────────────────────────────────────────────────
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score, f1_score,
        classification_report, confusion_matrix
    )

    accuracy = accuracy_score(all_labels, all_preds) * 100
    precision = precision_score(all_labels, all_preds, average='weighted') * 100
    recall = recall_score(all_labels, all_preds, average='weighted') * 100
    f1 = f1_score(all_labels, all_preds, average='weighted') * 100
    avg_inference_ms = np.mean(inference_times)

    # Model size
    model_size_mb = sum(p.numel() * p.element_size() for p in model.parameters()) / (1024 * 1024)

    # ── Print Results ──────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  📊 EVALUATION RESULTS")
    print("=" * 60)
    print(f"  Accuracy:       {accuracy:.2f}%")
    print(f"  Precision:      {precision:.2f}%")
    print(f"  Recall:         {recall:.2f}%")
    print(f"  F1-Score:       {f1:.2f}%")
    print(f"  Inference Time: {avg_inference_ms:.1f} ms/image")
    print(f"  Model Size:     {model_size_mb:.1f} MB")
    print("=" * 60)

    # ── Classification Report ──────────────────────────────────────────────
    report = classification_report(all_labels, all_preds, target_names=class_names)
    report_path = os.path.join(output_dir, 'classification_report.txt')
    with open(report_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("  PLANT DISEASE CLASSIFICATION REPORT\n")
        f.write("=" * 80 + "\n\n")
        f.write(report)
    print(f"\n  📄 Classification report saved to: {report_path}")

    # ── Generate Visualizations ────────────────────────────────────────────
    _plot_confusion_matrix(all_labels, all_preds, class_names, output_dir)
    _plot_per_class_accuracy(all_labels, all_preds, class_names, output_dir)
    _plot_roc_curves(all_labels, all_probs, class_names, output_dir)

    # ── Save Summary JSON ──────────────────────────────────────────────────
    summary = {
        'accuracy': round(accuracy, 2),
        'precision': round(precision, 2),
        'recall': round(recall, 2),
        'f1_score': round(f1, 2),
        'inference_time_ms': round(avg_inference_ms, 1),
        'model_size_mb': round(model_size_mb, 1),
        'num_classes': len(class_names),
        'total_test_samples': len(all_labels),
    }
    summary_path = os.path.join(output_dir, 'evaluation_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

    return summary


def _plot_confusion_matrix(labels, preds, class_names, output_dir):
    """
    Generates a 38×38 confusion matrix heatmap.

    The confusion matrix C where C[i,j] = number of samples with
    true label i predicted as class j.

    Diagonal elements = correct predictions
    Off-diagonal elements = misclassifications

    Perfect model → diagonal matrix (all off-diagonal elements = 0)
    """
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import seaborn as sns
        from sklearn.metrics import confusion_matrix

        cm = confusion_matrix(labels, preds)
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        # Short names for display (crop___disease → disease)
        short_names = [n.split('___')[-1][:18] if '___' in n else n[:18] for n in class_names]

        fig, ax = plt.subplots(figsize=(20, 18))
        sns.heatmap(
            cm_normalized, annot=False, fmt='.1f',
            xticklabels=short_names, yticklabels=short_names,
            cmap='YlOrRd', ax=ax, vmin=0, vmax=1,
            cbar_kws={'label': 'Prediction Rate'}
        )
        ax.set_xlabel('Predicted Label', fontsize=12)
        ax.set_ylabel('True Label', fontsize=12)
        ax.set_title('Confusion Matrix (Normalized)', fontsize=14, fontweight='bold')
        plt.xticks(rotation=45, ha='right', fontsize=7)
        plt.yticks(rotation=0, fontsize=7)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'), dpi=150)
        plt.close()
        print(f"  📊 Confusion matrix saved")
    except ImportError:
        print("  ⚠️  matplotlib/seaborn not installed — skipping confusion matrix")


def _plot_per_class_accuracy(labels, preds, class_names, output_dir):
    """
    Generates a horizontal bar chart showing accuracy per disease class.
    Useful for identifying which diseases the model struggles with.
    """
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        # Calculate per-class accuracy
        per_class_acc = []
        for i, name in enumerate(class_names):
            mask = labels == i
            if mask.sum() > 0:
                acc = (preds[mask] == i).mean() * 100
            else:
                acc = 0
            per_class_acc.append(acc)

        # Sort by accuracy
        sorted_indices = np.argsort(per_class_acc)
        sorted_names = [class_names[i].split('___')[-1][:25] for i in sorted_indices]
        sorted_accs = [per_class_acc[i] for i in sorted_indices]

        # Color by accuracy level
        colors = ['#f44336' if a < 80 else '#ff9800' if a < 90 else '#4caf50' for a in sorted_accs]

        fig, ax = plt.subplots(figsize=(12, max(10, len(class_names) * 0.35)))
        bars = ax.barh(range(len(sorted_names)), sorted_accs, color=colors, edgecolor='white')
        ax.set_yticks(range(len(sorted_names)))
        ax.set_yticklabels(sorted_names, fontsize=8)
        ax.set_xlabel('Accuracy (%)', fontsize=12)
        ax.set_title('Per-Class Accuracy', fontsize=14, fontweight='bold')
        ax.set_xlim(0, 105)
        ax.axvline(x=90, color='gray', linestyle='--', alpha=0.5, label='90% threshold')

        # Add value labels on bars
        for bar, acc in zip(bars, sorted_accs):
            ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
                   f'{acc:.1f}%', va='center', fontsize=7)

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'per_class_accuracy.png'), dpi=150)
        plt.close()
        print(f"  📊 Per-class accuracy chart saved")
    except ImportError:
        pass


def _plot_roc_curves(labels, probs, class_names, output_dir, top_k=10):
    """
    Generates ROC curves (One-vs-Rest) for the top-k classes.

    ROC CURVE:
      Plots True Positive Rate (Recall) vs False Positive Rate:
        TPR = TP / (TP + FN)
        FPR = FP / (FP + TN)

      Each point on the curve represents a different classification threshold.

    AUC (Area Under Curve):
      AUC = ∫₀¹ TPR(FPR) dFPR

      AUC = 1.0 → perfect classifier
      AUC = 0.5 → random guessing (diagonal line)
    """
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from sklearn.metrics import roc_curve, auc
        from sklearn.preprocessing import label_binarize

        num_classes = len(class_names)
        labels_bin = label_binarize(labels, classes=range(num_classes))

        # Calculate AUC for each class
        class_aucs = []
        for i in range(num_classes):
            if labels_bin[:, i].sum() > 0:
                fpr, tpr, _ = roc_curve(labels_bin[:, i], probs[:, i])
                class_aucs.append((auc(fpr, tpr), i))

        # Sort by AUC and take top-k for display
        class_aucs.sort(reverse=True)
        top_classes = class_aucs[:top_k]

        fig, ax = plt.subplots(figsize=(10, 8))

        colors = plt.cm.Set3(np.linspace(0, 1, top_k))
        for (auc_val, class_idx), color in zip(top_classes, colors):
            fpr, tpr, _ = roc_curve(labels_bin[:, class_idx], probs[:, class_idx])
            short_name = class_names[class_idx].split('___')[-1][:20]
            ax.plot(fpr, tpr, color=color, linewidth=2,
                   label=f'{short_name} (AUC={auc_val:.3f})')

        # Random classifier baseline
        ax.plot([0, 1], [0, 1], 'k--', alpha=0.3, label='Random (AUC=0.5)')

        ax.set_xlabel('False Positive Rate', fontsize=12)
        ax.set_ylabel('True Positive Rate', fontsize=12)
        ax.set_title(f'ROC Curves (Top {top_k} Classes)', fontsize=14, fontweight='bold')
        ax.legend(loc='lower right', fontsize=9)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'roc_curves.png'), dpi=150)
        plt.close()

        # Calculate macro-average AUC
        macro_auc = np.mean([a for a, _ in class_aucs])
        print(f"  📊 ROC curves saved (Macro AUC: {macro_auc:.4f})")
    except ImportError:
        pass


def test_robustness(
    model,
    test_dir: str,
    device: torch.device,
    output_dir: str = 'evaluation/outputs',
):
    """
    Tests model robustness against real-world image degradation.

    TESTS:
      1. Clean images (baseline)
      2. Gaussian blur (σ=2) — simulates out-of-focus camera
      3. Low brightness (-50%) — simulates low light / shadows
      4. Gaussian noise (σ=25) — simulates cheap camera sensor noise
      5. Low resolution (56×56 upscaled to 224) — simulates distant photos

    Each degradation simulates a realistic scenario a farmer might face
    when taking photos in the field.

    RESULTS saved to evaluation/outputs/robustness_results.json
    """
    from torchvision import transforms as T
    from PIL import ImageFilter

    print("\n  🔧 Running robustness tests...")

    class AddGaussianNoise:
        """Custom transform that adds Gaussian noise to an image tensor."""
        def __init__(self, std=0.1):
            self.std = std
        def __call__(self, tensor):
            return tensor + torch.randn_like(tensor) * self.std

    class AddBlur:
        """Custom transform that applies Gaussian blur to a PIL image."""
        def __init__(self, radius=2):
            self.radius = radius
        def __call__(self, img):
            return img.filter(ImageFilter.GaussianBlur(self.radius))

    class ReduceBrightness:
        """Custom transform that reduces image brightness."""
        def __init__(self, factor=0.5):
            self.factor = factor
        def __call__(self, img):
            from PIL import ImageEnhance
            enhancer = ImageEnhance.Brightness(img)
            return enhancer.enhance(self.factor)

    class LowResolution:
        """Downscale then upscale to simulate low-resolution capture."""
        def __init__(self, low_size=56):
            self.low_size = low_size
        def __call__(self, img):
            return img.resize((self.low_size, self.low_size)).resize(img.size)

    base_transform = T.Compose([
        T.Resize((256, 256)),
        T.CenterCrop(224),
    ])

    test_configs = {
        'clean': T.Compose([
            base_transform, T.ToTensor(),
            T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]),
        'blurry': T.Compose([
            base_transform, AddBlur(radius=3), T.ToTensor(),
            T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]),
        'dark': T.Compose([
            base_transform, ReduceBrightness(factor=0.4), T.ToTensor(),
            T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]),
        'noisy': T.Compose([
            base_transform, T.ToTensor(), AddGaussianNoise(std=0.1),
            T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]),
        'low_resolution': T.Compose([
            base_transform, LowResolution(low_size=56), T.ToTensor(),
            T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]),
    }

    results = {}
    for test_name, transform in test_configs.items():
        dataset = datasets.ImageFolder(root=test_dir, transform=transform)
        loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=0)

        correct = 0
        total = 0
        model.eval()
        with torch.no_grad():
            for images, labels in loader:
                images = images.to(device)
                outputs = model(images)
                _, preds = torch.max(outputs, 1)
                correct += (preds.cpu() == labels).sum().item()
                total += labels.size(0)

        acc = (correct / total * 100) if total > 0 else 0
        results[test_name] = round(acc, 2)
        print(f"     {test_name:18s}: {acc:.2f}%")

    # Save results
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, 'robustness_results.json'), 'w') as f:
        json.dump(results, f, indent=2)

    return results


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate trained plant disease model")
    parser.add_argument('--model_path', type=str, default='models/saved/best_model.pth')
    parser.add_argument('--data_dir', type=str, default='data/processed')
    parser.add_argument('--output_dir', type=str, default='evaluation/outputs')
    parser.add_argument('--batch_size', type=int, default=32)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n🌿 Plant Disease Model — Evaluation")
    print(f"   Device: {device}")

    # Load model
    model, class_names, _ = load_model(args.model_path, device)

    # Load test dataset
    test_dir = os.path.join(args.data_dir, 'test')
    if not os.path.isdir(test_dir):
        print(f"❌ Test data not found: {test_dir}")
        sys.exit(1)

    test_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    test_dataset = datasets.ImageFolder(root=test_dir, transform=test_transform)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    print(f"   Test samples: {len(test_dataset):,}")

    # Run evaluation
    summary = evaluate_model(model, test_loader, device, class_names, args.output_dir)

    # Run robustness tests
    robustness = test_robustness(model, test_dir, device, args.output_dir)

    print("\n  ✅ Evaluation complete!")
    print(f"  Results saved to: {args.output_dir}/")
