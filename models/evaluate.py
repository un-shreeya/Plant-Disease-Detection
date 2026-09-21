import argparse
import json
import os
import sys
import time
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.efficientnet_model import PlantDiseaseModel

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def load_model(model_path: str, device: torch.device) -> tuple:
    if not os.path.exists(model_path):
        print(f"❌ Model not found: {model_path}")
        print("   Train first: python models/train.py")
        sys.exit(1)

    checkpoint = torch.load(
        model_path, map_location=device, weights_only=False
    )

    num_classes = checkpoint.get("num_classes", 38)
    class_names = checkpoint.get(
        "class_names", [f"Class_{i}" for i in range(num_classes)]
    )

    model = PlantDiseaseModel(num_classes=num_classes, pretrained=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    return model, class_names, checkpoint


def evaluate_model(
    model,
    dataloader: DataLoader,
    device: torch.device,
    class_names: list,
    output_dir: str = "evaluation/outputs",
):
    os.makedirs(output_dir, exist_ok=True)

    model.eval()
    all_preds, all_labels, all_probs, inference_times = [], [], [], []

    print("\n  📊 Running evaluation on test set...")

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)

            start = time.time()
            outputs = model(images)
            inference_times.append(
                (time.time() - start) / images.size(0) * 1000
            )

            probs = F.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_probs.extend(probs.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)

    from sklearn.metrics import (
        accuracy_score,
        classification_report,
        f1_score,
        precision_score,
        recall_score,
    )

    accuracy = accuracy_score(all_labels, all_preds) * 100
    precision = (
        precision_score(all_labels, all_preds, average="weighted") * 100
    )
    recall = recall_score(all_labels, all_preds, average="weighted") * 100
    f1 = f1_score(all_labels, all_preds, average="weighted") * 100
    avg_inference_ms = np.mean(inference_times)

    model_size_mb = sum(
        p.numel() * p.element_size() for p in model.parameters()
    ) / (1024 * 1024)

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

    report = classification_report(
        all_labels, all_preds, target_names=class_names
    )
    report_path = os.path.join(output_dir, "classification_report.txt")
    with open(report_path, "w") as f:
        f.write("=" * 80 + "\n")
        f.write("  PLANT DISEASE CLASSIFICATION REPORT\n")
        f.write("=" * 80 + "\n\n")
        f.write(report)
    print(f"\n  📄 Classification report saved to: {report_path}")

    _plot_confusion_matrix(all_labels, all_preds, class_names, output_dir)
    _plot_per_class_accuracy(all_labels, all_preds, class_names, output_dir)
    _plot_roc_curves(all_labels, all_probs, class_names, output_dir)

    summary = {
        "accuracy": round(accuracy, 2),
        "precision": round(precision, 2),
        "recall": round(recall, 2),
        "f1_score": round(f1, 2),
        "inference_time_ms": round(avg_inference_ms, 1),
        "model_size_mb": round(model_size_mb, 1),
        "num_classes": len(class_names),
        "total_test_samples": len(all_labels),
    }
    summary_path = os.path.join(output_dir, "evaluation_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    return summary


def _plot_confusion_matrix(labels, preds, class_names, output_dir):
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import seaborn as sns
        from sklearn.metrics import confusion_matrix

        cm = confusion_matrix(labels, preds)
        cm_normalized = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

        short_names = [
            n.split("___")[-1][:18] if "___" in n else n[:18]
            for n in class_names
        ]

        fig, ax = plt.subplots(figsize=(20, 18))
        sns.heatmap(
            cm_normalized,
            annot=False,
            fmt=".1f",
            xticklabels=short_names,
            yticklabels=short_names,
            cmap="YlOrRd",
            ax=ax,
            vmin=0,
            vmax=1,
            cbar_kws={"label": "Prediction Rate"},
        )
        ax.set_xlabel("Predicted Label", fontsize=12)
        ax.set_ylabel("True Label", fontsize=12)
        ax.set_title(
            "Confusion Matrix (Normalized)", fontsize=14, fontweight="bold"
        )
        plt.xticks(rotation=45, ha="right", fontsize=7)
        plt.yticks(rotation=0, fontsize=7)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "confusion_matrix.png"), dpi=150)
        plt.close()
        print("  📊 Confusion matrix saved")
    except ImportError:
        print(" *matplotlib/seaborn not installed — skipping confusion matrix")


def _plot_per_class_accuracy(labels, preds, class_names, output_dir):
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        per_class_acc = []
        for i, name in enumerate(class_names):
            mask = labels == i
            acc = (preds[mask] == i).mean() * 100 if mask.sum() > 0 else 0
            per_class_acc.append(acc)

        sorted_indices = np.argsort(per_class_acc)
        sorted_names = [
            class_names[i].split("___")[-1][:25] for i in sorted_indices
        ]
        sorted_accs = [per_class_acc[i] for i in sorted_indices]

        colors = [
            "#f44336" if a < 80 else "#ff9800" if a < 90 else "#4caf50"
            for a in sorted_accs
        ]

        fig, ax = plt.subplots(
            figsize=(12, max(10, len(class_names) * 0.35))
        )
        bars = ax.barh(
            range(len(sorted_names)),
            sorted_accs,
            color=colors,
            edgecolor="white",
        )
        ax.set_yticks(range(len(sorted_names)))
        ax.set_yticklabels(sorted_names, fontsize=8)
        ax.set_xlabel("Accuracy (%)", fontsize=12)
        ax.set_title("Per-Class Accuracy", fontsize=14, fontweight="bold")
        ax.set_xlim(0, 105)
        ax.axvline(
            x=90, color="gray", linestyle="--", alpha=0.5, label="90% threshold"
        )

        for bar, acc in zip(bars, sorted_accs):
            ax.text(
                bar.get_width() + 0.5,
                bar.get_y() + bar.get_height() / 2,
                f"{acc:.1f}%",
                va="center",
                fontsize=7,
            )

        plt.tight_layout()
        plt.savefig(
            os.path.join(output_dir, "per_class_accuracy.png"), dpi=150
        )
        plt.close()
        print("  📊 Per-class accuracy chart saved")
    except ImportError:
        pass


def _plot_roc_curves(labels, probs, class_names, output_dir, top_k=10):
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from sklearn.metrics import auc, roc_curve
        from sklearn.preprocessing import label_binarize

        num_classes = len(class_names)
        labels_bin = label_binarize(labels, classes=range(num_classes))

        class_aucs = []
        for i in range(num_classes):
            if labels_bin[:, i].sum() > 0:
                fpr, tpr, _ = roc_curve(labels_bin[:, i], probs[:, i])
                class_aucs.append((auc(fpr, tpr), i))

        class_aucs.sort(reverse=True)
        top_classes = class_aucs[:top_k]

        fig, ax = plt.subplots(figsize=(10, 8))
        colors = plt.cm.Set3(np.linspace(0, 1, top_k))

        for (auc_val, class_idx), color in zip(top_classes, colors):
            fpr, tpr, _ = roc_curve(
                labels_bin[:, class_idx], probs[:, class_idx]
            )
            short_name = class_names[class_idx].split("___")[-1][:20]
            ax.plot(
                fpr,
                tpr,
                color=color,
                linewidth=2,
                label=f"{short_name} (AUC={auc_val:.3f})",
            )

        ax.plot([0, 1], [0, 1], "k--", alpha=0.3, label="Random (AUC=0.5)")
        ax.set_xlabel("False Positive Rate", fontsize=12)
        ax.set_ylabel("True Positive Rate", fontsize=12)
        ax.set_title(
            f"ROC Curves (Top {top_k} Classes)", fontsize=14, fontweight="bold"
        )
        ax.legend(loc="lower right", fontsize=9)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "roc_curves.png"), dpi=150)
        plt.close()

        macro_auc = np.mean([a for a, _ in class_aucs])
        print(f"  📊 ROC curves saved (Macro AUC: {macro_auc:.4f})")
    except ImportError:
        pass


def test_robustness(
    model,
    test_dir: str,
    device: torch.device,
    output_dir: str = "evaluation/outputs",
):
    from PIL import ImageFilter
    from torchvision import transforms as T

    print("\n  🔧 Running robustness tests...")

    class AddGaussianNoise:
        def __init__(self, std=0.1):
            self.std = std

        def __call__(self, tensor):
            return tensor + torch.randn_like(tensor) * self.std

    class AddBlur:
        def __init__(self, radius=2):
            self.radius = radius

        def __call__(self, img):
            return img.filter(ImageFilter.GaussianBlur(self.radius))

    class ReduceBrightness:
        def __init__(self, factor=0.5):
            self.factor = factor

        def __call__(self, img):
            from PIL import ImageEnhance

            return ImageEnhance.Brightness(img).enhance(self.factor)

    class LowResolution:
        def __init__(self, low_size=56):
            self.low_size = low_size

        def __call__(self, img):
            return img.resize((self.low_size, self.low_size)).resize(img.size)

    base_transform = T.Compose(
        [
            T.Resize((256, 256)),
            T.CenterCrop(224),
        ]
    )

    test_configs = {
        "clean": T.Compose(
            [
                base_transform,
                T.ToTensor(),
                T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ]
        ),
        "blurry": T.Compose(
            [
                base_transform,
                AddBlur(radius=3),
                T.ToTensor(),
                T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ]
        ),
        "dark": T.Compose(
            [
                base_transform,
                ReduceBrightness(factor=0.4),
                T.ToTensor(),
                T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ]
        ),
        "noisy": T.Compose(
            [
                base_transform,
                T.ToTensor(),
                AddGaussianNoise(std=0.1),
                T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ]
        ),
        "low_resolution": T.Compose(
            [
                base_transform,
                LowResolution(low_size=56),
                T.ToTensor(),
                T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ]
        ),
    }

    results = {}
    for test_name, transform in test_configs.items():
        dataset = datasets.ImageFolder(root=test_dir, transform=transform)
        loader = DataLoader(
            dataset, batch_size=32, shuffle=False, num_workers=0
        )

        correct, total = 0, 0
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

    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "robustness_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate trained plant disease model"
    )
    parser.add_argument(
        "--model_path", type=str, default="models/saved/best_model.pth"
    )
    parser.add_argument("--data_dir", type=str, default="data/processed")
    parser.add_argument(
        "--output_dir", type=str, default="evaluation/outputs"
    )
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("\n🌿 Plant Disease Model — Evaluation")
    print(f"   Device: {device}")

    model, class_names, _ = load_model(args.model_path, device)

    test_dir = os.path.join(args.data_dir, "test")
    if not os.path.isdir(test_dir):
        print(f"❌ Test data not found: {test_dir}")
        sys.exit(1)

    test_transform = transforms.Compose(
        [
            transforms.Resize((256, 256)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )

    test_dataset = datasets.ImageFolder(
        root=test_dir, transform=test_transform
    )
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0
    )

    print(f"   Test samples: {len(test_dataset):,}")

    summary = evaluate_model(
        model, test_loader, device, class_names, args.output_dir
    )
    robustness = test_robustness(model, test_dir, device, args.output_dir)

    print("\n  ✅ Evaluation complete!")
    print(f"  Results saved to: {args.output_dir}/")
