import argparse
import json
import os
import sys
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.efficientnet_model import create_model

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_train_transforms(image_size: int = 224) -> transforms.Compose:
    """Returns training data augmentation transforms."""
    return transforms.Compose([
        transforms.Resize((image_size + 32, image_size + 32)),
        transforms.RandomCrop(image_size),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.3),
        transforms.RandomRotation(degrees=15),
        transforms.RandomAffine(
            degrees=0,
            translate=(0.1, 0.1),
            scale=(0.9, 1.1),
        ),
        transforms.RandomPerspective(distortion_scale=0.2, p=0.3),
        transforms.ColorJitter(
            brightness=0.3,
            contrast=0.3,
            saturation=0.2,
            hue=0.1,
        ),
        transforms.RandomApply([
            transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5.0))
        ], p=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def get_val_transforms(image_size: int = 224) -> transforms.Compose:
    """Returns validation data transforms."""
    return transforms.Compose([
        transforms.Resize((image_size + 32, image_size + 32)),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


class EarlyStopping:
    """Stops training when validation loss stops improving."""

    def __init__(self, patience: int = 7, min_delta: float = 0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss: float) -> bool:
        if self.best_loss is None:
            self.best_loss = val_loss
            return False

        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            return False
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                return True
            return False


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    scaler: GradScaler = None,
    use_amp: bool = False,
) -> tuple:
    """Trains the model for one epoch."""
    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(dataloader, desc="   Training", leave=False)

    for images, labels in pbar:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        if use_amp and scaler is not None:
            with autocast():
                outputs = model(images)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{100.0 * correct / total:.1f}%'
        })

    epoch_loss = running_loss / total
    epoch_acc = 100.0 * correct / total

    return epoch_loss, epoch_acc


def validate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple:
    """Evaluates the model on validation data."""
    model.eval()

    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        pbar = tqdm(dataloader, desc="   Validation", leave=False)

        for images, labels in pbar:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    epoch_loss = running_loss / total
    epoch_acc = 100.0 * correct / total

    return epoch_loss, epoch_acc


def save_training_curves(history: dict, output_dir: str):
    """Plots and saves training/validation loss and accuracy curves."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import seaborn as sns

        sns.set_theme(style="darkgrid")
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        epochs = range(1, len(history['train_loss']) + 1)

        ax1.plot(epochs, history['train_loss'], 'b-', linewidth=2, label='Train Loss')
        ax1.plot(epochs, history['val_loss'], 'r-', linewidth=2, label='Val Loss')
        ax1.set_xlabel('Epoch', fontsize=12)
        ax1.set_ylabel('Loss', fontsize=12)
        ax1.set_title('Training & Validation Loss', fontsize=14, fontweight='bold')
        ax1.legend(fontsize=11)
        ax1.grid(True, alpha=0.3)

        if 'freeze_epochs' in history:
            ax1.axvline(x=history['freeze_epochs'], color='green',
                       linestyle='--', alpha=0.7, label='Backbone Unfrozen')

        ax2.plot(epochs, history['train_acc'], 'b-', linewidth=2, label='Train Acc')
        ax2.plot(epochs, history['val_acc'], 'r-', linewidth=2, label='Val Acc')
        ax2.set_xlabel('Epoch', fontsize=12)
        ax2.set_ylabel('Accuracy (%)', fontsize=12)
        ax2.set_title('Training & Validation Accuracy', fontsize=14, fontweight='bold')
        ax2.legend(fontsize=11)
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(os.path.join(output_dir, 'training_curves.png'), dpi=150, bbox_inches='tight')
        plt.close()
        print(f"   📈 Training curves saved to {output_dir}/training_curves.png")
    except ImportError:
        print("   ⚠️ matplotlib not installed — skipping curve plots")


def train(args):
    """Main training function."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    use_amp = torch.cuda.is_available()

    print("\n" + "=" * 60)
    print("   🌿 PLANT DISEASE MODEL — TRAINING PIPELINE")
    print("=" * 60)
    print(f"   Device:       {device}")
    print(f"   Mixed Prec:   {'✅ Enabled' if use_amp else '❌ Disabled (CPU)'}")
    print(f"   Epochs:       {args.epochs}")
    print(f"   Batch Size:   {args.batch_size}")
    print(f"   Learning Rate:{args.lr}")
    print(f"   Freeze Epochs:{args.freeze_epochs}")
    print(f"   Data Dir:     {args.data_dir}")
    print("=" * 60)

    train_dir = os.path.join(args.data_dir, 'train')
    val_dir = os.path.join(args.data_dir, 'val')

    if not os.path.isdir(train_dir):
        print(f"\n❌ Training data not found at: {train_dir}")
        print("   Run first: python data/download_dataset.py")
        sys.exit(1)

    train_dataset = datasets.ImageFolder(
        root=train_dir,
        transform=get_train_transforms(args.image_size)
    )
    val_dataset = datasets.ImageFolder(
        root=val_dir,
        transform=get_val_transforms(args.image_size)
    )

    num_classes = len(train_dataset.classes)
    print(f"\n   📂 Dataset loaded:")
    print(f"      Train:    {len(train_dataset):,} images")
    print(f"      Val:      {len(val_dataset):,} images")
    print(f"      Classes:  {num_classes}")

    num_workers = 0 if sys.platform == 'win32' else 4
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    model = create_model(num_classes=num_classes, pretrained=True)
    model = model.to(device)

    class_counts = np.zeros(num_classes)
    for _, label in train_dataset.samples:
        class_counts[label] += 1

    class_weights = np.sum(class_counts) / (num_classes * class_counts + 1e-6)
    class_weights = torch.FloatTensor(class_weights).to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights)

    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': [],
        'lr': [], 'freeze_epochs': args.freeze_epochs,
        'best_val_acc': 0.0, 'best_epoch': 0,
    }

    early_stopping = EarlyStopping(patience=args.patience)
    scaler = GradScaler() if use_amp else None

    os.makedirs(args.save_dir, exist_ok=True)

    best_val_acc = 0.0
    start_time = time.time()

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()

        if epoch <= args.freeze_epochs:
            if epoch == 1:
                model.freeze_backbone()
                optimizer = optim.Adam(
                    filter(lambda p: p.requires_grad, model.parameters()),
                    lr=args.lr,
                    weight_decay=1e-4,
                )
                scheduler = optim.lr_scheduler.CosineAnnealingLR(
                    optimizer,
                    T_max=args.freeze_epochs,
                    eta_min=1e-6,
                )
                print(f"\n   ═══ PHASE 1: Training Classifier Head (Backbone Frozen) ═══")
        else:
            if epoch == args.freeze_epochs + 1:
                model.unfreeze_backbone()
                optimizer = optim.Adam(
                    model.parameters(),
                    lr=args.lr * 0.1,
                    weight_decay=1e-4,
                )
                scheduler = optim.lr_scheduler.CosineAnnealingLR(
                    optimizer,
                    T_max=args.epochs - args.freeze_epochs,
                    eta_min=1e-7,
                )
                print(f"\n   ═══ PHASE 2: Fine-tuning Entire Model ═══")

        current_lr = optimizer.param_groups[0]['lr']
        print(f"\n   Epoch {epoch}/{args.epochs} [LR: {current_lr:.2e}]")

        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device, scaler, use_amp
        )

        val_loss, val_acc = validate(model, val_loader, criterion, device)
        scheduler.step()

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['lr'].append(current_lr)

        epoch_time = time.time() - epoch_start

        print(f"    Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"    Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.2f}%")
        print(f"    Time:       {epoch_time:.1f}s")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            history['best_val_acc'] = best_val_acc
            history['best_epoch'] = epoch
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'val_loss': val_loss,
                'num_classes': num_classes,
                'class_names': train_dataset.classes,
            }, os.path.join(args.save_dir, 'best_model.pth'))
            print(f"    🏆 New best! Val Acc: {val_acc:.2f}% → Saved model")

        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'history': history,
            'num_classes': num_classes,
            'class_names': train_dataset.classes,
        }, os.path.join(args.save_dir, 'checkpoint.pth'))

        if early_stopping(val_loss):
            print(f"\n   ⏹️ Early stopping triggered at epoch {epoch}")
            print(f"      Validation loss hasn't improved for {args.patience} epochs")
            break

    total_time = time.time() - start_time

    print("\n" + "=" * 60)
    print("   🎉 TRAINING COMPLETE")
    print("=" * 60)
    print(f"   Total Time:      {total_time/60:.1f} minutes")
    print(f"   Best Val Acc:    {history['best_val_acc']:.2f}%")
    print(f"   Best Epoch:      {history['best_epoch']}")
    print(f"   Model saved to:  {args.save_dir}/best_model.pth")
    print("=" * 60)

    history_path = os.path.join(args.save_dir, 'training_history.json')
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    print(f"   History saved to: {history_path}")

    save_training_curves(history, 'evaluation/outputs')

    return history


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train EfficientNet-B0 for plant disease classification"
    )

    parser.add_argument('--data_dir', type=str, default='data/processed',
                        help='Path to processed dataset with train/val/test splits')
    parser.add_argument('--image_size', type=int, default=224,
                        help='Input image size (default: 224 for EfficientNet-B0)')

    parser.add_argument('--epochs', type=int, default=30,
                        help='Total training epochs (default: 30)')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size (reduce if GPU OOM, default: 32)')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Initial learning rate (default: 0.001)')
    parser.add_argument('--freeze_epochs', type=int, default=5,
                        help='Epochs with backbone frozen (default: 5)')
    parser.add_argument('--patience', type=int, default=7,
                        help='Early stopping patience (default: 7)')

    parser.add_argument('--save_dir', type=str, default='models/saved',
                        help='Directory to save model weights')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint for resume training')

    parser.add_argument('--quick', action='store_true',
                        help='Quick training mode (5 epochs, for testing)')

    args = parser.parse_args()

    if args.quick:
        args.epochs = 5
        args.freeze_epochs = 2
        print("⚡ Quick training mode — 5 epochs only")

    train(args)
