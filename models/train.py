"""
=============================================================================
 Training Pipeline for Plant Disease Classification
 File: models/train.py
=============================================================================
 PURPOSE:
   Complete training pipeline with data augmentation, two-phase transfer
   learning, learning rate scheduling, early stopping, mixed precision,
   and comprehensive logging.

 USAGE:
   # Full training (30 epochs)
   python models/train.py

   # Quick training for testing (5 epochs)
   python models/train.py --epochs 5 --quick

   # Resume from checkpoint
   python models/train.py --resume models/saved/checkpoint.pth

   # Custom configuration
   python models/train.py --epochs 50 --batch_size 16 --lr 0.0005

 TRAINING STRATEGY (Two-Phase Transfer Learning):
   Phase 1 (Epochs 1 to freeze_epochs):
     - Backbone FROZEN (pre-trained EfficientNet-B0 weights preserved)
     - Only custom classifier head is trained
     - Higher learning rate (1e-3) since head is randomly initialized
     - Purpose: Learn disease-specific classification without corrupting
                the backbone's learned feature representations

   Phase 2 (Epochs freeze_epochs+1 to total_epochs):
     - ALL layers UNFROZEN (fine-tuning)
     - Lower learning rate (1e-4) to gently adapt backbone features
     - Cosine annealing LR schedule for smooth convergence
     - Purpose: Fine-tune the entire network for plant disease domain

 DATA AUGMENTATION:
   Training:
     - Random horizontal & vertical flip (p=0.5 each)
     - Random rotation (±15°)
     - Color jitter (brightness=0.2, contrast=0.2, saturation=0.1, hue=0.05)
     - Random affine (translate=10%, scale=90-110%)
     - Random perspective (p=0.3, distortion=0.2)
     - Normalize with ImageNet mean/std

   Validation/Test:
     - Resize to 256×256
     - Center crop to 224×224
     - Normalize with ImageNet mean/std (same as training)

 OUTPUT:
   models/saved/
     ├── best_model.pth          # Best validation accuracy weights
     ├── checkpoint.pth           # Latest checkpoint (for resume)
     └── training_history.json    # Loss/accuracy curves data
   evaluation/outputs/
     ├── training_curves.png      # Loss & accuracy plots
     └── training_log.txt         # Detailed training log
=============================================================================
"""

import os
import sys
import json
import time
import argparse
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.cuda.amp import GradScaler, autocast

import numpy as np
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.efficientnet_model import create_model


# =============================================================================
# DATA AUGMENTATION TRANSFORMS
# =============================================================================
# These transforms simulate real-world variations a farmer's phone camera
# would capture: different angles, lighting, zoom levels, etc.

# ImageNet normalization constants
# These are the mean and std of the ImageNet dataset used to pre-train
# EfficientNet-B0. We must use the same normalization for consistency.
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_train_transforms(image_size: int = 224):
    """
    Returns training data augmentation pipeline.

    AUGMENTATION RATIONALE:
      Each transform simulates a real-world variation:
      - RandomHorizontalFlip: Leaves can be photographed from either side
      - RandomVerticalFlip: Phone camera orientation varies
      - RandomRotation: Leaf angle varies in the field
      - ColorJitter: Lighting conditions change throughout the day
      - RandomAffine: Camera distance and angle vary
      - RandomPerspective: Simulates non-perpendicular viewing angles

    MATHEMATICAL NOTE:
      Augmentation effectively multiplies dataset size by ~10-20×
      without collecting new data. For N original images with K
      augmentation combinations, effective dataset size ≈ N × K.
    """
    return transforms.Compose([
        # Resize to slightly larger than target for random cropping
        transforms.Resize((image_size + 32, image_size + 32)),
        transforms.RandomCrop(image_size),

        # Geometric augmentations
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.3),
        transforms.RandomRotation(degrees=15),
        transforms.RandomAffine(
            degrees=0,
            translate=(0.1, 0.1),   # Up to 10% translation
            scale=(0.9, 1.1),       # 90% to 110% zoom
        ),
        transforms.RandomPerspective(distortion_scale=0.2, p=0.3),

        # Color augmentations (simulate lighting variations)
        transforms.ColorJitter(
            brightness=0.3,   # ±30% brightness (stronger for low light)
            contrast=0.3,     # ±30% contrast
            saturation=0.2,   # ±20% saturation
            hue=0.1,          # ±10% hue shift
        ),
        
        # Blur augmentation to simulate out-of-focus phone cameras
        transforms.RandomApply([
            transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5.0))
        ], p=0.2),

        # Convert to tensor and normalize
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def get_val_transforms(image_size: int = 224):
    """
    Returns validation/test data transforms (no augmentation).

    Validation data must be processed deterministically —
    no random augmentation — to get consistent evaluation metrics.
    Center crop ensures the leaf is centered in the frame.
    """
    return transforms.Compose([
        transforms.Resize((image_size + 32, image_size + 32)),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


# =============================================================================
# EARLY STOPPING
# =============================================================================

class EarlyStopping:
    """
    Stops training when validation loss hasn't improved for 'patience' epochs.

    MECHANISM:
      Tracks the best validation loss seen so far. If the loss doesn't
      improve (decrease) for 'patience' consecutive epochs, training is
      stopped to prevent overfitting.

    MATHEMATICAL BASIS:
      At epoch t, if val_loss(t) > val_loss_best for patience consecutive
      epochs, the model has likely started overfitting:
        train_loss(t) ↓ but val_loss(t) ↑ → overfitting

    ATTRIBUTES:
      patience (int): Number of epochs to wait for improvement
      counter (int): Current count of non-improving epochs
      best_loss (float): Lowest validation loss seen so far
      early_stop (bool): Set to True when patience is exhausted
    """

    def __init__(self, patience: int = 7, min_delta: float = 0.001):
        """
        PARAMETERS:
          patience (int): How many epochs to wait for improvement
          min_delta (float): Minimum change to qualify as improvement.
                            val_loss must decrease by at least min_delta
                            to count as improvement.
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss: float) -> bool:
        """
        Check if training should stop.

        RETURNS:
          bool: True if training should stop (patience exhausted)
        """
        if self.best_loss is None:
            # First epoch — set baseline
            self.best_loss = val_loss
            return False

        if val_loss < self.best_loss - self.min_delta:
            # Improvement detected — reset counter
            self.best_loss = val_loss
            self.counter = 0
            return False
        else:
            # No improvement — increment counter
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                return True
            return False


# =============================================================================
# TRAINING FUNCTIONS
# =============================================================================

def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    scaler: GradScaler = None,
    use_amp: bool = False,
) -> tuple:
    """
    Trains the model for one complete epoch.

    ONE EPOCH = one complete pass through the entire training dataset.
    For each mini-batch:
      1. Forward pass: predictions = model(images)
      2. Compute loss: L = CrossEntropy(predictions, labels)
      3. Backward pass: compute gradients ∂L/∂w for all parameters
      4. Update weights: w_new = w_old - η × ∂L/∂w (via Adam optimizer)

    MIXED PRECISION TRAINING (when use_amp=True):
      Uses FP16 for forward pass (faster, less memory) and FP32 for
      backward pass (numerical stability). GradScaler prevents underflow
      by scaling the loss before backprop.

    PARAMETERS:
      model: The neural network
      dataloader: Training data iterator (batches of images + labels)
      criterion: Loss function (CrossEntropyLoss)
      optimizer: Weight update algorithm (Adam)
      device: CPU or CUDA device
      scaler: GradScaler for mixed precision training
      use_amp: Whether to use Automatic Mixed Precision

    RETURNS:
      tuple: (average_loss, accuracy_percentage)
    """
    model.train()  # Set model to training mode (enables dropout, batch norm)

    running_loss = 0.0
    correct = 0
    total = 0

    # Progress bar for the training loop
    pbar = tqdm(dataloader, desc="   Training", leave=False)

    for images, labels in pbar:
        # Move data to device (GPU if available)
        images = images.to(device)
        labels = labels.to(device)

        # Zero out gradients from previous iteration
        # Without this, gradients would accumulate across batches
        optimizer.zero_grad()

        if use_amp and scaler is not None:
            # Mixed precision forward pass (FP16)
            with autocast():
                outputs = model(images)
                loss = criterion(outputs, labels)

            # Scale loss and backpropagate (prevents FP16 underflow)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            # Standard FP32 forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backpropagation: compute ∂L/∂w for all parameters
            loss.backward()

            # Update weights using optimizer (Adam)
            optimizer.step()

        # Track metrics
        running_loss += loss.item() * images.size(0)  # Sum of losses
        _, predicted = torch.max(outputs, 1)  # Get class with highest logit
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        # Update progress bar
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{100.0 * correct / total:.1f}%'
        })

    # Calculate epoch-level metrics
    epoch_loss = running_loss / total  # Average loss per sample
    epoch_acc = 100.0 * correct / total  # Accuracy percentage

    return epoch_loss, epoch_acc


def validate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple:
    """
    Evaluates the model on validation data (no gradient computation).

    This function runs the model in evaluation mode:
      - Dropout is DISABLED (all neurons active)
      - BatchNorm uses running statistics (not batch statistics)
      - torch.no_grad() prevents gradient computation (saves memory + speed)

    PARAMETERS:
      model: The neural network
      dataloader: Validation data iterator
      criterion: Loss function (CrossEntropyLoss)
      device: CPU or CUDA device

    RETURNS:
      tuple: (average_loss, accuracy_percentage)
    """
    model.eval()  # Set model to evaluation mode

    running_loss = 0.0
    correct = 0
    total = 0

    # No gradient computation during validation — saves memory and speed
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
    """
    Plots and saves training/validation loss and accuracy curves.

    These curves are essential for the research paper:
      - Convergence analysis: Does the model converge smoothly?
      - Overfitting detection: Does val_loss diverge from train_loss?
      - Phase transition: Is there a visible improvement when backbone unfreezes?

    GENERATES:
      training_curves.png — Side-by-side loss and accuracy plots
    """
    try:
        import matplotlib
        matplotlib.use('Agg')  # Non-interactive backend for servers
        import matplotlib.pyplot as plt
        import seaborn as sns

        sns.set_theme(style="darkgrid")
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        epochs = range(1, len(history['train_loss']) + 1)

        # ── Loss Curves ───────────────────────────────────────────────────
        ax1.plot(epochs, history['train_loss'], 'b-', linewidth=2, label='Train Loss')
        ax1.plot(epochs, history['val_loss'], 'r-', linewidth=2, label='Val Loss')
        ax1.set_xlabel('Epoch', fontsize=12)
        ax1.set_ylabel('Loss', fontsize=12)
        ax1.set_title('Training & Validation Loss', fontsize=14, fontweight='bold')
        ax1.legend(fontsize=11)
        ax1.grid(True, alpha=0.3)

        # Mark the phase transition (backbone unfreeze)
        if 'freeze_epochs' in history:
            ax1.axvline(x=history['freeze_epochs'], color='green',
                       linestyle='--', alpha=0.7, label='Backbone Unfrozen')

        # ── Accuracy Curves ────────────────────────────────────────────────
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
        print("   ⚠️  matplotlib not installed — skipping curve plots")


# =============================================================================
# MAIN TRAINING LOOP
# =============================================================================

def train(args):
    """
    Main training function — orchestrates the entire training pipeline.

    COMPLETE WORKFLOW:
      1. Set up device (GPU/CPU)
      2. Load datasets with augmentation transforms
      3. Create model with pre-trained backbone
      4. Phase 1: Train classifier head (backbone frozen)
      5. Phase 2: Fine-tune entire model (backbone unfrozen)
      6. Save best model, history, and training curves
      7. Print final results

    HYPERPARAMETERS:
      Epochs:        30 (default), configurable via --epochs
      Batch Size:    32 (default), reduce for GPU memory issues
      Learning Rate: 1e-3 (Phase 1), 1e-4 (Phase 2)
      Optimizer:     Adam (β₁=0.9, β₂=0.999, ε=1e-8)
      LR Scheduler:  CosineAnnealingLR (smooth decay)
      Loss:          CrossEntropyLoss (handles class imbalance via weights)
      Early Stop:    Patience=7 epochs

    CROSS-ENTROPY LOSS:
      L = -Σᵢ yᵢ · log(ŷᵢ)

      where:
        yᵢ = 1 if sample belongs to class i, 0 otherwise (one-hot)
        ŷᵢ = softmax(zᵢ) = exp(zᵢ) / Σⱼ exp(zⱼ)  (predicted probability)

    ADAM OPTIMIZER:
      Updates each parameter using adaptive learning rates:
        mₜ = β₁ · mₜ₋₁ + (1 - β₁) · gₜ        (first moment / mean)
        vₜ = β₂ · vₜ₋₁ + (1 - β₂) · gₜ²       (second moment / variance)
        m̂ₜ = mₜ / (1 - β₁ᵗ)                     (bias correction)
        v̂ₜ = vₜ / (1 - β₂ᵗ)                     (bias correction)
        θₜ = θₜ₋₁ - η · m̂ₜ / (√v̂ₜ + ε)         (parameter update)
    """
    # ── 1. Device Setup ────────────────────────────────────────────────────
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    use_amp = torch.cuda.is_available()  # Mixed precision only on GPU

    print("\n" + "=" * 60)
    print("  🌿 PLANT DISEASE MODEL — TRAINING PIPELINE")
    print("=" * 60)
    print(f"  Device:       {device}")
    print(f"  Mixed Prec:   {'✅ Enabled' if use_amp else '❌ Disabled (CPU)'}")
    print(f"  Epochs:       {args.epochs}")
    print(f"  Batch Size:   {args.batch_size}")
    print(f"  Learning Rate:{args.lr}")
    print(f"  Freeze Epochs:{args.freeze_epochs}")
    print(f"  Data Dir:     {args.data_dir}")
    print("=" * 60)

    # ── 2. Load Datasets ───────────────────────────────────────────────────
    train_dir = os.path.join(args.data_dir, 'train')
    val_dir = os.path.join(args.data_dir, 'val')

    if not os.path.isdir(train_dir):
        print(f"\n❌ Training data not found at: {train_dir}")
        print("   Run first: python data/download_dataset.py")
        sys.exit(1)

    # ImageFolder automatically maps subdirectory names to class labels
    train_dataset = datasets.ImageFolder(
        root=train_dir,
        transform=get_train_transforms(args.image_size)
    )
    val_dataset = datasets.ImageFolder(
        root=val_dir,
        transform=get_val_transforms(args.image_size)
    )

    num_classes = len(train_dataset.classes)
    print(f"\n  📂 Dataset loaded:")
    print(f"     Train:    {len(train_dataset):,} images")
    print(f"     Val:      {len(val_dataset):,} images")
    print(f"     Classes:  {num_classes}")

    # DataLoader handles batching, shuffling, and parallel data loading
    # num_workers: number of CPU threads for data loading (0 for Windows compatibility)
    num_workers = 0 if sys.platform == 'win32' else 4
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,           # Random order each epoch
        num_workers=num_workers,
        pin_memory=True,        # Faster GPU transfer
        drop_last=True,         # Drop incomplete final batch (for BatchNorm)
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,          # Deterministic order for validation
        num_workers=num_workers,
        pin_memory=True,
    )

    # ── 3. Create Model ───────────────────────────────────────────────────
    model = create_model(num_classes=num_classes, pretrained=True)
    model = model.to(device)

    # ── 4. Loss Function with Class Weights ────────────────────────────────
    # Compute class weights to handle imbalanced classes
    # Weight inversely proportional to class frequency:
    #   w_c = N_total / (C × N_c)
    # where N_c is the count of samples in class c, C is number of classes
    class_counts = np.zeros(num_classes)
    for _, label in train_dataset.samples:
        class_counts[label] += 1

    # Avoid division by zero for empty classes
    class_weights = np.sum(class_counts) / (num_classes * class_counts + 1e-6)
    class_weights = torch.FloatTensor(class_weights).to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # ── 5. Training History ────────────────────────────────────────────────
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': [],
        'lr': [], 'freeze_epochs': args.freeze_epochs,
        'best_val_acc': 0.0, 'best_epoch': 0,
    }

    # Early stopping and GradScaler
    early_stopping = EarlyStopping(patience=args.patience)
    scaler = GradScaler() if use_amp else None

    # Create save directory
    os.makedirs(args.save_dir, exist_ok=True)

    best_val_acc = 0.0
    start_time = time.time()

    # ── 6. Training Loop ──────────────────────────────────────────────────
    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()

        # ── Phase Transition Logic ─────────────────────────────────────
        if epoch <= args.freeze_epochs:
            # Phase 1: Backbone frozen
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
                print(f"\n  ═══ PHASE 1: Training Classifier Head (Backbone Frozen) ═══")
        else:
            # Phase 2: Full fine-tuning
            if epoch == args.freeze_epochs + 1:
                model.unfreeze_backbone()
                # Lower learning rate for fine-tuning to prevent
                # destroying pre-trained features
                optimizer = optim.Adam(
                    model.parameters(),
                    lr=args.lr * 0.1,  # 10× lower LR for backbone
                    weight_decay=1e-4,
                )
                scheduler = optim.lr_scheduler.CosineAnnealingLR(
                    optimizer,
                    T_max=args.epochs - args.freeze_epochs,
                    eta_min=1e-7,
                )
                print(f"\n  ═══ PHASE 2: Fine-tuning Entire Model ═══")

        # ── Train for one epoch ────────────────────────────────────────
        current_lr = optimizer.param_groups[0]['lr']
        print(f"\n  Epoch {epoch}/{args.epochs} [LR: {current_lr:.2e}]")

        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device, scaler, use_amp
        )

        # ── Validate ───────────────────────────────────────────────────
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        # ── Update LR Scheduler ────────────────────────────────────────
        scheduler.step()

        # ── Record History ─────────────────────────────────────────────
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['lr'].append(current_lr)

        epoch_time = time.time() - epoch_start

        # ── Print Epoch Summary ────────────────────────────────────────
        print(f"   Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"   Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.2f}%")
        print(f"   Time:       {epoch_time:.1f}s")

        # ── Save Best Model ────────────────────────────────────────────
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
            print(f"   🏆 New best! Val Acc: {val_acc:.2f}% → Saved model")

        # ── Save Checkpoint ────────────────────────────────────────────
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'history': history,
            'num_classes': num_classes,
            'class_names': train_dataset.classes,
        }, os.path.join(args.save_dir, 'checkpoint.pth'))

        # ── Early Stopping Check ───────────────────────────────────────
        if early_stopping(val_loss):
            print(f"\n   ⏹️  Early stopping triggered at epoch {epoch}")
            print(f"      Validation loss hasn't improved for {args.patience} epochs")
            break

    # ── 7. Training Complete ───────────────────────────────────────────────
    total_time = time.time() - start_time

    print("\n" + "=" * 60)
    print("  🎉 TRAINING COMPLETE")
    print("=" * 60)
    print(f"  Total Time:      {total_time/60:.1f} minutes")
    print(f"  Best Val Acc:    {history['best_val_acc']:.2f}%")
    print(f"  Best Epoch:      {history['best_epoch']}")
    print(f"  Model saved to:  {args.save_dir}/best_model.pth")
    print("=" * 60)

    # ── 8. Save Training History ───────────────────────────────────────────
    history_path = os.path.join(args.save_dir, 'training_history.json')
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    print(f"  History saved to: {history_path}")

    # ── 9. Save Training Curves ────────────────────────────────────────────
    save_training_curves(history, 'evaluation/outputs')

    return history


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train EfficientNet-B0 for plant disease classification"
    )

    # Data arguments
    parser.add_argument('--data_dir', type=str, default='data/processed',
                       help='Path to processed dataset with train/val/test splits')
    parser.add_argument('--image_size', type=int, default=224,
                       help='Input image size (default: 224 for EfficientNet-B0)')

    # Training arguments
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

    # Save/Resume
    parser.add_argument('--save_dir', type=str, default='models/saved',
                       help='Directory to save model weights')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint for resume training')

    # Quick mode
    parser.add_argument('--quick', action='store_true',
                       help='Quick training mode (5 epochs, for testing)')

    args = parser.parse_args()

    # Quick mode overrides
    if args.quick:
        args.epochs = 5
        args.freeze_epochs = 2
        print("⚡ Quick training mode — 5 epochs only")

    # Start training
    train(args)
