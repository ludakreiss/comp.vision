import time
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm

import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)

import config
from dataset import get_dataloaders
from model import build_model
from transforms import get_degradation_transform_for_epoch

class DeepfakeLoss(nn.Module):
    def __init__(self, loss_type="focal", alpha=0.50, gamma=1.5, smoothing=0.05, class_weights=None):
        super().__init__()
        self.loss_type = loss_type
        self.alpha = alpha
        self.gamma = gamma
        self.smoothing = smoothing
        self.class_weights = class_weights # dict {0: w0, 1: w1}

    def forward(self, inputs, targets):
        inputs = inputs.float()
        targets = targets.float()
        
        # Apply label smoothing
        if self.smoothing > 0:
            targets_smooth = targets * (1.0 - self.smoothing) + 0.5 * self.smoothing
        else:
            targets_smooth = targets
            
        bce_loss = nn.functional.binary_cross_entropy_with_logits(inputs, targets_smooth, reduction='none')
        
        # Apply class weights if available
        if self.class_weights is not None:
            w0 = self.class_weights[0]
            w1 = self.class_weights[1]
            weight_t = targets * w1 + (1.0 - targets) * w0
            bce_loss = bce_loss * weight_t

        if self.loss_type == "focal":
            probs = torch.sigmoid(inputs)
            p_t = probs * targets + (1.0 - probs) * (1.0 - targets)
            p_t = torch.clamp(p_t, 1e-6, 1.0 - 1e-6)
            focal_weight = (1.0 - p_t) ** self.gamma
            alpha_t = self.alpha * targets + (1.0 - self.alpha) * (1.0 - targets)
            loss = alpha_t * focal_weight * bce_loss
        else:
            loss = bce_loss
            
        return loss.mean()


class LRFinder:
    def __init__(self, model, optimizer, criterion, device):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        
    def range_test(self, train_loader, start_lr=1e-7, end_lr=10.0, num_iter=100):
        # Save model state
        save_dict = {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict()
        }
        
        self.model.train()
        lrs = []
        losses = []
        best_loss = float('inf')
        
        mult = (end_lr / start_lr) ** (1.0 / num_iter)
        lr = start_lr
        
        # Set learning rate in optimizer
        for pg in self.optimizer.param_groups:
            pg['lr'] = lr
            
        iterator = iter(train_loader)
        
        for i in range(num_iter):
            try:
                batch = next(iterator)
            except StopIteration:
                iterator = iter(train_loader)
                batch = next(iterator)
                
            images = batch["image"].to(self.device)
            labels = batch["label"].to(self.device)
            
            self.optimizer.zero_grad(set_to_none=True)
            if self.device.type == "cuda":
                with torch.amp.autocast(device_type="cuda"):
                    logits, _ = self.model(images)
                    logits = logits.squeeze(1)
                    loss = self.criterion(logits, labels)
            else:
                logits, _ = self.model(images)
                logits = logits.squeeze(1)
                loss = self.criterion(logits, labels)
            
            if torch.isnan(loss) or loss > best_loss * 4:
                break
                
            if loss.item() < best_loss:
                best_loss = loss.item()
                
            losses.append(loss.item())
            lrs.append(lr)
            
            loss.backward()
            self.optimizer.step()
            
            lr *= mult
            for pg in self.optimizer.param_groups:
                pg['lr'] = lr
                
        # Restore model and optimizer
        self.model.load_state_dict(save_dict["model"])
        self.optimizer.load_state_dict(save_dict["optimizer"])
        
        return lrs, losses



class EMA:
    def __init__(self, model, decay=0.999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        self._num_updates = 0
        self.register()

    def register(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def register_new_parameters(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and name not in self.shadow:
                self.shadow[name] = param.data.clone()

    def update(self):
        # Step-ramped warmup decay (timm / PyTorch convention):
        # decay ramps from ~0 up to config.EMA_DECAY as step count grows,
        # ensuring the shadow weights track the live model closely in early training
        # rather than being locked near the random-initialised starting point.
        self._num_updates += 1
        effective_decay = min(self.decay, (1.0 + self._num_updates) / (10.0 + self._num_updates))
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                new_average = (1.0 - effective_decay) * param.data + effective_decay * self.shadow[name]
                self.shadow[name] = new_average.clone()

    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone()
                param.data.copy_(self.shadow[name])

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.backup
                param.data.copy_(self.backup[name])
        self.backup = {}


def plot_diagnostic_curves(labels, probabilities, output_directory):
    labels = np.asarray(labels).astype(int)
    probabilities = np.asarray(probabilities)
    predictions = (probabilities >= 0.5).astype(int)
    
    # 1. Confusion Matrix
    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
    cm = confusion_matrix(labels, predictions)
    plt.figure(figsize=(6, 5))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Real", "Fake"])
    disp.plot(cmap="Blues", values_format="d")
    plt.title("Validation Confusion Matrix")
    plt.savefig(output_directory / "val_confusion_matrix.png", dpi=150, bbox_inches="tight")
    plt.close()
    
    # 2. ROC Curve
    from sklearn.metrics import roc_curve, auc
    if len(np.unique(labels)) == 2:
        fpr, tpr, _ = roc_curve(labels, probabilities)
        roc_auc = auc(fpr, tpr)
        plt.figure(figsize=(6, 5))
        plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (AUC = {roc_auc:.4f})")
        plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("Validation ROC Curve")
        plt.legend(loc="lower right")
        plt.grid(alpha=0.3)
        plt.savefig(output_directory / "val_roc_curve.png", dpi=150, bbox_inches="tight")
        plt.close()
        
    # 3. Precision-Recall Curve
    from sklearn.metrics import precision_recall_curve, average_precision_score
    if len(np.unique(labels)) == 2:
        precision, recall, _ = precision_recall_curve(labels, probabilities)
        ap = average_precision_score(labels, probabilities)
        plt.figure(figsize=(6, 5))
        plt.plot(recall, precision, color="green", lw=2, label=f"PR curve (AP = {ap:.4f})")
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title("Validation Precision-Recall Curve")
        plt.legend(loc="lower left")
        plt.grid(alpha=0.3)
        plt.savefig(output_directory / "val_precision_recall_curve.png", dpi=150, bbox_inches="tight")
        plt.close()
        
    # 4. Calibration Curve
    from sklearn.calibration import calibration_curve
    if len(np.unique(labels)) == 2:
        prob_true, prob_pred = calibration_curve(labels, probabilities, n_bins=10)
        plt.figure(figsize=(6, 5))
        plt.plot(prob_pred, prob_true, marker="s", lw=1, label="Model")
        plt.plot([0, 1], [0, 1], linestyle="--", label="Perfect Calibration")
        plt.xlabel("Mean Predicted Probability")
        plt.ylabel("Fraction of Positives")
        plt.title("Validation Calibration Plot")
        plt.legend(loc="upper left")
        plt.grid(alpha=0.3)
        plt.savefig(output_directory / "val_calibration_curve.png", dpi=150, bbox_inches="tight")
        plt.close()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def calculate_binary_metrics(labels, probabilities, threshold=0.5):
    labels = np.asarray(labels).astype(int)
    probabilities = np.asarray(probabilities)
    predictions = (probabilities >= threshold).astype(int)

    metrics = {
        "accuracy": accuracy_score(labels, predictions),
        "balanced_accuracy": balanced_accuracy_score(labels, predictions),
        "precision": precision_score(labels, predictions, zero_division=0),
        "recall": recall_score(labels, predictions, zero_division=0),
        "f1": f1_score(labels, predictions, zero_division=0),
        "roc_auc": float("nan"),
    }

    if len(np.unique(labels)) == 2:
        metrics["roc_auc"] = roc_auc_score(labels, probabilities)

    return metrics


def sync_scheduler_param_groups(scheduler, optimizer):
    """Synchronize learning rate scheduler internal base_lrs and _last_lr when new param groups are added."""
    target_len = len(optimizer.param_groups)
    sub_schedulers = getattr(scheduler, "_schedulers", [scheduler])
    for sub in sub_schedulers:
        while hasattr(sub, "base_lrs") and len(sub.base_lrs) < target_len:
            sub.base_lrs.append(optimizer.param_groups[len(sub.base_lrs)]["lr"])
        while hasattr(sub, "_last_lr") and len(sub._last_lr) < target_len:
            sub._last_lr.append(optimizer.param_groups[len(sub._last_lr)]["lr"])
    if hasattr(scheduler, "_last_lr"):
        while len(scheduler._last_lr) < target_len:
            scheduler._last_lr.append(optimizer.param_groups[len(scheduler._last_lr)]["lr"])


def train_one_epoch(model, loader, criterion, optimizer, scaler, device, limit_batches=None, ema=None):
    model.train()
    running_loss = 0.0
    all_labels = []
    all_probabilities = []
    accum_steps = max(1, getattr(config, "GRADIENT_ACCUMULATION_STEPS", 1))

    optimizer.zero_grad(set_to_none=True)

    for i, batch in enumerate(tqdm(loader, desc="Training", leave=False)):
        if limit_batches is not None and i >= limit_batches:
            break

        images = batch["image"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)

        use_mixup = (
            getattr(config, "USE_MIXUP", False)
            and config.TRAINING_STRATEGY != "clean"  # Never mix clean baseline
            and (random.random() < getattr(config, "MIXUP_PROB", 0.5))
        )
        if use_mixup and images.size(0) > 1:
            lam = np.random.beta(getattr(config, "MIXUP_ALPHA", 0.8), getattr(config, "MIXUP_ALPHA", 0.8))
            index = torch.randperm(images.size(0)).to(device)
            images = lam * images + (1 - lam) * images[index]
            targets_a, targets_b = labels, labels[index]
        else:
            use_mixup = False

        is_accumulating = ((i + 1) % accum_steps != 0) and ((i + 1) != len(loader))

        if device.type == "cuda":
            with torch.amp.autocast(device_type="cuda"):
                logits, _ = model(images)
                logits = logits.squeeze(1)
                if use_mixup:
                    loss = lam * criterion(logits, targets_a) + (1 - lam) * criterion(logits, targets_b)
                else:
                    loss = criterion(logits, labels)
                scaled_loss = loss / accum_steps

            scaler.scale(scaled_loss).backward()

            if not is_accumulating:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.GRADIENT_CLIPPING)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                if ema is not None:
                    ema.update()
        else:
            logits, _ = model(images)
            logits = logits.squeeze(1)
            if use_mixup:
                loss = lam * criterion(logits, targets_a) + (1 - lam) * criterion(logits, targets_b)
            else:
                loss = criterion(logits, labels)
            scaled_loss = loss / accum_steps
            scaled_loss.backward()

            if not is_accumulating:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.GRADIENT_CLIPPING)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                if ema is not None:
                    ema.update()

        probabilities = torch.sigmoid(logits.detach())
        running_loss += loss.item() * images.size(0)
        all_labels.extend(labels.detach().cpu().numpy().tolist())
        all_probabilities.extend(probabilities.cpu().numpy().tolist())

    metrics = calculate_binary_metrics(all_labels, all_probabilities)
    metrics["loss"] = running_loss / len(all_labels) if all_labels else 0.0
    return metrics


@torch.no_grad()
def evaluate_model(model, loader, criterion, device, limit_batches=None, use_tta=False):
    model.eval()
    running_loss = 0.0
    all_labels = []
    all_probabilities = []
    all_paths = []
    all_video_ids = []

    for i, batch in enumerate(tqdm(loader, desc="Evaluating", leave=False)):
        if limit_batches is not None and i >= limit_batches:
            break

        images = batch["image"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)

        if use_tta:
            # Original
            logits1, _ = model(images)
            probs1 = torch.sigmoid(logits1.squeeze(1))
            
            # Horizontal Flip
            images_hf = torch.flip(images, [3])
            logits2, _ = model(images_hf)
            probs2 = torch.sigmoid(logits2.squeeze(1))
            
            # Mild Resize
            images_res = torch.nn.functional.interpolate(images, scale_factor=0.9, mode='bilinear', align_corners=False)
            images_res = torch.nn.functional.interpolate(images_res, size=images.shape[2:], mode='bilinear', align_corners=False)
            logits3, _ = model(images_res)
            probs3 = torch.sigmoid(logits3.squeeze(1))
            
            probabilities = (probs1 + probs2 + probs3) / 3.0
            probs_clamped = torch.clamp(probabilities, 1e-6, 1.0 - 1e-6)
            logits = torch.logit(probs_clamped)
            loss = criterion(logits, labels)
        else:
            logits, _ = model(images)
            logits = logits.squeeze(1)
            loss = criterion(logits, labels)
            probabilities = torch.sigmoid(logits)

        running_loss += loss.item() * images.size(0)
        all_labels.extend(labels.cpu().numpy().tolist())
        all_probabilities.extend(probabilities.cpu().numpy().tolist())
        all_paths.extend(batch["path"])
        all_video_ids.extend(batch["video_id"])

    metrics = calculate_binary_metrics(all_labels, all_probabilities)
    metrics["loss"] = running_loss / len(all_labels) if all_labels else 0.0

    predictions_df = pd.DataFrame({
        "image_path": all_paths,
        "video_id": all_video_ids,
        "label": all_labels,
        "prob_fake": all_probabilities,
    })

    return metrics, predictions_df


def plot_training_curves(history_df, output_directory):
    plt.figure(figsize=(10, 5))
    plt.plot(history_df["epoch"], history_df["train_loss"], marker="o", label="Train Loss")
    plt.plot(history_df["epoch"], history_df["validation_loss"], marker="o", label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"{config.MODEL_NAME} - Loss Curves ({config.TRAINING_STRATEGY})")
    plt.legend()
    plt.grid(alpha=0.3)
    loss_plot_path = output_directory / "loss_curves.png"
    plt.savefig(loss_plot_path, dpi=150)
    plt.close()

    plt.figure(figsize=(10, 5))
    if "validation_roc_auc" in history_df.columns:
        plt.plot(history_df["epoch"], history_df["validation_roc_auc"], marker="o", label="Val ROC-AUC")
    plt.plot(history_df["epoch"], history_df["validation_accuracy"], marker="o", label="Val Accuracy")
    plt.plot(history_df["epoch"], history_df["validation_f1"], marker="o", label="Val F1")
    plt.xlabel("Epoch")
    plt.ylabel("Score")
    plt.title(f"{config.MODEL_NAME} - Validation Metrics ({config.TRAINING_STRATEGY})")
    plt.legend()
    plt.grid(alpha=0.3)
    metrics_plot_path = output_directory / "metrics_curves.png"
    plt.savefig(metrics_plot_path, dpi=150)
    plt.close()
    
    print(f"Generated metric plots in {output_directory}")


def compute_class_balanced_weights(dataframe, beta=0.999):
    class_counts = dataframe["label"].value_counts().to_dict()
    c0 = class_counts.get(0.0, 1)
    c1 = class_counts.get(1.0, 1)
    w0 = (1.0 - beta) / (1.0 - np.power(beta, c0))
    w1 = (1.0 - beta) / (1.0 - np.power(beta, c1))
    sum_w = w0 + w1
    w0 = (w0 / sum_w) * 2.0
    w1 = (w1 / sum_w) * 2.0
    return {0: w0, 1: w1}


def average_checkpoints(checkpoint_paths, output_path, device):
    print(f"\n[+] Running Post-Training Checkpoint Averaging over {len(checkpoint_paths)} checkpoints...")
    if not checkpoint_paths:
        return
    
    first_ckpt = torch.load(checkpoint_paths[0], map_location=device, weights_only=False)
    averaged_state = first_ckpt["model_state_dict"]
    
    for path in checkpoint_paths[1:]:
        state = torch.load(path, map_location=device, weights_only=False)["model_state_dict"]
        for key in averaged_state.keys():
            averaged_state[key] = averaged_state[key] + state[key]
            
    num_checkpoints = len(checkpoint_paths)
    for key in averaged_state.keys():
        if averaged_state[key].is_floating_point():
            averaged_state[key] = averaged_state[key] / num_checkpoints
        else:
            averaged_state[key] = first_ckpt["model_state_dict"][key]
            
    first_ckpt["model_state_dict"] = averaged_state
    first_ckpt["averaged_checkpoints"] = [str(p) for p in checkpoint_paths]
    
    torch.save(first_ckpt, output_path)
    print(f"--> Saved averaged model checkpoint to: {output_path}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Train deepfake detection model")
    parser.add_argument("--model", type=str, default=config.MODEL_NAME, choices=["efficientnet_b0", "efficientnet_b4", "convnext_tiny", "resnet18", "resnet50", "mobilenet_v3_small", "shufflenet_v2", "densenet121"], help="Model backbone")
    parser.add_argument("--strategy", type=str, default=config.TRAINING_STRATEGY, choices=["clean", "standard", "degradation"], help="Training strategy")
    parser.add_argument("--epochs", type=int, default=config.NUM_EPOCHS, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=config.BATCH_SIZE, help="Batch size")
    parser.add_argument("--num_workers", type=int, default=config.NUM_WORKERS, help="Number of dataloader workers")
    parser.add_argument("--backbone_lr", type=float, default=config.BACKBONE_LR, help="Backbone learning rate")
    parser.add_argument("--classifier_lr", type=float, default=config.CLASSIFIER_LR, help="Classifier learning rate")
    parser.add_argument("--patience", type=int, default=config.PATIENCE, help="Early stopping patience")
    parser.add_argument("--limit_batches", type=int, default=None, help="Limit number of batches per epoch (for sanity check)")
    parser.add_argument("--find_lr", action="store_true", help="Run learning rate finder and exit")
    args = parser.parse_args()

    # Override config global values
    config.MODEL_NAME = args.model
    config.TRAINING_STRATEGY = args.strategy
    config.NUM_EPOCHS = args.epochs
    config.BATCH_SIZE = args.batch_size
    config.NUM_WORKERS = args.num_workers
    config.BACKBONE_LR = args.backbone_lr
    config.CLASSIFIER_LR = args.classifier_lr
    config.PATIENCE = args.patience

    # Dynamic image size adaptation
    if config.MODEL_NAME == "efficientnet_b4":
        config.IMAGE_SIZE = 380
        print(f"Adapting image size to {config.IMAGE_SIZE} for efficientnet_b4 backbone")
    else:
        config.IMAGE_SIZE = 224

    set_seed(config.SEED)

    if not config.MANIFEST_PATH.exists():
        print(f"Manifest file not found at {config.MANIFEST_PATH}. Please run extract_faces.py first.")
        return

    # Load data manifest
    print("Loading data manifest...")
    manifest_df = pd.read_csv(config.MANIFEST_PATH)
    
    # Get dataloaders
    print("Setting up dataloaders...")
    train_loader, val_loader, _ = get_dataloaders(manifest_df)
    print(f"Dataloaders initialized. Train batches: {len(train_loader)} | Val batches: {len(val_loader)}")

    # Device Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Selected device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Build Model
    print(f"Building model: {config.MODEL_NAME} (pretrained={config.PRETRAINED})...")
    model = build_model(config.MODEL_NAME, pretrained=config.PRETRAINED).to(device)
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of trainable parameters: {trainable_params:,}")

    # Loss Configuration (including Class-Balanced Weights)
    class_weights = None
    if config.BALANCING_STRATEGY == "class_balanced_loss":
        train_df_labels = manifest_df[manifest_df["split"] == "train"].copy()
        class_weights = compute_class_balanced_weights(train_df_labels, beta=config.CB_BETA)
        print(f"--> Using Class-Balanced Loss weights: {class_weights}")

    loss_type = "bce"
    if config.BALANCING_STRATEGY == "focal_loss":
        loss_type = "focal"

    focal_alpha = getattr(config, "FOCAL_ALPHA", 0.50)
    focal_gamma = getattr(config, "FOCAL_GAMMA", 1.5)
    criterion = DeepfakeLoss(
        loss_type=loss_type,
        alpha=focal_alpha,
        gamma=focal_gamma,
        smoothing=config.LABEL_SMOOTHING,
        class_weights=class_weights
    )

    # Separate parameters: 2D weights get weight decay, 1D vectors (BN params, biases) get 0 weight decay
    decay_backbone = [p for n, p in model.named_parameters() if p.requires_grad and not ("classifier" in n or "mlp" in n or "head" in n) and p.ndim >= 2]
    no_decay_backbone = [p for n, p in model.named_parameters() if p.requires_grad and not ("classifier" in n or "mlp" in n or "head" in n) and p.ndim < 2]
    decay_classifier = [p for n, p in model.named_parameters() if p.requires_grad and ("classifier" in n or "mlp" in n or "head" in n) and p.ndim >= 2]
    no_decay_classifier = [p for n, p in model.named_parameters() if p.requires_grad and ("classifier" in n or "mlp" in n or "head" in n) and p.ndim < 2]

    optimizer = torch.optim.AdamW(
        [
            {"params": decay_backbone, "lr": config.BACKBONE_LR, "weight_decay": config.WEIGHT_DECAY},
            {"params": no_decay_backbone, "lr": config.BACKBONE_LR, "weight_decay": 0.0},
            {"params": decay_classifier, "lr": config.CLASSIFIER_LR, "weight_decay": config.WEIGHT_DECAY},
            {"params": no_decay_classifier, "lr": config.CLASSIFIER_LR, "weight_decay": 0.0},
        ]
    )

    # Smith-style Learning Rate Finder execution
    if args.find_lr:
        print("\n[+] Triggering Learning Rate Range Test...")
        lrs, losses = LRFinder(model, optimizer, criterion, device).range_test(train_loader, num_iter=100)
        
        lr_plot_path = config.OUTPUT_ROOT / "lr_finder.png"
        plt.figure(figsize=(10, 5))
        plt.plot(lrs, losses)
        plt.xscale('log')
        plt.xlabel('Learning Rate')
        plt.ylabel('Loss')
        plt.title(f'Learning Rate Finder: {config.MODEL_NAME}')
        plt.grid(True)
        plt.savefig(lr_plot_path, dpi=150)
        plt.close()
        print(f"--> Learning Rate Finder complete. Diagnostic plot saved to: {lr_plot_path}")
        
        gradients = np.diff(losses)
        if len(gradients) > 0:
            best_idx = np.argmin(gradients)
            suggested_lr = lrs[best_idx]
            print(f"--> Suggested Steepest Descent Learning Rate: {suggested_lr:.2e}")
        return

    # Schedulers
    if config.SCHEDULER == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.5, patience=3
        )
    else:
        warmup_epochs = getattr(config, "WARMUP_EPOCHS", 2)
        scheduler1 = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_epochs
        )
        scheduler2 = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(1, config.NUM_EPOCHS - warmup_epochs)
        )
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer, schedulers=[scheduler1, scheduler2], milestones=[warmup_epochs]
        )
    scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" else None

    # Output experiment setup
    experiment_name = f"{config.MODEL_NAME}_{config.TRAINING_STRATEGY}"
    experiment_directory = config.OUTPUT_ROOT / experiment_name
    experiment_directory.mkdir(parents=True, exist_ok=True)

    checkpoint_path = experiment_directory / "best_model.pt"
    last_checkpoint_path = experiment_directory / "last_model.pt"
    history_path = experiment_directory / "history.csv"

    # TensorBoard setup
    writer = None
    try:
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(log_dir=str(config.TENSORBOARD_DIR / experiment_name))
        print(f"TensorBoard logging initialized at: {config.TENSORBOARD_DIR / experiment_name}")
    except Exception as e:
        print(f"Could not load TensorBoard writer: {e}")

    # Auto-resume training if last checkpoint exists
    start_epoch = 1
    best_validation_auc = -np.inf
    best_validation_score = -np.inf
    using_auc_tracking = True
    epochs_without_improvement = 0
    history = []

    if last_checkpoint_path.exists():
        print(f"Found existing last checkpoint at {last_checkpoint_path}. Resuming...")
        try:
            checkpoint = torch.load(
                last_checkpoint_path,
                map_location=device,
                weights_only=False,
            )
            model.load_state_dict(checkpoint["model_state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            if hasattr(scheduler, "T_max"):
                scheduler.T_max = config.NUM_EPOCHS
            elif hasattr(scheduler, "_schedulers") and len(scheduler._schedulers) > 1:
                scheduler._schedulers[1].T_max = max(1, config.NUM_EPOCHS - 5)
            if scaler is not None and checkpoint.get("scaler_state_dict") is not None:
                scaler.load_state_dict(checkpoint["scaler_state_dict"])
            
            start_epoch = checkpoint["epoch"] + 1
            best_validation_auc = checkpoint["best_validation_auc"]
            best_validation_score = checkpoint.get("best_validation_score", best_validation_auc)
            using_auc_tracking = checkpoint.get("using_auc_tracking", True)
            history = checkpoint.get("history", [])
            epochs_without_improvement = checkpoint.get("epochs_without_improvement", 0)
            tracking_desc = "AUC" if using_auc_tracking else "Balanced Accuracy"
            print(
                f"Resuming training from Epoch {start_epoch} | "
                f"Best AUC: {best_validation_auc:.4f} | "
                f"Best Score: {best_validation_score:.4f} (tracking {tracking_desc})"
            )
            
            # Rebuild optimizer with correct param groups, preserving momentum states
            resumed_backbone_params = [p for n, p in model.named_parameters() if p.requires_grad and not ("classifier" in n or "mlp" in n)]
            resumed_classifier_params = [p for n, p in model.named_parameters() if p.requires_grad and ("classifier" in n or "mlp" in n)]
            old_state = optimizer.state
            optimizer = torch.optim.AdamW(
                [
                    {"params": resumed_backbone_params, "lr": config.BACKBONE_LR},
                    {"params": resumed_classifier_params, "lr": config.CLASSIFIER_LR},
                ],
                weight_decay=config.WEIGHT_DECAY,
            )
            for group in optimizer.param_groups:
                for p in group["params"]:
                    if p in old_state:
                        optimizer.state[p] = old_state[p]
        except Exception as error:
            print(f"Could not load checkpoint ({error}). Starting training from scratch.")

    # Initialize EMA tracker
    ema = EMA(model, decay=config.EMA_DECAY)

    # Track best checkpoints for averaging
    best_checkpoints = [] # list of (val_score, file_path)

    # Determine num_freeze
    num_features = len(model.features) if hasattr(model, "features") else 0
    num_freeze = int(num_features * config.FREEZE_PERCENT)

    print(f"Starting training loop from Epoch {start_epoch} to {config.NUM_EPOCHS}...")
    training_start_time = time.time()
    for epoch in range(start_epoch, config.NUM_EPOCHS + 1):
        epoch_start_time = time.time()
        
        # Generic progressive unfreezing logic
        unfrozen_this_epoch = False
        if config.PROGRESSIVE_UNFREEZE and num_freeze > 0:
            unfreeze_interval = max(1, int((config.NUM_EPOCHS * 0.8) / num_freeze))
            if (epoch - 1) % unfreeze_interval == 0:
                unfreeze_idx = num_freeze - 1 - ((epoch - 1) // unfreeze_interval)
                if unfreeze_idx >= 0:
                    for param in model.features[unfreeze_idx].parameters():
                        param.requires_grad = True
                    print(f"--> Epoch {epoch}: Progressive Unfreezing block {unfreeze_idx} of backbone.")
                    unfrozen_this_epoch = True

        if unfrozen_this_epoch:
            existing_params = set()
            for group in optimizer.param_groups:
                existing_params.update(group["params"])

            newly_unfrozen = [
                p for p in model.parameters()
                if p.requires_grad and p not in existing_params
            ]

            if newly_unfrozen:
                current_backbone_lr = optimizer.param_groups[0]["lr"]
                optimizer.add_param_group({
                    "params": newly_unfrozen,
                    "lr": current_backbone_lr,
                    "weight_decay": config.WEIGHT_DECAY,
                })
                sync_scheduler_param_groups(scheduler, optimizer)

            ema.register_new_parameters()
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f"--> Updated number of trainable parameters: {trainable_params:,}")
        
        # Curriculum: update training transform severity each epoch (degradation strategy only)
        if (
            config.TRAINING_STRATEGY == "degradation"
            and getattr(config, "CURRICULUM_ENABLED", False)
        ):
            epoch_transform = get_degradation_transform_for_epoch(epoch, config.NUM_EPOCHS)
            train_loader.dataset.transform = epoch_transform
            if epoch == 1 or epoch == getattr(config, "CURRICULUM_RAMP_EPOCHS", 0):
                severity = getattr(config, "CURRICULUM_SEVERITY_START", 0.3) if epoch == 1 else 1.0
                print(f"---> Epoch {epoch}: Curriculum severity = {severity:.2f}")

        train_metrics = train_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
            limit_batches=args.limit_batches,
            ema=ema,
        )

        ema.apply_shadow()
        val_metrics, val_predictions_df = evaluate_model(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
            limit_batches=args.limit_batches,
        )
        ema.restore()

        current_learning_rate = optimizer.param_groups[0]["lr"]
        if config.SCHEDULER == "plateau":
            scheduler.step(val_metrics["roc_auc"])
        else:
            scheduler.step()
        epoch_duration = time.time() - epoch_start_time

        epoch_results = {
            "epoch": epoch,
            "learning_rate": current_learning_rate,
            "train_loss": train_metrics["loss"],
            "train_accuracy": train_metrics["accuracy"],
            "train_balanced_accuracy": train_metrics["balanced_accuracy"],
            "train_precision": train_metrics["precision"],
            "train_recall": train_metrics["recall"],
            "train_f1": train_metrics["f1"],
            "train_roc_auc": train_metrics["roc_auc"],
            "validation_loss": val_metrics["loss"],
            "validation_accuracy": val_metrics["accuracy"],
            "validation_balanced_accuracy": val_metrics["balanced_accuracy"],
            "validation_precision": val_metrics["precision"],
            "validation_recall": val_metrics["recall"],
            "validation_f1": val_metrics["f1"],
            "validation_roc_auc": val_metrics["roc_auc"],
        }
        history.append(epoch_results)

        history_df = pd.DataFrame(history)
        history_df.to_csv(history_path, index=False)
        plot_training_curves(history_df, experiment_directory)

        # TensorBoard Logging
        if writer is not None:
            writer.add_scalar("Loss/Train", train_metrics["loss"], epoch)
            writer.add_scalar("Loss/Val", val_metrics["loss"], epoch)
            writer.add_scalar("Accuracy/Train", train_metrics["accuracy"], epoch)
            writer.add_scalar("Accuracy/Val", val_metrics["accuracy"], epoch)
            writer.add_scalar("BalancedAccuracy/Val", val_metrics["balanced_accuracy"], epoch)
            if not np.isnan(val_metrics["roc_auc"]):
                writer.add_scalar("ROC-AUC/Val", val_metrics["roc_auc"], epoch)
            writer.add_scalar("LR/Backbone", current_learning_rate, epoch)

        print(
            f"Epoch {epoch:02d}/{config.NUM_EPOCHS} ({epoch_duration:.1f}s) | "
            f"Train Loss: {train_metrics['loss']:.4f} | "
            f"Val Loss: {val_metrics['loss']:.4f} | "
            f"Val AUC: {val_metrics['roc_auc']:.4f} | "
            f"Val F1: {val_metrics['f1']:.4f}"
        )

        # Save last checkpoint
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "scaler_state_dict": scaler.state_dict() if scaler is not None else None,
                "best_validation_auc": best_validation_auc,
                "best_validation_score": best_validation_score,
                "using_auc_tracking": using_auc_tracking,
                "epochs_without_improvement": epochs_without_improvement,
                "history": history,
            },
            last_checkpoint_path,
        )

        current_auc = val_metrics["roc_auc"]
        current_bal_acc = val_metrics["balanced_accuracy"]

        if not np.isnan(current_auc):
            metric_name = "ROC-AUC"
            current_score = current_auc
            if not using_auc_tracking:
                using_auc_tracking = True
                best_validation_score = -np.inf
            is_best = (best_validation_score == -np.inf) or (current_score > best_validation_score)
        else:
            metric_name = "Balanced Accuracy (Fallback)"
            current_score = current_bal_acc
            if using_auc_tracking and best_validation_score == -np.inf:
                using_auc_tracking = False
                best_validation_score = -np.inf
            is_best = (not using_auc_tracking) and ((best_validation_score == -np.inf) or (current_score > best_validation_score))

        if is_best:
            best_validation_score = current_score
            if not np.isnan(current_auc):
                best_validation_auc = current_auc
            epochs_without_improvement = 0

            # Save best epoch checkpoint
            epoch_ckpt_path = experiment_directory / f"best_model_epoch_{epoch}.pt"
            total_training_time_so_far = time.time() - training_start_time
            
            ema.apply_shadow()
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "model_name": config.MODEL_NAME,
                    "training_strategy": config.TRAINING_STRATEGY,
                    "image_size": config.IMAGE_SIZE,
                    "best_validation_auc": best_validation_auc,
                    "best_validation_score": best_validation_score,
                    "using_auc_tracking": using_auc_tracking,
                    "epoch": epoch,
                    "trainable_parameters": trainable_params,
                    "training_time_seconds": total_training_time_so_far,
                    "configuration": {
                        "batch_size": config.BATCH_SIZE,
                        "backbone_lr": config.BACKBONE_LR,
                        "classifier_lr": config.CLASSIFIER_LR,
                        "weight_decay": config.WEIGHT_DECAY,
                        "seed": config.SEED,
                    },
                },
                epoch_ckpt_path,
            )
            ema.restore()
            print(f"--> Saved best epoch checkpoint to: {epoch_ckpt_path} (score: {best_validation_score:.4f})")

            best_checkpoints.append((best_validation_score, epoch_ckpt_path))
            best_checkpoints.sort(key=lambda x: x[0], reverse=True)

            if len(best_checkpoints) > config.NUM_CHECKPOINTS_TO_AVERAGE:
                _, worst_path = best_checkpoints.pop()
                if worst_path.exists():
                    worst_path.unlink()

            plot_diagnostic_curves(
                val_predictions_df["label"].to_list(),
                val_predictions_df["prob_fake"].to_list(),
                experiment_directory
            )
        else:
            epochs_without_improvement += 1

        # Check Early Stopping Trigger
        if epochs_without_improvement >= config.PATIENCE:
            print(f"--> Early stopping triggered: validation {metric_name} did not improve for {config.PATIENCE} epochs.")
            break

    # Run checkpoint averaging at the end
    if best_checkpoints:
        final_ckpt_paths = [path for _, path in best_checkpoints]
        average_checkpoints(final_ckpt_paths, checkpoint_path, device)
        # Cleanup individual checkpoints
        for _, path in best_checkpoints:
            if path.exists() and path != checkpoint_path:
                path.unlink()

        # Post-averaging threshold calibration (Youden's J on val split)
        # The averaged model may differ from any individual EMA checkpoint,
        # so we re-evaluate on val to find the optimal decision threshold.
        print("\n[+] Running post-averaging Youden's J threshold calibration on val split...")
        try:
            averaged_ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
            model.load_state_dict(averaged_ckpt["model_state_dict"])
            model.eval()
            _, calib_preds_df = evaluate_model(
                model=model,
                loader=val_loader,
                criterion=criterion,
                device=device,
            )
            calib_labels = calib_preds_df["label"].to_numpy().astype(int)
            calib_probs = calib_preds_df["prob_fake"].to_numpy().astype(float)

            # Sweep thresholds and pick the one maximising Youden's J = Sensitivity + Specificity - 1
            from sklearn.metrics import roc_curve
            fpr_vals, tpr_vals, thresh_vals = roc_curve(calib_labels, calib_probs)
            youden_j = tpr_vals - fpr_vals  # J = TPR - FPR
            best_thresh_idx = int(np.argmax(youden_j))
            optimal_threshold = float(thresh_vals[best_thresh_idx])
            best_youden = float(youden_j[best_thresh_idx])
            print(
                f"---> Optimal threshold (Youden's J={best_youden:.4f}): {optimal_threshold:.4f}  "
                f"(vs. fixed 0.5)"
            )

            # Persist the threshold in the checkpoint's configuration dict
            averaged_ckpt.setdefault("configuration", {})
            averaged_ckpt["configuration"]["optimal_threshold"] = optimal_threshold
            averaged_ckpt["configuration"]["threshold_criterion"] = "youdens_j"
            averaged_ckpt["configuration"]["threshold_youden_j"] = best_youden
            torch.save(averaged_ckpt, checkpoint_path)
            print(f"---> Threshold saved to checkpoint: {checkpoint_path}")
        except Exception as calib_err:
            print(f"Warning: Threshold calibration failed ({calib_err}). Checkpoint left at threshold=0.5.")
    else:
        print("Warning: No best checkpoints were saved during training.")

    total_train_time = time.time() - training_start_time
    print(f"Training completed in {total_train_time:.1f}s. Best validation ROC-AUC: {best_validation_auc:.4f}")
    if writer is not None:
        writer.close()


if __name__ == "__main__":
    main()
