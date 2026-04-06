"""
training/trainer.py
Training loop với:
- Label Smoothing Cross-Entropy
- AdamW optimizer
- CosineAnnealingLR scheduler
- Gradient clipping
- Auto save best checkpoint
- Early stopping
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
from pathlib import Path
import time


class LabelSmoothingCE(nn.Module):
    def __init__(self, n_classes: int, smoothing: float = 0.1):
        super().__init__()
        self.smoothing = smoothing
        self.n = n_classes

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        confidence = 1.0 - self.smoothing
        smooth_val = self.smoothing / (self.n - 1)
        one_hot = torch.full_like(pred, smooth_val)
        one_hot.scatter_(1, target.unsqueeze(1), confidence)
        log_prob = torch.log_softmax(pred, dim=1)
        return -(one_hot * log_prob).sum(dim=1).mean()


class VSLTrainer:
    def __init__(self,
                 model,
                 train_ds,
                 val_ds,
                 n_classes:      int,
                 lr:             float = 1e-3,
                 weight_decay:   float = 1e-4,
                 epochs:         int   = 80,
                 batch_size:     int   = 16,
                 grad_clip:      float = 1.0,
                 label_smoothing:float = 0.1,
                 early_stop:     int   = 15,
                 device:         str   = 'cuda',
                 save_dir:       str   = 'weights'):

        self.model    = model.to(device)
        self.device   = device
        self.epochs   = epochs
        self.grad_clip= grad_clip
        self.early_stop_patience = early_stop
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True)
        self.model_name = model.__class__.__name__

        self.train_loader = DataLoader(
            train_ds, batch_size=batch_size,
            shuffle=True, num_workers=4,
            pin_memory=(device=='cuda'), drop_last=True
        )
        self.val_loader = DataLoader(
            val_ds, batch_size=batch_size,
            shuffle=False, num_workers=2,
            pin_memory=(device=='cuda')
        )

        self.criterion = LabelSmoothingCE(n_classes, label_smoothing)
        self.optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=epochs, eta_min=1e-5)

        self.best_acc   = 0.0
        self.no_improve = 0
        self.history    = {'train_loss':[], 'train_acc':[], 'val_loss':[], 'val_acc':[]}

    def train(self):
        print(f"\n{'='*60}")
        print(f"  Training {self.model_name}")
        print(f"  Device : {self.device} | Epochs: {self.epochs}")
        print(f"  Train  : {len(self.train_loader.dataset)} samples")
        print(f"  Val    : {len(self.val_loader.dataset)} samples")
        print(f"{'='*60}\n")

        start_time = time.time()

        for epoch in range(1, self.epochs + 1):
            t0 = time.time()

            train_loss, train_acc = self._run_epoch(train=True)
            val_loss,   val_acc   = self._run_epoch(train=False)
            self.scheduler.step()

            # Log
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)

            elapsed  = time.time() - t0
            is_best  = val_acc > self.best_acc

            if is_best:
                self.best_acc   = val_acc
                self.no_improve = 0
                torch.save({
                    'epoch':       epoch,
                    'model_state': self.model.state_dict(),
                    'val_acc':     val_acc,
                    'optimizer':   self.optimizer.state_dict(),
                }, self.save_dir / f"{self.model_name}_best.pth")
            else:
                self.no_improve += 1

            marker = " ★ BEST" if is_best else ""
            lr_now = self.optimizer.param_groups[0]['lr']
            print(f"Epoch {epoch:3d}/{self.epochs} | "
                  f"lr={lr_now:.5f} | "
                  f"train {train_loss:.4f}/{train_acc:.2%} | "
                  f"val {val_loss:.4f}/{val_acc:.2%} | "
                  f"{elapsed:.1f}s{marker}")

            # Early stopping
            if self.no_improve >= self.early_stop_patience:
                print(f"\nEarly stopping tại epoch {epoch} "
                      f"(best val_acc={self.best_acc:.2%})")
                break

        total = time.time() - start_time
        print(f"\n✓ Done! Best val acc: {self.best_acc:.2%} | "
              f"Thời gian: {total/60:.1f} phút")
        print(f"  Checkpoint: {self.save_dir / f'{self.model_name}_best.pth'}")
        return self.history

    def _run_epoch(self, train: bool = True):
        self.model.train(train)
        loader = self.train_loader if train else self.val_loader

        total_loss = 0.0
        correct    = 0
        total      = 0

        with torch.set_grad_enabled(train):
            for x, y in loader:
                x = x.to(self.device, non_blocking=True)
                y = y.to(self.device, non_blocking=True)

                out  = self.model(x)
                loss = self.criterion(out, y)

                if train:
                    self.optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.grad_clip)
                    self.optimizer.step()

                bs = y.size(0)
                total_loss += loss.item() * bs
                correct    += (out.argmax(1) == y).sum().item()
                total      += bs

        return total_loss / total, correct / total
