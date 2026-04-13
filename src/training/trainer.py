import os
import torch
import torch.nn as nn
from typing import Dict, Any, Optional
from tqdm import tqdm

from src.training.mlflow_logger import MLFlowLogger
from src.models.metrics import calculate_metrics

class Trainer:
    """Core PyTorch Training Loop handling AMP, clipping, and checkpointing."""
    
    def __init__(self, 
                 model: nn.Module,
                 optimizer: torch.optim.Optimizer,
                 criterion: nn.Module,
                 config: Any,
                 device: torch.device,
                 scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
                 logger: Optional[MLFlowLogger] = None):
                 
        self.model = model.to(device)
        self.optimizer = optimizer
        self.criterion = criterion
        self.scheduler = scheduler
        self.config = config
        self.device = device
        self.logger = logger
        
        # Gradient accumulation limit (default 1)
        self.grad_accum_steps = getattr(self.config.training, 'gradient_accumulation_steps', 1)
        
        self.scaler = torch.amp.GradScaler('cuda', enabled=(device.type == 'cuda'))
        
        # Checkpointing state
        self.best_val_mae = float('inf')
        self.start_epoch = 1
        self.patience_counter = 0
        self.early_stop_triggered = False
        
        # History accumulation for training curves
        self.history = {"train_loss": [], "train_mae": [], "val_loss": [], "val_mae": []}
        
        # Progressive-training state (written by Cell 7 after each asset completes)
        self.training_state = {}
        
        # Prepare output directory
        self.checkpoint_dir = getattr(config.output, 'checkpoint_dir', './checkpoints/')
        if self.checkpoint_dir.startswith("file://"):
            self.checkpoint_dir = self.checkpoint_dir.replace("file://", "")
        os.makedirs(self.checkpoint_dir, exist_ok=True)

    def _save_checkpoint(self, epoch: int, is_best: bool, filename: str):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'scaler_state_dict': self.scaler.state_dict(),
            'best_val_mae': self.best_val_mae,
            'config': self.config.__dict__ if hasattr(self.config, '__dict__') else self.config,
            # Persisted across session interruptions
            'history': self.history,
            'training_state': self.training_state,
        }
        
        filepath = os.path.join(self.checkpoint_dir, filename)
        torch.save(checkpoint, filepath)
        
        if is_best:
            best_filepath = os.path.join(self.checkpoint_dir, f"{self.config.experiment.name}_best.pt")
            torch.save(checkpoint, best_filepath)
        
        # Write a sidecar file with the active MLflow run_id so that Cell 2 can
        # resume THE SAME run after a Colab disconnect instead of opening a new one.
        if self.logger is not None and getattr(self.logger, 'active_run', None) is not None:
            run_id_file = os.path.join(self.checkpoint_dir, 'mlflow_run_id.txt')
            with open(run_id_file, 'w') as _f:
                _f.write(self.logger.active_run.info.run_id)

    def load_checkpoint(self, path: str):
        if not os.path.isfile(path):
            print(f"No checkpoint found at {path}")
            return
            
        print(f"Loading checkpoint from {path}")
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if self.scheduler and checkpoint['scheduler_state_dict']:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            
        self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        self.start_epoch = checkpoint['epoch'] + 1
        self.best_val_mae = checkpoint.get('best_val_mae', float('inf'))
        # Restore training curves so plots reflect the full run, not just the resumed portion
        self.history = checkpoint.get(
            'history', {"train_loss": [], "train_mae": [], "val_loss": [], "val_mae": []}
        )
        # Restore progressive-training state so Cell 3 can correctly skip completed assets
        self.training_state = checkpoint.get('training_state', {})
        print(f"Resumed from epoch {checkpoint['epoch']} with Best MAE: {self.best_val_mae:.4f}")
        print(f"  Restored {len(self.history['train_loss'])} history entries.")

    def train_epoch(self, dataloader) -> Dict[str, float]:
        self.model.train()
        total_loss = 0.0
        
        # We track metrics over the epoch
        all_preds, all_targets, all_masks = [], [], []
        
        pbar = tqdm(dataloader, desc="Training")
        
        # Ensure gradients are zeroed before we start accumulating
        self.optimizer.zero_grad(set_to_none=True)
        
        for batch_idx, batch in enumerate(pbar):
            # Dict batch from ShotGatherDataset via collate_fn
            if isinstance(batch, (list, tuple)):
                x = batch[0].to(self.device).float()
                y = batch[1].to(self.device).float()
                mask = batch[2].to(self.device).bool()
            else:
                x = batch["traces"].to(self.device).float()
                y = batch["labels_ms"].to(self.device).float()
                # label_mask: True only for traces with ground-truth labels.
                # valid_mask: True for all real (non-padded) traces.
                # For 2D gather models, valid_mask includes unlabeled Sudbury
                # traces whose labels_ms = NaN — we must use label_mask.
                # For 1D trace models, trace_collate_fn pre-filters so both
                # masks are equivalent; fallback to valid_mask is safe.
                mask = batch.get("label_mask", batch["valid_mask"]).to(self.device).bool()
            
            # Forward pass with AMP
            with torch.amp.autocast('cuda', enabled=(self.device.type == 'cuda')):
                preds_ms = self.model(x)
                loss = self.criterion(preds_ms, y, mask)
                # Scale loss by accumulation steps
                scaled_loss = loss / self.grad_accum_steps
                
            if mask.any():
                # Backward
                self.scaler.scale(scaled_loss).backward()
                
            # Perform optimization step only after accumulating 'grad_accum_steps' batches
            # or if this is the very last batch in the epoch
            if (batch_idx + 1) % self.grad_accum_steps == 0 or (batch_idx + 1) == len(dataloader):
                # Gradient Clipping
                self.scaler.unscale_(self.optimizer)
                max_norm = getattr(self.config.training, 'gradient_clip_norm', 1.0)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=max_norm)
                
                self.scaler.step(self.optimizer)
                # Update the scale state unconditionally so elapsed iteration counts structurally map safely
                self.scaler.update()
                
                self.optimizer.zero_grad(set_to_none=True)
                
            total_loss += loss.item()
            
            all_preds.append(preds_ms.detach())
            all_targets.append(y.detach())
            all_masks.append(mask.detach())
            
            pbar.set_postfix({'loss': loss.item()})
            
        avg_loss = total_loss / len(dataloader)
        
        # Calculate train metrics globally for the epoch
        cat_preds = torch.cat([p.view(-1) for p in all_preds])
        cat_targets = torch.cat([t.view(-1) for t in all_targets])
        cat_masks = torch.cat([m.view(-1) for m in all_masks])
        
        metrics = calculate_metrics(cat_preds, cat_targets, cat_masks)
        metrics['loss'] = avg_loss
        return metrics

    @torch.no_grad()
    def validate_epoch(self, dataloader) -> Dict[str, float]:
        self.model.eval()
        total_loss = 0.0
        
        all_preds, all_targets, all_masks = [], [], []
        
        for batch in dataloader:
            if isinstance(batch, (list, tuple)):
                x = batch[0].to(self.device).float()
                y = batch[1].to(self.device).float()
                mask = batch[2].to(self.device).bool()
            else:
                x = batch["traces"].to(self.device).float()
                y = batch["labels_ms"].to(self.device).float()
                # Same label_mask-first logic as train_epoch — see comment there.
                mask = batch.get("label_mask", batch["valid_mask"]).to(self.device).bool()
            
            with torch.amp.autocast('cuda', enabled=(self.device.type == 'cuda')):
                preds_ms = self.model(x)
                loss = self.criterion(preds_ms, y, mask)
                
            total_loss += loss.item()
            all_preds.append(preds_ms)
            all_targets.append(y)
            all_masks.append(mask)
            
        avg_loss = total_loss / len(dataloader)
        
        cat_preds = torch.cat([p.view(-1) for p in all_preds])
        cat_targets = torch.cat([t.view(-1) for t in all_targets])
        cat_masks = torch.cat([m.view(-1) for m in all_masks])
        
        metrics = calculate_metrics(cat_preds, cat_targets, cat_masks)
        metrics['loss'] = avg_loss
        return metrics

    def run(self, train_loader, val_loader,
            start_epoch: int = None, total_epochs: int = None):
        """Execute the training loop.

        Args:
            train_loader: DataLoader for training data.
            val_loader:   DataLoader for validation data.
            start_epoch:  Last *completed* epoch (0 = fresh start).  When
                          provided it overrides self.start_epoch so that the
                          progressive training loop in the notebook can resume
                          from the correct position without calling
                          load_checkpoint() explicitly.
            total_epochs: Total epochs for this training phase.  Defaults to
                          config.training.epochs.  Used by progressive mode to
                          train each asset for a different number of epochs.
        """
        max_epochs = total_epochs if total_epochs is not None else getattr(self.config.training, 'epochs', 100)
        # start_epoch semantics: last completed epoch (0 = none completed yet).
        # self.start_epoch defaults to 1 (set in __init__) or epoch+1 (from load_checkpoint).
        if start_epoch is not None:
            self.start_epoch = start_epoch + 1   # resume AFTER the specified epoch
        patience = getattr(self.config.training, 'early_stopping_patience', 15)
        
        for epoch in range(self.start_epoch, max_epochs + 1):
            print(f"\n--- Epoch {epoch}/{max_epochs} ---")
            
            # Train
            train_metrics = self.train_epoch(train_loader)
            print(f"Train | Loss: {train_metrics['loss']:.4f} | MAE: {train_metrics['mae_ms']:.2f}ms")
            
            # Validate
            val_metrics = self.validate_epoch(val_loader)
            val_mae = val_metrics['mae_ms']
            print(f"Val   | Loss: {val_metrics['loss']:.4f} | MAE: {val_mae:.2f}ms | In-5ms: {val_metrics['within_5ms_pct']:.2f}%")
            
            # LR Scheduler (per epoch)
            if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                self.scheduler.step(val_mae)
            elif self.scheduler:
                self.scheduler.step()
                
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # MLFlow logging
            if self.logger:
                self.logger.log_metrics({
                    "train_loss": train_metrics['loss'],
                    "train_mae": train_metrics['mae_ms'],
                    "val_loss": val_metrics['loss'],
                    "val_mae": val_mae,
                    "val_within_5ms": val_metrics['within_5ms_pct'],
                    "lr": current_lr
                }, step=epoch)
            
            # Accumulate history for curve plotting
            self.history["train_loss"].append(train_metrics['loss'])
            self.history["train_mae"].append(train_metrics['mae_ms'])
            self.history["val_loss"].append(val_metrics['loss'])
            self.history["val_mae"].append(val_mae)
            
            # Checkpointing & Early Stopping
            is_best = val_mae < self.best_val_mae
            if is_best:
                self.best_val_mae = val_mae
                self.patience_counter = 0
                print(f"---> New Best MAE! Saving checkpoint.")
            else:
                self.patience_counter += 1
                
            self._save_checkpoint(epoch, is_best, f"{self.config.experiment.name}_latest.pt")
                
            if epoch % 10 == 0:
                self._save_checkpoint(epoch, False, f"{self.config.experiment.name}_epoch_{epoch}.pt")
                
            if self.patience_counter >= patience:
                print(f"Early stopping triggered after {patience} epochs without improvement.")
                self.early_stop_triggered = True
                break
                
        # End of training sequence
        if self.logger:
            best_model_path = os.path.join(self.checkpoint_dir, f"{self.config.experiment.name}_best.pt")
            if os.path.exists(best_model_path):
                self.logger.log_artifact(best_model_path)
