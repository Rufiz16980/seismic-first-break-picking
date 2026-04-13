"""Universal Model Evaluator (Phase 4.11).

Works identically for all three model tiers:
  - Tier 0: Classical signal processing (STALTAPicker, MERPicker, AICPicker)
  - Tier 1: Tabular ML (LightGBMWrapper)
  - Tier 2/3: PyTorch DL (any nn.Module)

Produces and logs to MLflow:
  - Scalar metrics: MAE, P50/P90/P95 MAE, within_5ms_pct, within_10ms_pct
  - Per-asset metric breakdowns
  - Inference latency (ms per trace) and throughput (traces/sec)
  - Model parameter count / size estimate
  - Scatter plot: prediction vs ground truth
  - Error distribution histogram
  - Training curves (if history dict provided)
  - All figures saved as MLflow artifacts
"""
import os
import time
import tempfile
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Headless rendering for Colab
import matplotlib.pyplot as plt
from typing import Optional, Dict, Any, List

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


class ModelEvaluator:
    """Universal evaluator compatible with Classical, Tabular, and DL models.

    Args:
        model:      Any model with a `.predict(traces, ...)` or `.forward()` method.
        val_loader: PyTorch DataLoader producing dict batches.
        logger:     MLFlowLogger instance (already in an active run).
        device:     torch.device (ignored for classical/tabular models).
        model_key:  Short model name string (e.g. 'cnn1d') for artifact naming.
        is_dl:      True for PyTorch nn.Module, False for classical/tabular.
        history:    Optional dict {'train_mae': [...], 'val_mae': [...], ...} from Trainer.
    """

    def __init__(
        self,
        model,
        val_loader,
        logger,
        device=None,
        model_key: str = "model",
        is_dl: bool = True,
        history: Optional[Dict[str, List[float]]] = None,
    ):
        self.model = model
        self.val_loader = val_loader
        self.logger = logger
        self.device = device
        self.model_key = model_key
        self.is_dl = is_dl
        self.history = history or {}

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------
    def run(self) -> Dict[str, float]:
        """Run the full evaluation pipeline. Returns the scalar metrics dict."""
        print(f"\n{'='*60}")
        print(f"MODEL EVALUATOR: {self.model_key}")
        print(f"{'='*60}")

        all_preds, all_labels, all_assets, all_latencies = self._run_inference()

        if len(all_preds) == 0:
            print("WARNING: No valid labeled traces in val set. Skipping metrics.")
            return {}

        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        all_assets = np.array(all_assets)

        # --- Scalar Metrics ---
        metrics = self._compute_metrics(all_preds, all_labels)

        # --- Latency Metrics ---
        if all_latencies:
            metrics["inference_latency_ms_per_trace"] = float(np.mean(all_latencies))
            metrics["inference_throughput_traces_per_sec"] = float(
                1000.0 / (np.mean(all_latencies) + 1e-9)
            )
        metrics["n_params"] = float(self._count_params())

        # --- Per-asset Metrics ---
        asset_metrics = self._compute_per_asset(all_preds, all_labels, all_assets)

        # --- Print Results ---
        self._print_results(metrics, asset_metrics)

        # --- Generate & Log Figures ---
        with tempfile.TemporaryDirectory() as tmpdir:
            # 1. Scatter: Predicted vs Ground Truth
            fig_scatter = self._plot_scatter(all_preds, all_labels)
            scatter_path = os.path.join(tmpdir, f"{self.model_key}_scatter.png")
            fig_scatter.savefig(scatter_path, dpi=150, bbox_inches="tight")
            plt.close(fig_scatter)

            # 2. Error Histogram
            fig_hist = self._plot_error_hist(all_preds, all_labels)
            hist_path = os.path.join(tmpdir, f"{self.model_key}_error_hist.png")
            fig_hist.savefig(hist_path, dpi=150, bbox_inches="tight")
            plt.close(fig_hist)

            # 3. Per-asset Bar Chart
            fig_asset = self._plot_asset_bars(asset_metrics)
            asset_path = os.path.join(tmpdir, f"{self.model_key}_per_asset_mae.png")
            fig_asset.savefig(asset_path, dpi=150, bbox_inches="tight")
            plt.close(fig_asset)

            # 4. Training Curves (if history available)
            if self.history:
                fig_curves = self._plot_training_curves()
                curves_path = os.path.join(tmpdir, f"{self.model_key}_training_curves.png")
                fig_curves.savefig(curves_path, dpi=150, bbox_inches="tight")
                plt.close(fig_curves)
                self.logger.log_artifact(curves_path, "plots")

            # Log all figures as MLflow artifacts
            self.logger.log_artifact(scatter_path, "plots")
            self.logger.log_artifact(hist_path, "plots")
            self.logger.log_artifact(asset_path, "plots")

        # --- Log scalar metrics to MLflow ---
        log_metrics = {**metrics}
        for asset, am in asset_metrics.items():
            for k, v in am.items():
                log_metrics[f"{asset}_{k}"] = v

        self.logger.log_metrics(log_metrics, step=0)
        print(f"\nAll metrics and plots logged to MLflow.")
        return metrics

    # ------------------------------------------------------------------
    # Inference loop (unified for all model tiers)
    # ------------------------------------------------------------------
    def _run_inference(self):
        """Iterate val_loader, run model inference, collect predictions."""
        all_preds, all_labels, all_assets, all_latencies = [], [], [], []

        if self.is_dl and TORCH_AVAILABLE:
            self.model.eval()

        for batch in self.val_loader:
            # Dict batch from ShotGatherDataset
            if isinstance(batch, (list, tuple)):
                # Fallback for tuple-style batches
                traces_t = batch[0]
                labels_ms_t = batch[1]
                valid_mask_t = batch[2]
                batch_assets = ["unknown"] * traces_t.shape[0]
            else:
                # Both collate_fn variants emit 'valid_mask' and 'assets'.
                # For gather models (2D), variable_width_collate_fn also emits
                # 'label_mask' (True only for labeled traces). We MUST prefer
                # label_mask here so that unlabeled Sudbury traces (labels_ms=NaN)
                # are excluded from metric computations. For trace models (1D),
                # trace_collate_fn pre-filters, so valid_mask == label_mask and
                # the fallback is safe.
                traces_t = batch["traces"]
                labels_ms_t = batch["labels_ms"]
                valid_mask_t = batch.get("label_mask", batch["valid_mask"])
                batch_assets = batch.get("assets", ["unknown"] * traces_t.shape[0])

            # Numpy for classical/tabular; keep as tensor for DL
            traces_np = traces_t.squeeze(1).numpy()  # [B, 751]
            labels_np = labels_ms_t.numpy()           # [B] or [B, W]
            mask_np = valid_mask_t.numpy()             # [B] or [B, W]

            if not mask_np.any():
                continue

            # --- Inference with latency timing ---
            t0 = time.perf_counter()

            if self.is_dl and TORCH_AVAILABLE:
                with torch.no_grad():
                    # DL models may return [B] or [B, W] tensors
                    x = traces_t.to(self.device).float()
                    preds_t = self.model(x)
                    preds_np = preds_t.cpu().numpy()
            else:
                # Classical / Tabular: expects [B, n_samples] numpy
                preds_np = self.model.predict(traces_np)  # [B] for classical

            t1 = time.perf_counter()

            n_traces = int(mask_np.sum())
            if n_traces > 0:
                latency_ms = (t1 - t0) * 1000.0
                per_trace_ms = latency_ms / n_traces
                all_latencies.append(per_trace_ms)

            # --- Flatten per-trace results ---
            if preds_np.ndim == 1:
                # Classical output: one prediction per gather — skip trace-level comparison
                # We broadcast as "all traces in the gather share one prediction" for
                # computing a comparable MAE against per-trace labels
                for b in range(traces_np.shape[0]):
                    trace_mask = mask_np[b] if mask_np.ndim == 2 else np.array([mask_np[b]])
                    trace_labels = labels_np[b] if labels_np.ndim == 2 else np.array([labels_np[b]])
                    for i, valid in enumerate(trace_mask.flat):
                        if valid:
                            all_preds.append(float(preds_np[b]))
                            all_labels.append(float(trace_labels.flat[i]))
                            all_assets.append(
                                batch_assets[b] if isinstance(batch_assets, (list, np.ndarray)) else "unknown"
                            )
            else:
                # DL output [B, W]: one prediction per trace
                flat_preds = preds_np.reshape(-1)
                flat_labels = labels_np.reshape(-1)
                flat_mask = mask_np.reshape(-1).astype(bool)
                flat_assets = []
                if labels_np.ndim == 2:
                    for b in range(traces_np.shape[0]):
                        flat_assets.extend([batch_assets[b]] * labels_np.shape[1])
                else:
                    flat_assets = list(batch_assets)

                all_preds.extend(flat_preds[flat_mask].tolist())
                all_labels.extend(flat_labels[flat_mask].tolist())
                all_assets.extend(np.array(flat_assets)[flat_mask].tolist())

        return all_preds, all_labels, all_assets, all_latencies

    # ------------------------------------------------------------------
    # Metric computations
    # ------------------------------------------------------------------
    def _compute_metrics(self, preds: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
        errors = np.abs(preds - labels)
        return {
            "val_mae_ms":         float(np.mean(errors)),
            "val_mae_p50_ms":     float(np.percentile(errors, 50)),
            "val_mae_p90_ms":     float(np.percentile(errors, 90)),
            "val_mae_p95_ms":     float(np.percentile(errors, 95)),
            "val_mae_p99_ms":     float(np.percentile(errors, 99)),
            "val_within_5ms_pct": float(np.mean(errors <= 5.0) * 100),
            "val_within_10ms_pct": float(np.mean(errors <= 10.0) * 100),
            "val_within_20ms_pct": float(np.mean(errors <= 20.0) * 100),
            "val_n_traces":       float(len(errors)),
        }

    def _compute_per_asset(
        self, preds: np.ndarray, labels: np.ndarray, assets: np.ndarray
    ) -> Dict[str, Dict[str, float]]:
        result = {}
        for asset in np.unique(assets):
            mask = assets == asset
            errors = np.abs(preds[mask] - labels[mask])
            result[asset] = {
                "mae_ms": float(np.mean(errors)),
                "within_5ms_pct": float(np.mean(errors <= 5.0) * 100),
                "n_traces": float(len(errors)),
            }
        return result

    def _count_params(self) -> int:
        """Count trainable parameters for DL models; returns -1 otherwise."""
        if self.is_dl and TORCH_AVAILABLE and hasattr(self.model, "parameters"):
            return sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        return -1

    # ------------------------------------------------------------------
    # Plotting
    # ------------------------------------------------------------------
    def _plot_scatter(self, preds: np.ndarray, labels: np.ndarray) -> plt.Figure:
        fig, ax = plt.subplots(figsize=(7, 7))
        max_val = max(preds.max(), labels.max()) * 1.05
        ax.scatter(labels, preds, s=1, alpha=0.2, color="#4C72B0")
        ax.plot([0, max_val], [0, max_val], "r--", lw=1.5, label="Perfect")
        ax.set_xlabel("Ground Truth (ms)")
        ax.set_ylabel("Predicted (ms)")
        ax.set_title(f"{self.model_key} — Predicted vs Ground Truth")
        ax.legend()
        ax.set_xlim(0, max_val)
        ax.set_ylim(0, max_val)
        fig.tight_layout()
        return fig

    def _plot_error_hist(self, preds: np.ndarray, labels: np.ndarray) -> plt.Figure:
        errors = np.abs(preds - labels)
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.hist(errors, bins=100, color="#4C72B0", edgecolor="none", alpha=0.8)
        ax.axvline(5.0, color="red", linestyle="--", label="5ms threshold")
        ax.axvline(10.0, color="orange", linestyle="--", label="10ms threshold")
        ax.axvline(np.mean(errors), color="green", linestyle="-", lw=2,
                   label=f"Mean MAE={np.mean(errors):.1f}ms")
        ax.set_xlabel("Absolute Error (ms)")
        ax.set_ylabel("Count")
        ax.set_title(f"{self.model_key} — Error Distribution")
        ax.legend()
        fig.tight_layout()
        return fig

    def _plot_asset_bars(self, asset_metrics: Dict[str, Dict[str, float]]) -> plt.Figure:
        assets = list(asset_metrics.keys())
        maes = [asset_metrics[a]["mae_ms"] for a in assets]
        pcts = [asset_metrics[a]["within_5ms_pct"] for a in assets]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
        colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]

        ax1.bar(assets, maes, color=colors[: len(assets)])
        ax1.set_ylabel("MAE (ms)")
        ax1.set_title(f"{self.model_key} — Per-Asset MAE")

        ax2.bar(assets, pcts, color=colors[: len(assets)])
        ax2.set_ylabel("% within 5ms")
        ax2.set_ylim(0, 100)
        ax2.set_title(f"{self.model_key} — Per-Asset Within-5ms")

        fig.tight_layout()
        return fig

    def _plot_training_curves(self) -> plt.Figure:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        # Loss curves
        ax = axes[0]
        if "train_loss" in self.history:
            ax.plot(self.history["train_loss"], label="Train Loss", color="#4C72B0")
        if "val_loss" in self.history:
            ax.plot(self.history["val_loss"], label="Val Loss", color="#DD8452")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_title(f"{self.model_key} — Loss Curves")
        ax.legend()

        # MAE curves
        ax = axes[1]
        if "train_mae" in self.history:
            ax.plot(self.history["train_mae"], label="Train MAE", color="#4C72B0")
        if "val_mae" in self.history:
            ax.plot(self.history["val_mae"], label="Val MAE", color="#DD8452")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("MAE (ms)")
        ax.set_title(f"{self.model_key} — MAE Curves")
        ax.legend()

        fig.tight_layout()
        return fig

    # ------------------------------------------------------------------
    # Console summary
    # ------------------------------------------------------------------
    def _print_results(self, metrics: Dict[str, float], asset_metrics: Dict):
        print(f"\n{'─'*50}")
        print(f"  VALIDATION METRICS")
        print(f"{'─'*50}")
        print(f"  MAE (mean):      {metrics['val_mae_ms']:.2f} ms")
        print(f"  MAE P50:         {metrics['val_mae_p50_ms']:.2f} ms")
        print(f"  MAE P90:         {metrics['val_mae_p90_ms']:.2f} ms")
        print(f"  MAE P95:         {metrics['val_mae_p95_ms']:.2f} ms")
        print(f"  Within   5ms:    {metrics['val_within_5ms_pct']:.1f}%")
        print(f"  Within  10ms:    {metrics['val_within_10ms_pct']:.1f}%")
        print(f"  Within  20ms:    {metrics['val_within_20ms_pct']:.1f}%")
        n_params = int(metrics.get("n_params", -1))
        if n_params > 0:
            print(f"  Parameters:      {n_params:,}")
        if "inference_latency_ms_per_trace" in metrics:
            print(f"  Latency/trace:   {metrics['inference_latency_ms_per_trace']:.3f} ms")
            print(f"  Throughput:      {metrics['inference_throughput_traces_per_sec']:.0f} traces/s")
        print(f"\n  PER-ASSET BREAKDOWN")
        for asset, am in asset_metrics.items():
            print(f"  {asset:12s} | MAE: {am['mae_ms']:6.1f} ms | <=5ms: {am['within_5ms_pct']:5.1f}%  ({int(am['n_traces'])} traces)")
        print(f"{'─'*50}")
