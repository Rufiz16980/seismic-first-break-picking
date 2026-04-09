# Phase 5 — Benchmarking, Error Analysis & Final Model Selection

This phase begins after all models from Phase 3 have completed training. Its goal is not just to rank models by a single number, but to deeply understand where each model succeeds and fails, surgically improve the best candidate, and produce a defensible final result.

---

## 5.1 — The Benchmarking Notebook Structure

Create a dedicated notebook `04_benchmark_and_compare.ipynb`. This notebook never trains anything. It only loads finished models, runs evaluation, and produces analysis. Keeping it separate from training notebooks means you can re-run it at any time without risk of overwriting checkpoints or triggering training.

---

### 5.1.1 — Loading All Models for Comparison

At the top of the benchmarking notebook, build a model registry — a dict that maps experiment names to their checkpoint paths and configs. Load this programmatically from `benchmark_results.csv` rather than hardcoding paths. Any model whose checkpoint file exists on Drive is automatically included in the comparison. Any model whose checkpoint is missing is skipped with a warning, not a crash.

For each model in the registry, instantiate the model architecture from its saved config, load the state dict from `checkpoint_best_overall.pt`, set to `eval()` mode, and move to the available device. Hold all loaded models in memory simultaneously only if VRAM permits — otherwise evaluate one at a time and store results to disk between models.

---

### 5.1.2 — Standardized Evaluation Protocol

Every model is evaluated using an identical protocol. Deviating from this for any model — even slightly — invalidates the comparison.

**The protocol:**

Run inference on the full test split of each asset independently. For each test shot gather, call `model.predict()` which always returns first break times in ms regardless of framing. Compute all metrics defined in Phase 3.8. Store per-trace predictions and ground truth for every test gather to a results NPZ file named `{experiment_name}_test_predictions.npz`. This file enables all downstream analysis without re-running inference.

**Inference batching:**
Even during evaluation, process gathers in batches of 8 to avoid OOM. Use `torch.no_grad()` and disable AMP during evaluation. Clear the CUDA cache between models.

**Timing:**
Record inference time per gather in milliseconds. A model that achieves 4ms MAE but takes 10 seconds per gather is not deployable. Log this to MLflow and include it in the benchmark table.

---

## 5.2 — The Master Benchmark Table

After all models are evaluated, produce the master comparison table. This is the central artifact of Phase 5.

---

### 5.2.1 — Table Structure

The table has one row per model and the following columns grouped into sections:

**Identity:** experiment name, architecture, framing, pretrained backbone, parameter count in millions, training time in hours.

**Overall performance:** test MAE (ms), test RMSE (ms), within-5ms accuracy (%), within-10ms accuracy (%).

**Per-asset MAE:** brunswick, halfmile, lalor, sudbury — four separate columns. This is the most diagnostic section of the table. A model that averages 4ms overall but achieves 2ms on three assets and 10ms on one has a serious generalization problem that the average conceals.

**Robustness indicators:** MAE on high-SNR traces only, MAE on low-SNR traces only, MAE on near-offset traces (bottom quartile of offset), MAE on far-offset traces (top quartile of offset). These four columns reveal where each model breaks down structurally.

**Practical indicators:** inference time per gather (ms), peak VRAM during inference (MB), whether the model supports batch inference.

Render this table in the notebook using pandas styling — color-code each cell on a green-to-red gradient within its column so the best and worst values are immediately visually obvious. Save the styled table as an HTML file to Drive.

---

### 5.2.2 — Statistical Significance Testing

A difference of 0.3ms MAE between two models might be noise rather than a real difference. Before declaring one model better than another, verify the difference is statistically significant.

For any two models you want to compare, run a **paired Wilcoxon signed-rank test** on their per-trace absolute errors across the full test set. This test is appropriate because errors are not normally distributed (they are skewed and bounded below by zero) and you have paired observations (both models evaluated on the same traces).

If the p-value is above 0.05, you cannot claim one model is better than the other — report them as equivalent. This matters especially for your final model selection decision. A ResNet-UNet that is 0.2ms better than a plain U-Net with p=0.3 is not actually better — it is just noisily different.

Save p-values for all pairwise comparisons between your top 5 models to a matrix CSV. This is also the kind of rigorous detail that distinguishes a professional analysis.

---

## 5.3 — Error Analysis

Ranking models by MAE tells you which model won. Error analysis tells you why it won and what to do next. This is where you learn enough to make targeted improvements rather than guessing what to try.

---

### 5.3.1 — Error Distribution Analysis

For your best-performing model, compute the distribution of per-trace errors (predicted minus ground truth, signed) across the entire test set.

**What to look for and what it means:**

A distribution centered near zero with light tails indicates a well-calibrated model with no systematic bias. This is what you want.

A distribution shifted positive (model consistently predicts too late) or negative (too early) indicates a systematic bias. Late predictions often indicate the model is picking the peak of the first arrival rather than its onset. Early predictions often indicate the model is picking a noise burst before the true arrival. Both are fixable.

A distribution with heavy tails (large kurtosis) indicates the model performs well on most traces but catastrophically fails on a subset. Understanding what makes those traces hard is more valuable than any hyperparameter tuning.

A bimodal distribution (two separate peaks) indicates the model has two qualitatively different failure modes — for example, it correctly picks the first break on direct waves but systematically misidentifies refracted waves as the first arrival.

**Plot:** For each asset separately, plot the signed error distribution as a histogram with KDE overlay. Mark the mean and median with vertical lines. If mean and median differ by more than 1ms, you have a skewed error distribution worth investigating.

---

### 5.3.2 — Spatial Error Analysis

Plot prediction errors geographically. For each test trace, you have the receiver coordinates (REC_X, REC_Y) and the signed error. Create a scatter plot of receiver positions colored by error magnitude. Spatial clustering of high errors reveals acquisition geometry problems — areas where the ground model is more complex, where surface conditions caused poor coupling, or where the recording equipment had issues.

If you see spatial clusters of high error, cross-reference with the raw gather images for those areas. Are those gathers visually noisier? Do they have dead traces? Is the first break curve discontinuous there? Understanding the spatial pattern of failures guides your data augmentation strategy — if errors cluster in high-noise areas, more aggressive noise augmentation during training may help.

---

### 5.3.3 — Failure Mode Taxonomy

Manually inspect the 50 worst predictions from your best model across the test set. For each failure case, load the shot gather, overlay ground truth and prediction, and categorize the failure. You will likely find that nearly all failures fall into a small number of categories:

**Category A — Noise burst misidentification:**
The model picks a high-amplitude noise burst occurring before the true first arrival. Visible in the gather as an isolated amplitude spike unrelated to the main wavefront.

**Category B — Cycle skip:**
The model picks one cycle (wavelength) late or early relative to the true onset. Visible as a prediction that is consistently offset by approximately one period of the dominant frequency.

**Category C — Dead or clipped trace spillover:**
A dead or heavily clipped trace causes the model to make a bad prediction on its neighbors. The error is spatially isolated to a few adjacent traces around a bad channel.

**Category D — Complex wavefront geometry:**
In areas with strong lateral velocity variations, the first break curve in the gather is non-monotonic or has sharp kinks. The model predicts a smooth curve and fails at the kinks.

**Category E — Ground truth label error:**
The model's prediction is actually more physically consistent than the ground truth label. This is surprisingly common — human pickers make mistakes, especially on low-SNR traces. These are false failures that actually indicate the model is working correctly.

For each category, record the count and percentage of failures it represents. This taxonomy directly informs your improvement strategy in Section 5.5.

---

### 5.3.4 — Offset-Stratified Error Analysis

Divide your test traces into 10 offset bins (deciles). For each bin, compute MAE. Plot MAE as a function of median offset in that bin.

Expected pattern: MAE is higher at very short offsets (near-offset traces see the source noise burst and the first break is poorly defined), MAE decreases through mid-offsets (where first breaks are clearest), then MAE increases again at far offsets (where signal amplitude decreases with distance and the first break merges with noise).

If your model deviates from this pattern — for example, if MAE increases monotonically with offset — it suggests the model has not learned the physics of wave propagation. In that case, feeding offset as an auxiliary input feature to the model (as discussed in Phase 3.3.1) would likely yield significant improvement.

---

### 5.3.5 — Cross-Asset Generalization Analysis

Train your best model on three assets and test on the held-out fourth. Repeat for all four combinations. This 4-fold cross-asset experiment answers the question: if this model is deployed on a new seismic survey it has never seen, how well does it generalize?

This analysis is separate from your main benchmark and should be run in a dedicated notebook. It is computationally expensive (four training runs) but is arguably the most practically important evaluation in the entire project — because in real deployment, the model always encounters new surveys.

Compare the leave-one-asset-out MAE to the within-distribution test MAE from the main benchmark. If the gap is small (< 2ms), your model generalizes well. If the gap is large (> 5ms), the model has overfit to the specific character of the training assets and would need retraining on any new survey. This is a critical finding to understand before drawing conclusions.

---

## 5.4 — Ensemble Strategies

After individual model benchmarking and error analysis, combining the best models into an ensemble almost always improves performance. The key insight is that different model families make uncorrelated errors — a 1D ResNet and a U-Net fail on different traces, so combining their predictions reduces overall error.

---

### 5.4.1 — Measuring Ensemble Potential

Before building any ensemble, measure whether the models you intend to combine actually make uncorrelated errors. For your top 4–5 models, compute the pairwise **Pearson correlation of their per-trace absolute errors** across the test set. 

If two models have error correlation close to 1.0, they fail on the same traces and ensembling them will help very little. If correlation is 0.3–0.5, they fail independently and ensembling will provide meaningful gains. This analysis takes 5 minutes and prevents wasted effort on ensembles that cannot help.

---

### 5.4.2 — Simple Averaging Ensemble

The baseline ensemble: for each test trace, average the predicted first break times from N models. This requires zero additional training and often achieves 80% of the gains possible from ensembling.

Weight the models by their inverse validation MAE — models with lower validation MAE get higher weight. Specifically, weight for model i = (1 / val_MAE_i) / sum(1 / val_MAE_j for all j). This is a principled weighting that requires no additional optimization.

---

### 5.4.3 — Learned Ensemble (Stacking)

A learned ensemble uses a small meta-model that takes the predictions from all base models as input and learns the optimal combination.

**Input to meta-model per trace:**
- Predicted first break times from each base model (N values)
- Prediction variance across base models (1 value — high variance indicates disagreement, which correlates with error)
- Offset of the trace (1 value)
- Per-model confidence scores if available (e.g., the probability at the predicted first break sample from a segmentation model)

**Meta-model:** A simple LightGBM regressor trained on the validation set predictions. Use the validation set outputs from each base model as training data and the ground truth labels as targets. Test the meta-model on the test set.

**Critical data hygiene:** The meta-model must be trained on validation predictions and tested on test predictions. Never train the meta-model on training set predictions — the base models have memorized the training set, so their training predictions are overconfident and the meta-model learns to trust overconfident predictions.

---

### 5.4.4 — Uncertainty Estimation via Monte Carlo Dropout

For your best single model, enable **test-time dropout** by keeping the model in train mode during inference (which leaves dropout active). Run inference on each test gather 20 times with different dropout masks. The mean of the 20 predictions is your final estimate. The standard deviation is your uncertainty estimate.

This gives you per-trace confidence scores at zero additional training cost. Traces with high uncertainty (large standard deviation across MC samples) are the ones where you should distrust the prediction. Plotting uncertainty vs. actual error magnitude verifies that your uncertainty estimates are calibrated — high uncertainty should correlate with high error.

This uncertainty estimate is also a useful auxiliary feature for the stacking meta-model above.

---

## 5.5 — Targeted Improvements Based on Error Analysis

After completing Sections 5.1–5.4, you know exactly what your best model gets wrong and why. This section describes how to address each failure category identified in 5.3.3.

---

### 5.5.1 — Addressing Category A (Noise Burst Misidentification)

If the model frequently picks noise bursts before the true arrival, it has not learned that pre-arrival energy should be close to zero relative to the first arrival energy. Two targeted fixes:

**Fix 1 — Pre-arrival energy penalty in the loss:**
Add a term to the loss that penalizes predictions that have high energy in a window before the predicted first break relative to the energy after it. This directly teaches the model the physical constraint that pre-arrival should be quiet.

**Fix 2 — Augmentation targeting this failure mode:**
During training, synthetically add noise bursts before the true first break in a fraction of training samples. Force the model to learn to ignore them during training rather than discovering the failure mode at test time.

---

### 5.5.2 — Addressing Category B (Cycle Skip)

Cycle skip errors are almost always a fixed offset of approximately one dominant period. This suggests the model has learned to pick a consistent feature of the waveform (e.g., the first peak) rather than the true onset.

**Fix — Phase-insensitive input representation:**
Before feeding traces to the model, compute the **envelope** of the trace using the Hilbert transform. The envelope highlights amplitude onset without phase information. Feed both the raw trace and its envelope as two input channels. This gives the model access to a phase-insensitive representation while retaining the raw signal for precise timing.

---

### 5.5.3 — Addressing Category C (Dead Trace Spillover)

If errors cluster around dead or clipped traces, your trace dropout augmentation (from Phase 2.6) was not sufficient. Increase the dropout probability or add an explicit dead-trace detector to your preprocessing that masks dead traces more aggressively in the label mask.

Additionally, add a dedicated dead-trace indicator channel as a second input channel to your 2D model — a binary map that is 1 where a trace is dead or clipped and 0 elsewhere. This explicitly tells the model which traces to distrust spatially.

---

### 5.5.4 — Addressing Category D (Complex Wavefront Geometry)

If the model fails on non-monotonic first break curves, it has imposed an implicit smoothness prior that the data violates. Two approaches:

**Fix 1 — Reduce the spatial smoothing in the model:**
If your U-Net pools aggressively along the trace axis, it forces spatial smoothness in the output. Reduce trace-axis pooling to preserve sharper spatial transitions.

**Fix 2 — Post-processing with physics constraint relaxation:**
After model inference, apply a smoothing filter to the predicted first break curve only where the model has high confidence (low MC dropout variance). At positions of low confidence, let the raw model prediction stand without smoothing. This avoids over-smoothing genuine discontinuities.

---

### 5.5.5 — Pseudo-Labeling on Unlabeled Traces

After your best model is fully trained and validated, use it to generate pseudo-labels for all unlabeled traces across all four assets.

**The procedure:**

Run inference on every unlabeled trace. For each prediction, also compute MC dropout uncertainty. Accept pseudo-labels only where:
- The uncertainty (MC dropout std) is below the 25th percentile of uncertainty scores across all unlabeled traces
- The predicted first break time is physically plausible given the trace's offset (it falls within 2 standard deviations of the expected first break time for that offset, computed from a linear moveout model fit to the labeled data)

Assign these accepted pseudo-labels a reduced weight (0.3–0.5) in the loss function, reflecting lower confidence relative to human labels.

Retrain your best model from its current checkpoint on the combined labeled + pseudo-labeled dataset with the weighted loss. Expect a small but consistent improvement — typically 5–15% reduction in MAE — because the model now has substantially more training signal.

---

## 5.6 — Final Model Selection Decision Framework

After all experiments, ensembles, and targeted improvements are complete, you must select one final model to submit. This decision is not purely about MAE.

---

### 5.6.1 — The Decision Matrix

Score each candidate on five criteria, each on a 1–5 scale:

**Criterion 1 — Test MAE (weight: 35%):**
Score based on percentile rank among all tested models. The model with the lowest MAE gets 5, others scaled proportionally.

**Criterion 2 — Cross-asset generalization (weight: 25%):**
Score based on the leave-one-asset-out experiment. Models that generalize well across assets score higher regardless of in-distribution MAE. A model that achieves 4ms MAE in-distribution but 12ms out-of-distribution scores lower than one that achieves 5ms in both settings.

**Criterion 3 — Robustness to noise (weight: 20%):**
Score based on MAE on low-SNR test traces specifically. A model that performs well on easy traces but degrades catastrophically on hard traces is less valuable than one that degrades gracefully.

**Criterion 4 — Statistical confidence (weight: 10%):**
If the model's improvement over the second-best model is not statistically significant (Wilcoxon p > 0.05), reduce its score on this criterion. A numerically better but statistically equivalent model should not automatically win.

**Criterion 5 — Practical constraints (weight: 10%):**
Inference speed and model size. A model that requires 2 seconds per gather vs. one that requires 50ms is meaningfully different in deployment.

Compute the weighted score for each candidate. The highest-scoring model is your submission model. Document this decision explicitly — the reasoning matters as much as the result.

---

### 5.6.2 — Final Deliverable Checklist

Before declaring Phase 5 complete, verify every item:

**Code deliverables:**
- All notebooks run cleanly from top to bottom in a fresh Colab session
- All `src/` modules importable without errors
- `requirements.txt` complete and pinned to specific versions
- README documents the full pipeline from download to final results
- MLflow tracking database on Drive with all experiment runs logged
- `benchmark_results.csv` complete with all models

**Result deliverables:**
- Master benchmark table as styled HTML
- Per-asset error distribution plots for the final model
- Spatial error maps for each asset
- Failure mode taxonomy document
- Cross-asset generalization table
- Ensemble vs. single model comparison
- Statistical significance matrix for top 5 models
- 12 visualization example composites (3 per asset, all difficulty levels)
- Final model checkpoint saved as `final_model.pt` with full metadata

**Reproducibility verification:**
Delete all cached results (but not the raw data or checkpoints). Re-run the benchmarking notebook from scratch. Verify the numbers match what you previously recorded to within floating point tolerance. If they do not, you have a reproducibility bug — find and fix it before submitting.

---

This completes the full plan across all five phases. The complete pipeline runs from raw compressed HDF5 files on Google Drive through EDA, preprocessing, model training across four assets with rerun-safe progressive fine-tuning, MLflow-tracked benchmarking, deep error analysis, targeted improvements, and a principled final model selection — all within Colab's free-tier constraints.