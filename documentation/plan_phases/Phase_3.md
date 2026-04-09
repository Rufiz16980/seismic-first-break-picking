# Phase 3 — Model Catalog & Training Strategy

This phase covers every viable model family for seismic first break picking, ordered from simplest to most complex. The goal is not to pick one upfront — it is to understand the full landscape so your benchmark is principled, not random.

---

## 3.1 — Framing Reminder Before Models

Every model family below can be applied under one of two framings established in Phase 1. Your architecture choice and your framing choice are **separate decisions** and interact in important ways.

**Framing A — Trace-level regression:**
Input: single 1D trace, shape `[n_samples]`
Output: single float (first break time in ms)
The model sees one trace at a time with no context from neighbors.

**Framing B — Shot-gather regression (recommended for most models):**
Input: full 2D shot gather, shape `[n_samples, n_traces]`
Output: 1D vector of first break times, shape `[n_traces]`
The model sees the full spatial context of the gather.

**Framing C — Segmentation:**
Input: full 2D shot gather, shape `[n_samples, n_traces]`
Output: 2D binary mask, shape `[n_samples, n_traces]` — 1 above first break, 0 below (or vice versa)
The first break time per trace is then extracted as the boundary row index.

Keep these three framings in mind as you read each model section. Most models below support multiple framings.

---

## 3.2 — Tier 0: Classical Signal Processing Baselines

These are not machine learning models. They are deterministic algorithms. You must implement at least one as your absolute baseline — if your ML models cannot beat physics-based methods, something is seriously wrong with your pipeline.

---

### 3.2.1 — STA/LTA (Short-Term Average / Long-Term Average)

**What it is:**
The oldest and most widely used automatic first break picker in seismic processing. For each sample position in a trace, it computes the ratio of the signal energy in a short window immediately after that sample (STA) to the energy in a longer window preceding it (LTA). A first break is declared where this ratio exceeds a threshold.

**Why it works for this task:**
Before the first arrival, the trace is dominated by background noise (low energy). After the first arrival, signal energy spikes. The STA/LTA ratio captures this transition sharply.

**Framing:** Trace-level only (Framing A).

**Parameters to tune:**
- STA window length (typically 5–20ms)
- LTA window length (typically 50–200ms)
- Threshold ratio (typically 2–5)

**Advantages:** No training required, runs instantly, interpretable, well-understood failure modes.

**Disadvantages:** Sensitive to noise, requires manual threshold tuning per asset, fails on low-SNR traces, does not use neighboring trace context.

**Implementation:** `obspy` library has a robust STA/LTA implementation. Alternatively implement from scratch in numpy — it is straightforward.

---

### 3.2.2 — Modified Energy Ratio (MER)

**What it is:**
A refinement of STA/LTA specifically designed for first break picking. It computes the ratio of energy in a window after a candidate sample to the energy before it, with an additional term that penalizes the candidate if the trace energy is uniform throughout (no clear onset).

**Why it outperforms STA/LTA:**
It is less sensitive to threshold choice and handles gradual onset first breaks better.

**Framing:** Trace-level only.

**Implementation:** Described in detail in the paper "Automatic first arrival picking: New strategies" (Wong et al., 2009). This is a short algorithm, easily implemented in numpy.

---

### 3.2.3 — Akaike Information Criterion (AIC) Picker

**What it is:**
Treats the seismic trace as a piecewise-stationary process. Computes the AIC for every possible split point in the trace (treating the left portion as one stationary process and the right portion as another). The minimum AIC point is the first break — it is the time that best separates the pre-arrival noise from the post-arrival signal.

**Why it is valuable:**
Theoretically grounded, does not require threshold tuning, often performs better than STA/LTA on clean data.

**Disadvantages:** Computationally heavier (though still fast compared to ML), can be fooled by non-stationary noise.

**Framing:** Trace-level only.

**Recommendation:** Implement all three classical methods and compute their MAE on your test set per asset. Their scores define the floor your ML models must beat to be worth deploying.

---

## 3.3 — Tier 1: Classical Machine Learning on Handcrafted Features

These methods require you to manually engineer features from each trace or gather, then feed those features to a standard ML classifier/regressor.

---

### 3.3.1 — Feature Engineering for Seismic Traces

Before discussing models, you need a feature vocabulary. For each trace (or local window of a trace), you can compute:

**Time-domain features:**
- RMS energy in sliding windows across the trace
- Maximum absolute amplitude and its sample position
- Zero-crossing rate in sliding windows
- Kurtosis in sliding windows (seismic arrivals often have high kurtosis)
- Skewness in sliding windows
- STA/LTA curve itself (as a feature vector, not a threshold picker)

**Frequency-domain features:**
- Dominant frequency in pre-arrival vs. post-arrival windows
- Spectral centroid shift across the trace
- Short-time Fourier transform (STFT) magnitude along the trace

**Context features (using neighboring traces):**
- Trace-to-trace cross-correlation at candidate first break times
- Coherence of waveforms across adjacent traces in the shot gather

**Offset feature:**
- The offset distance itself is a powerful feature — first break time is strongly correlated with offset through the near-surface velocity. Include offset as an input feature for all classical ML models.

The quality of your features determines the ceiling of all Tier 1 models. This is their fundamental limitation and the reason deep learning methods exist.

---

### 3.3.2 — Random Forest Regressor

**Framing:** Trace-level regression (Framing A). Input: feature vector per trace. Output: first break time in ms.

**Why include it:**
Random Forests are robust to irrelevant features, handle non-linear relationships well, require minimal hyperparameter tuning, and provide feature importance scores — which tell you which hand-engineered features actually matter.

**Key hyperparameters:** n_estimators (100–500), max_depth (None or 10–30), min_samples_leaf.

**Advantages:** Fast to train, interpretable feature importance, good baseline with moderate features.
**Disadvantages:** Does not use spatial context between traces, limited by feature quality.

---

### 3.3.3 — Gradient Boosting (XGBoost / LightGBM)

**Framing:** Trace-level regression.

**Why include it:**
Gradient boosting consistently outperforms Random Forests on tabular data with engineered features. XGBoost is the standard; LightGBM is faster and handles larger datasets better.

**Why LightGBM is preferable here:**
With potentially millions of labeled traces across four assets, training speed matters. LightGBM is 5–10× faster than XGBoost on large datasets with negligible accuracy difference.

**Key hyperparameters:** n_estimators, learning_rate, max_depth, subsample, colsample_bytree. Use Optuna for hyperparameter search — it is lightweight and Colab-compatible.

**Advantages:** Often the best classical ML performer on tabular data, fast inference.
**Disadvantages:** Same fundamental limitation as Random Forest — no spatial context.

---

### 3.3.4 — SVM with HOG or LBP Features (Hybrid)

**Framing:** Can work at trace-level or on small 2D patches.

**HOG (Histogram of Oriented Gradients):**
Originally designed for pedestrian detection. Applied to a 2D seismic gather patch around a candidate first break position, HOG captures the directional gradient structure — which is physically meaningful because first break arrivals create a strong gradient boundary in the gather image.

**LBP (Local Binary Patterns):**
Captures local texture patterns in a 2D image patch. Less physically motivated for seismic data than HOG but still worth trying.

**Workflow for SVM+HOG:**
1. For each trace in a gather, extract a 2D patch centered on candidate first break times (e.g., proposed by STA/LTA).
2. Compute HOG features on the patch.
3. Train an SVM to classify whether the candidate time is a true first break or not (binary classification).
4. The highest-scoring candidate becomes the predicted first break.

**Honest assessment:** This is significantly more complex to implement than the pure tabular approaches above, and deep learning methods will likely outperform it. Include it only if you have time. It is worth understanding conceptually.

---

## 3.4 — Tier 2: Deep Learning on 1D Traces

These models treat each trace as a 1D time series and use learned feature extraction rather than hand-engineered features.

---

### 3.4.1 — 1D CNN Regressor

**Framing:** Trace-level regression (Framing A).

**Architecture:**
A stack of 1D convolutional layers with increasing receptive fields, followed by global average pooling and a dense regression head.

```
Input: [batch, 1, n_samples]
→ Conv1D(32, kernel=7) → BN → ReLU
→ Conv1D(64, kernel=5) → BN → ReLU → MaxPool
→ Conv1D(128, kernel=5) → BN → ReLU → MaxPool
→ Conv1D(256, kernel=3) → BN → ReLU
→ GlobalAveragePool
→ Dense(128) → ReLU → Dropout(0.3)
→ Dense(1)   ← output: first break time in ms
```

**Why this is a solid Tier 2 baseline:**
Simple to implement, trains in minutes, no pretrained weights needed, directly processes the raw signal. The receptive field of stacked convolutions naturally captures the transition from pre-arrival noise to post-arrival signal.

**Key design decision:** Use **causal convolutions** (only look at past samples) or standard convolutions (look at full context). Standard convolutions are better here because the pre-arrival character informs the post-arrival context and vice versa.

**Loss function:** MAE (L1 loss) is more robust to occasional label outliers than MSE. Use MAE as your primary loss across all regression models.

---

### 3.4.2 — 1D CNN with Residual Connections (1D ResNet)

**Framing:** Trace-level regression.

**Why residual connections matter:**
Deeper networks (more layers = larger receptive field = more context) train more stably with skip connections. For a trace of 1500ms at 0.25ms sample rate (6000 samples), you need a large receptive field to capture the full character of the pre-arrival noise, which can span hundreds of milliseconds.

**Architecture:** Stack of residual blocks, each containing two Conv1D layers with a skip connection, followed by a global pooling and regression head. This is your primary 1D deep learning model.

---

### 3.4.3 — Temporal Convolutional Network (TCN)

**Framing:** Trace-level, but can be adapted to output a sequence.

**What it is:**
A 1D CNN variant using **dilated causal convolutions** — convolutions where the kernel is applied to every 2nd, 4th, 8th... sample rather than every adjacent sample. This exponentially expands the receptive field without increasing the number of parameters.

**Why it is relevant:**
A standard 1D CNN with kernel size 3 needs 10 layers to achieve a receptive field of ~20 samples. A TCN with dilation factors [1, 2, 4, 8, 16, 32] achieves a receptive field of 63 samples with only 6 layers. For long seismic traces, this efficiency matters enormously.

**Adaptation for first break picking:**
Rather than outputting a single time value (pure regression), a TCN can output a probability score at every sample position — treating the task as finding the one sample where "noise ends and signal begins." The first break is then the argmax of this score vector. This is a sequence-to-sequence formulation and is powerful.

---

### 3.4.4 — LSTM / BiLSTM Regressor

**Framing:** Trace-level regression.

**What it is:**
A recurrent neural network that processes the trace sample by sample, maintaining a hidden state that captures temporal context. BiLSTM processes the trace in both forward and backward directions and concatenates the results.

**Honest assessment for this task:**
LSTMs are theoretically appealing for sequential data but in practice have been consistently outperformed by 1D CNNs and TCNs on seismic first break picking. They are slower to train, more prone to vanishing gradients on long sequences, and require more careful initialization. Include one BiLSTM experiment to verify this empirically, then deprioritize it.

The main value of including it is completeness — you can state in your benchmark that recurrent approaches were tested and found inferior, which is a meaningful result.

---

## 3.5 — Tier 3: Deep Learning on 2D Shot Gathers

This is the most powerful tier. Models in this tier see the full spatial context of a shot gather and can exploit the coherence of the first break curve across adjacent traces — something no Tier 0, 1, or 2 model can do.

---

### 3.5.1 — 2D CNN Regressor

**Framing:** Shot-gather regression (Framing B).

**Architecture:**
A standard 2D CNN encoder that processes the full gather image and outputs a vector of first break times.

```
Input: [batch, 1, n_samples, n_traces]
→ Several 2D Conv blocks (Conv2D → BN → ReLU → Pool along time axis only)
→ Flatten along time axis → shape: [batch, channels, n_traces]
→ 1D Conv along trace axis
→ Dense head per trace position → [batch, n_traces]
```

**Critical architectural note:**
Pool only along the time axis (vertical), never along the trace axis (horizontal). You need to preserve the per-trace spatial resolution to output one prediction per trace. Pooling along the trace axis would destroy the spatial correspondence between output positions and input traces.

**Why this is a major step up from Tier 2:**
A neighboring trace that has a very clean first break arrival implicitly regularizes the prediction for a noisy neighboring trace. The model learns the spatial coherence of the first break curve as a geometric prior.

---

### 3.5.2 — U-Net (Segmentation Framing)

**Framing:** Segmentation (Framing C). This is the most important single architecture in your entire catalog.

**What it is:**
Originally designed for biomedical image segmentation. Takes a 2D image as input and outputs a 2D mask of the same spatial resolution. In your case, the mask encodes which region of the gather is above the first break (0) and which is below (1), with the boundary being the first break curve.

**Why U-Net is the gold standard for this task:**
- Encoder-decoder architecture with skip connections preserves both high-resolution spatial detail (needed for precise time localization) and high-level semantic context (needed to distinguish signal from noise globally).
- The segmentation framing is a natural fit — the first break curve is literally a boundary in the 2D gather image.
- U-Net has been applied to seismic first break picking in multiple published papers and consistently achieves state-of-the-art results.
- The architecture is proven, well-understood, and available in multiple PyTorch implementations (segmentation-models-pytorch library).

**Architecture specifics for your task:**

```
Input: [batch, 1, n_samples, n_traces]   ← single channel (amplitude)

Encoder:
→ DoubleConv(1→64)   → MaxPool2D    [skip1]
→ DoubleConv(64→128) → MaxPool2D    [skip2]
→ DoubleConv(128→256)→ MaxPool2D    [skip3]
→ DoubleConv(256→512)→ MaxPool2D    [skip4]
→ Bottleneck DoubleConv(512→1024)

Decoder:
→ Upsample + concat(skip4) → DoubleConv(1024→512)
→ Upsample + concat(skip3) → DoubleConv(512→256)
→ Upsample + concat(skip2) → DoubleConv(256→128)
→ Upsample + concat(skip1) → DoubleConv(128→64)
→ Conv1×1(64→1) → Sigmoid          ← binary mask output
```

**Post-processing to extract first break times from the mask:**
For each trace column in the output mask, find the first row where the value exceeds 0.5 (or use argmax on the gradient of the mask column). This gives you the first break sample index per trace, which you convert to ms.

**Loss function for segmentation framing:**
Combine Binary Cross-Entropy loss with Dice loss. BCE alone can suffer from class imbalance (above the first break is often a minority of samples). The combined loss handles this naturally.

**Important variation — direct regression head on U-Net:**
Instead of outputting a binary mask, modify the final layer to output a 1D regression vector of shape `[n_traces]` directly. The encoder-decoder still runs on the full 2D gather, but the output head pools spatially and regresses. This hybrid gives you the contextual power of U-Net with the simpler MAE regression loss. Train both variants and compare.

---

### 3.5.3 — Pretrained ResNet / EfficientNet as 2D Encoder (Transfer Learning)

**Framing:** Shot-gather regression or segmentation (Framings B or C).

**The approach:**
Use a ResNet-50 or EfficientNet-B4 pretrained on ImageNet as the encoder backbone, replace or adapt the final layers for your task.

**The elephant in the room — domain mismatch:**
ImageNet pretraining learns features relevant to natural photographs (edges, textures, objects). Seismic gathers are grayscale, single-channel, and have very different statistical properties. The low-level features (edge detectors in early layers) transfer reasonably well. The high-level semantic features (object detectors in late layers) do not transfer at all.

**How to handle this properly:**

Step 1 — Convert your single-channel gather to 3 channels by repeating it: `gather_3ch = gather.repeat(1, 3, 1, 1)`. This lets you use standard pretrained weights without architectural modification.

Step 2 — Use the pretrained model as a frozen feature extractor first. Add only a small task-specific head and train that head for a few epochs. This gives you a quick baseline.

Step 3 — Gradually unfreeze layers from the top of the encoder downward and fine-tune with a very small learning rate (1e-5 or lower for pretrained layers vs. 1e-3 for the new head). This is called **discriminative learning rates** or **gradual unfreezing** and prevents catastrophic forgetting of useful pretrained features.

**ResNet as encoder in U-Net (ResNet-UNet):**
The most powerful use of pretrained ResNet for this task is as the encoder inside a U-Net architecture. The `segmentation-models-pytorch` library provides this out of the box with a single line change. This gives you ImageNet pretraining, U-Net spatial context, and skip connections — the strongest combination available.

**EfficientNet considerations:**
EfficientNet-B0 through B4 offer an excellent accuracy/parameter tradeoff. B0 is fast enough for Colab; B4 gives the best accuracy. For your Colab constraints, start with B0 or B2.

---

### 3.5.4 — Vision Transformer (ViT) / Swin Transformer

**Framing:** Shot-gather (Framing B or C).

**What it is:**
Transformer architectures applied to image patches. The image is divided into fixed-size patches, each linearly embedded into a token, and self-attention is computed globally across all tokens.

**Honest assessment for your task:**
Vision Transformers require significantly more data than CNNs to train from scratch. With the amount of data in four seismic assets, you are in the regime where a well-designed CNN will almost certainly outperform a ViT trained from scratch.

**Where ViT becomes relevant:**
If you use a pretrained Swin Transformer (pretrained on ImageNet-21k, for example) as the encoder backbone — the same strategy as ResNet transfer learning above. `segmentation-models-pytorch` supports Swin Transformer backbones. This is worth one experiment late in your benchmark to see if attention-based global context helps over CNN local features.

**Recommendation:** Do not prioritize ViT. Include one Swin-UNet experiment if time permits after your CNN baselines are solid.

---

### 3.5.5 — Existing Published Solutions (Prior Art)

**This is the step most practitioners skip and should not.**

Before training your own custom architecture, you should research the existing published literature specifically on these four datasets. The Brunswick, Halfmile, Lalor, and Sudbury 3D datasets are part of a public benchmark introduced around 2019-2020. The following have been published on this exact benchmark:

- **FBPicker** — A deep learning framework specifically for seismic first break picking using these datasets
- **PhaseNet adaptations** — Originally a phase arrival picker for earthquake seismology, adapted for near-surface seismic
- Papers in *Geophysics*, *IEEE Transactions on Geoscience and Remote Sensing*, and proceedings of *SEG Annual Meeting* specifically addressing this benchmark

**What to do:**
Search Google Scholar for "seismic first break picking deep learning Brunswick Halfmile" and "automatic first break picking HDF5 mining seismic." Download any papers you find. For each paper, record: their architecture, their preprocessing choices, their reported MAE per asset, and whether they released code. If code is released on GitHub, study it carefully — not to copy it, but to understand their design decisions and avoid their documented failure modes.

The best published MAE scores on these datasets become your **target benchmark**. If you can match or exceed them, your solution is competitive with published research.

---

## 3.6 — Model Selection Summary Table

| Tier | Model | Framing | Expected MAE | Training Time (Colab) | Priority |
|---|---|---|---|---|---|
| 0 | STA/LTA | Trace | High (poor) | Seconds | Must-do |
| 0 | AIC Picker | Trace | Medium | Seconds | Must-do |
| 1 | LightGBM + features | Trace | Medium | Minutes | Do |
| 1 | Random Forest | Trace | Medium-High | Minutes | Do |
| 2 | 1D ResNet | Trace | Medium-Low | 15-30 min | Do |
| 2 | TCN | Trace | Medium-Low | 20-40 min | Do |
| 3 | 2D CNN | Gather | Low | 30-60 min | Do |
| 3 | **U-Net** | Gather/Seg | **Very Low** | **1-3 hr** | **Priority** |
| 3 | ResNet-UNet | Gather/Seg | Very Low | 1-2 hr | Priority |
| 3 | Swin-UNet | Gather/Seg | Very Low | 2-4 hr | Optional |

---

## 3.7 — Loss Functions Catalog

Your choice of loss function is a model decision that affects convergence behavior, not just a minor hyperparameter.

**MAE (L1 Loss):**
`loss = mean(|prediction - label|)`
Robust to outlier labels. Does not heavily penalize large errors. Best for noisy ground truth labels (which yours likely have, given they are manually picked).

**MSE (L2 Loss):**
`loss = mean((prediction - label)^2)`
Penalizes large errors quadratically. Will be dominated by mispicked labels. Generally worse for this task but worth one experiment to verify.

**Huber Loss:**
Behaves like MSE for small errors and MAE for large errors, controlled by a delta parameter. Good compromise between stability of MSE and outlier robustness of MAE. Use delta = 5ms as a starting point (errors smaller than 5ms are treated as MSE, larger as MAE).

**Wing Loss:**
Designed specifically for facial landmark localization (which is structurally similar to first break picking — finding a precise time/position in an image). Amplifies the gradient for small errors more than MAE. Worth trying on your best-performing model.

**Weighted MAE with label quality mask:**
If you tracked suspicious labels during preprocessing, down-weight their contribution to the loss. Assign weight 1.0 to clean labels and weight 0.1 to suspicious labels during training. This is a simple but often effective technique.

**For segmentation framing (U-Net with mask output):**
Use `BCE + Dice` combined loss. The weighting between them (typically 0.5/0.5) can be tuned. Alternatively, use **Focal Loss** which is particularly effective when the mask is imbalanced (far more samples above the first break than below in shallow surveys).

---

## 3.8 — Evaluation Metrics

These must be consistent across every model you test for the benchmark to be meaningful.

**Primary metric — MAE in milliseconds:**
`MAE = mean(|predicted_fb_ms - true_fb_ms|)` over all labeled test traces.
This is the standard metric in published first break picking literature and is directly interpretable in physical units.

**Secondary metrics:**

- **RMSE in ms** — sensitive to large errors, complements MAE
- **Within-5ms accuracy** — percentage of predictions within 5ms of ground truth. Physically, 5ms corresponds to roughly one wavelength for typical near-surface seismic, making it a practically meaningful threshold.
- **Within-10ms accuracy** — same, more lenient threshold
- **Per-asset MAE** — report MAE separately for each of the four assets. A model that averages well across assets but fails completely on one is not robust.
- **MAE vs. offset** — plot MAE as a function of source-receiver offset. Models often perform worse at very short offsets (where the first break is early and near the source noise burst) and very long offsets (where the signal is weak). Understanding where your model fails by offset is essential for targeted improvement.
- **MAE vs. SNR** — plot MAE as a function of a per-trace SNR proxy computed during EDA. Shows how gracefully performance degrades as noise increases.

---

## 3.9 — Recommended Training Order

Do not train all models simultaneously. Follow this sequence:

**Week 1 targets:**
1. Classical baselines (STA/LTA, AIC) — establishes the floor
2. LightGBM with engineered features — establishes the classical ML ceiling
3. 1D ResNet — first deep learning result

**Week 2 targets:**
4. 2D CNN regressor — first gather-level result, expect a significant MAE jump downward
5. U-Net with segmentation framing — your main model
6. U-Net with regression head variant — compare to segmentation variant

**Week 3 targets:**
7. ResNet-UNet with pretrained backbone — your strongest model
8. Hyperparameter tuning on the best performer from week 2–3
9. Pseudo-labeling experiment on unlabeled traces using best model
10. Ensemble of top 2–3 models

---

