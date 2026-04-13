"""Tabular feature extraction from 1D seismic traces."""

import numpy as np

def extract_features(traces: np.ndarray, offsets: np.ndarray) -> np.ndarray:
    """Extract tabular features mapped directly to LightGBM.
    Args:
        traces: [B, n_samples] spatial tensor.
        offsets: [B] float distances.
    Returns:
        [B, n_features] tabular array.
    """
    B, L = traces.shape
    
    # 1. offset
    feat_offset = offsets[:, np.newaxis]
    
    # 2. max absolute amplitude and position
    abs_x = np.abs(traces)
    max_amp = np.max(abs_x, axis=1, keepdims=True)
    max_idx = np.argmax(abs_x, axis=1)[:, np.newaxis].astype(np.float32)
    
    # 3. zero crossings total
    zero_crossings = np.sum(np.diff(np.sign(traces), axis=1) != 0, axis=1, keepdims=True)
    
    # 4. RMS energy in 5 sliding windows
    window_size = L // 5
    rms_feats = []
    for w in range(5):
        win = traces[:, w * window_size : (w + 1) * window_size]
        rms = np.sqrt(np.mean(win**2, axis=1, keepdims=True) + 1e-12)
        rms_feats.append(rms)
            
    # 5. STA/LTA vectorized implementation
    x_sq = traces ** 2
    cumsum = np.insert(np.cumsum(x_sq, axis=1), 0, 0, axis=1)
    
    sta_samps = 5
    lta_samps = 50
    
    sta = np.zeros_like(traces)
    lta = np.zeros_like(traces)
    
    if L >= lta_samps:
        sta_sum = cumsum[:, sta_samps:] - cumsum[:, :-sta_samps]
        sta[:, sta_samps - 1:] = sta_sum / sta_samps
        lta_sum = cumsum[:, lta_samps:] - cumsum[:, :-lta_samps]
        lta[:, lta_samps - 1:] = lta_sum / lta_samps
        
    ratio = np.zeros_like(traces)
    valid = lta > 1e-12
    np.divide(sta, lta, out=ratio, where=valid)
    
    max_ratio = np.max(ratio, axis=1, keepdims=True)
    argmax_ratio = np.argmax(ratio, axis=1)[:, np.newaxis].astype(np.float32)
    
    # Combine all features along axis=1
    features = np.hstack([
        feat_offset,
        max_amp,
        max_idx,
        zero_crossings,
        *rms_feats,
        max_ratio,
        argmax_ratio
    ])
    
    return features.astype(np.float32)
