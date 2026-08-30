"""
Train and export CNN model weights (cnn_best.pt) and meta-learner (meta_learner.pkl)
using the local dataset signals so the web application runs with real trained weights.
"""

from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import xgboost as xgb
import joblib

import config
import models as model_module
import pipeline
import signal_io

def main():
    print("=== Training Local ALS Screening Models ===")
    models_dir = Path("c:/Users/Pritam/Downloads/medicalllm/als-webapp/models")
    models_dir.mkdir(parents=True, exist_ok=True)
    
    samples_dir = Path("c:/Users/Pritam/Downloads/medicalllm/als-webapp/sample_signals")
    
    signals = []
    labels = []
    names = []
    
    for path in sorted(samples_dir.iterdir()):
        if path.suffix.lower() not in {".npy", ".csv"}:
            continue
        try:
            sig = signal_io.read_signal_path(path)
            if len(sig) != config.SIGNAL_LENGTH:
                continue
            
            sig_clean = pipeline.preprocess_signal(sig)
            
            if "_als" in path.name.lower():
                label = 1.0
            elif "_normal" in path.name.lower():
                label = 0.0
            else:
                continue
                
            signals.append(sig_clean)
            labels.append(label)
            names.append(path.name)
        except Exception as e:
            continue
            
    print(f"Loaded {len(signals)} signals ({sum(labels)} ALS, {len(labels) - sum(labels)} Normal).")
    
    X = np.array(signals, dtype=np.float32)[:, np.newaxis, :]  # Shape: (N, 1, 23437)
    y = np.array(labels, dtype=np.float32)[:, np.newaxis]       # Shape: (N, 1)
    
    # Train 1-D CNN
    device = torch.device("cpu")
    torch.manual_seed(42)
    np.random.seed(42)
    
    cnn = model_module.EMG_CNN().to(device)
    optimizer = torch.optim.Adam(cnn.parameters(), lr=5e-4, weight_decay=1e-4)
    criterion = nn.BCELoss()
    
    dataset = TensorDataset(torch.from_numpy(X), torch.from_numpy(y))
    loader = DataLoader(dataset, batch_size=8, shuffle=True)
    
    best_loss = float("inf")
    best_state = None
    
    cnn.train()
    print("Training 1-D CNN backbone...")
    for epoch in range(25):
        epoch_loss = 0.0
        correct = 0
        total = 0
        for batch_x, batch_y in loader:
            optimizer.zero_grad()
            out = cnn(batch_x)
            loss = criterion(out, batch_y)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item() * len(batch_x)
            preds = (out >= 0.5).float()
            correct += (preds == batch_y).sum().item()
            total += len(batch_x)
            
        avg_loss = epoch_loss / total
        acc = correct / total * 100
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1:02d}/25 - Loss: {avg_loss:.4f} - Accuracy: {acc:.1f}%")
            
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_state = cnn.state_dict().copy()
            
    # Save best CNN checkpoint
    cnn_best_path = models_dir / "cnn_best.pt"
    torch.save(best_state, cnn_best_path)
    print(f" Saved trained CNN checkpoint to {cnn_best_path}")
    
    # Train XGBoost Meta-Learner on features
    cnn.load_state_dict(best_state)
    cnn.eval()
    
    meta_features_list = []
    with torch.no_grad():
        for i in range(len(X)):
            sig_t = torch.from_numpy(X[i:i+1])
            prob_t, feat_t = cnn(sig_t, return_features=True)
            prob = float(prob_t.item())
            feat = feat_t.cpu().numpy().flatten()
            
            # Dummy Florence embedding vector (768-D) for metadata vector assembly
            dummy_florence_prob = prob + np.random.normal(0, 0.05)
            dummy_florence_prob = float(np.clip(dummy_florence_prob, 0.01, 0.99))
            dummy_florence_emb = np.random.normal(0, 1, config.FLORENCE_EMBED_DIM).astype(np.float32)
            
            vec = pipeline.build_meta_features(prob, feat, dummy_florence_prob, dummy_florence_emb)
            meta_features_list.append(vec.flatten())
            
    X_meta = np.array(meta_features_list)
    y_meta = np.array(labels, dtype=int)
    
    print("Training XGBoost meta-learner...")
    meta_learner = xgb.XGBClassifier(
        n_estimators=30,
        max_depth=3,
        learning_rate=0.1,
        eval_metric="logloss",
        random_state=42
    )
    meta_learner.fit(X_meta, y_meta)
    
    meta_path = models_dir / "meta_learner.pkl"
    joblib.dump(meta_learner, meta_path)
    print(f" Saved trained XGBoost meta-learner to {meta_path}")
    print("=== Training Complete ===")

if __name__ == "__main__":
    main()
