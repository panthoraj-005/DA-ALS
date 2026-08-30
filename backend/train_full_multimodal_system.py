"""
Comprehensive Multi-Modal Training & Alignment Script
Trains:
  1. 1-D CNN Backbone (cnn_best.pt) on preprocessed EMG signals.
  2. Florence-2 Classification Head (florence_als_head.pt) on Florence-2 vision embeddings extracted from rendered signal images.
  3. XGBoost Meta-Learner (meta_learner.pkl) on the unified 838-D multi-modal feature vectors.

Ensures tight accuracy alignment between 1-D CNN, Florence-2 VLM, and XGBoost Meta-Learner.
"""

from pathlib import Path
import os
import gc
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from PIL import Image as PILImage
from transformers import AutoProcessor, AutoModelForCausalLM
import xgboost as xgb
import joblib

import config
import models as model_module
import pipeline
import signal_io

def main():
    print("=" * 60)
    print("=== MULTI-MODAL ALS MODEL TRAINING & ALIGNMENT ===")
    print("=" * 60)
    
    models_dir = Path("c:/Users/Pritam/Downloads/medicalllm/als-webapp/models")
    models_dir.mkdir(parents=True, exist_ok=True)
    samples_dir = Path("c:/Users/Pritam/Downloads/medicalllm/als-webapp/sample_signals")
    outputs_dir = Path("c:/Users/Pritam/Downloads/medicalllm/als-webapp/outputs")
    outputs_dir.mkdir(parents=True, exist_ok=True)
    
    # -------------------------------------------------------------------
    # Step 1: Load and Preprocess All Signals
    # -------------------------------------------------------------------
    signals_raw = []
    signals_processed = []
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
                
            signals_raw.append(sig)
            signals_processed.append(sig_clean)
            labels.append(label)
            names.append(path.name)
        except Exception as e:
            continue
            
    n_samples = len(signals_processed)
    print(f"Loaded {n_samples} signals ({int(sum(labels))} ALS, {int(n_samples - sum(labels))} Normal).")
    
    X_cnn = np.array(signals_processed, dtype=np.float32)[:, np.newaxis, :]  # Shape: (N, 1, 23437)
    y_arr = np.array(labels, dtype=np.float32)[:, np.newaxis]                # Shape: (N, 1)
    
    # -------------------------------------------------------------------
    # Step 2: Train 1-D CNN Backbone
    # -------------------------------------------------------------------
    print("\n--- Training 1-D CNN Backbone ---")
    device = torch.device("cpu")
    torch.manual_seed(42)
    np.random.seed(42)
    
    cnn = model_module.EMG_CNN().to(device)
    optimizer = torch.optim.Adam(cnn.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.BCELoss()
    
    dataset = TensorDataset(torch.from_numpy(X_cnn), torch.from_numpy(y_arr))
    loader = DataLoader(dataset, batch_size=8, shuffle=True)
    
    best_loss = float("inf")
    best_cnn_state = None
    
    for epoch in range(30):
        cnn.train()
        epoch_loss = 0.0
        correct = 0
        total = 0
        for bx, by in loader:
            optimizer.zero_grad()
            out = cnn(bx)
            loss = criterion(out, by)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item() * len(bx)
            preds = (out >= 0.5).float()
            correct += (preds == by).sum().item()
            total += len(bx)
            
        avg_loss = epoch_loss / total
        acc = (correct / total) * 100
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"  [CNN] Epoch {epoch+1:02d}/30 | Loss: {avg_loss:.4f} | Accuracy: {acc:.1f}%")
            
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_cnn_state = {k: v.cpu().clone() for k, v in cnn.state_dict().items()}
            
    cnn_best_path = models_dir / "cnn_best.pt"
    torch.save(best_cnn_state, cnn_best_path)
    cnn.load_state_dict(best_cnn_state)
    cnn.eval()
    print(f" Saved trained CNN checkpoint to: {cnn_best_path}")
    
    # -------------------------------------------------------------------
    # Step 3: Render Signal Images & Extract Florence-2 Embeddings
    # -------------------------------------------------------------------
    print("\n--- Loading Florence-2 & Extracting Vision Embeddings ---")
    model_id = "microsoft/Florence-2-base"
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    florence_model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.float32, trust_remote_code=True
    ).to(device)
    florence_model.eval()
    
    temp_img_dir = outputs_dir / "training_signal_images"
    temp_img_dir.mkdir(parents=True, exist_ok=True)
    
    florence_embeddings = []
    
    print(f"Rendering {n_samples} signal images and extracting 768-D visual embeddings...")
    for idx, (sig_clean, name) in enumerate(zip(signals_processed, names)):
        abn_regions, _ = pipeline.find_abnormal_regions(sig_clean)
        img_path = temp_img_dir / f"train_sig_{idx:03d}.png"
        pipeline.render_signal_image(sig_clean, abn_regions, img_path, title=name)
        
        pil_img = PILImage.open(img_path).convert("RGB")
        inputs = processor(text=["<MORE_DETAILED_CAPTION>"], images=pil_img, return_tensors="pt").to(device)
        
        with torch.no_grad():
            vision_out = florence_model._encode_image(inputs["pixel_values"])
            if isinstance(vision_out, (tuple, list)):
                vision_out = vision_out[0]
            emb = vision_out.mean(dim=1).cpu().numpy().reshape(-1) # 768-D
            florence_embeddings.append(emb)
            
        pil_img.close()
        if (idx + 1) % 20 == 0 or idx == n_samples - 1:
            print(f"  Processed {idx + 1}/{n_samples} images...")
            
    X_florence_emb = np.array(florence_embeddings, dtype=np.float32)
    print(f"Extracted Florence-2 embeddings shape: {X_florence_emb.shape}")
    
    # -------------------------------------------------------------------
    # Step 4: Train Florence-2 Classification Head (florence_als_head.pt)
    # -------------------------------------------------------------------
    print("\n--- Training Florence-2 Classification Head ---")
    florence_head = model_module.FlorenceALSHead().to(device)
    head_opt = torch.optim.AdamW(florence_head.parameters(), lr=1e-3, weight_decay=1e-4)
    head_criterion = nn.BCEWithLogitsLoss()
    
    head_dataset = TensorDataset(torch.from_numpy(X_florence_emb), torch.from_numpy(y_arr))
    head_loader = DataLoader(head_dataset, batch_size=8, shuffle=True)
    
    best_head_loss = float("inf")
    best_head_state = None
    
    for epoch in range(35):
        florence_head.train()
        epoch_loss = 0.0
        correct = 0
        total = 0
        for bx, by in head_loader:
            head_opt.zero_grad()
            logits = florence_head(bx)
            loss = head_criterion(logits, by)
            loss.backward()
            head_opt.step()
            
            epoch_loss += loss.item() * len(bx)
            preds = (torch.sigmoid(logits) >= 0.5).float()
            correct += (preds == by).sum().item()
            total += len(bx)
            
        avg_loss = epoch_loss / total
        acc = (correct / total) * 100
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"  [Florence Head] Epoch {epoch+1:02d}/35 | Loss: {avg_loss:.4f} | Accuracy: {acc:.1f}%")
            
        if avg_loss < best_head_loss:
            best_head_loss = avg_loss
            best_head_state = {k: v.cpu().clone() for k, v in florence_head.state_dict().items()}
            
    head_path = models_dir / "florence_als_head.pt"
    torch.save(best_head_state, head_path)
    florence_head.load_state_dict(best_head_state)
    florence_head.eval()
    print(f" Saved trained Florence-2 head checkpoint to: {head_path}")
    
    # -------------------------------------------------------------------
    # Step 5: Build Unified 838-D Feature Matrix & Train XGBoost Meta-Learner
    # -------------------------------------------------------------------
    print("\n--- Building Unified 838-D Feature Matrix for XGBoost ---")
    with torch.no_grad():
        # CNN predictions and 64-D features
        cnn_probs_t, cnn_feats_t = cnn(torch.from_numpy(X_cnn), return_features=True)
        cnn_probs = cnn_probs_t.cpu().numpy().flatten()
        cnn_feats = cnn_feats_t.cpu().numpy()
        
        # Florence predictions
        florence_logits_t = florence_head(torch.from_numpy(X_florence_emb))
        florence_probs = torch.sigmoid(florence_logits_t).cpu().numpy().flatten()
        
    meta_features_list = []
    for i in range(n_samples):
        c_prob = float(cnn_probs[i])
        c_feat = cnn_feats[i]
        f_prob = float(florence_probs[i])
        f_emb = X_florence_emb[i]
        
        vec = pipeline.build_meta_features(c_prob, c_feat, f_prob, f_emb)
        meta_features_list.append(vec.flatten())
        
    X_meta = np.array(meta_features_list, dtype=np.float32)
    y_meta = np.array(labels, dtype=int)
    
    print(f"Unified Meta Feature Matrix shape: {X_meta.shape}")
    
    print("\n--- Training XGBoost Meta-Learner ---")
    meta_learner = xgb.XGBClassifier(
        n_estimators=60,
        max_depth=3,
        learning_rate=0.08,
        subsample=0.85,
        colsample_bytree=0.85,
        eval_metric="logloss",
        random_state=42
    )
    meta_learner.fit(X_meta, y_meta)
    
    meta_preds = meta_learner.predict(X_meta)
    meta_acc = (meta_preds == y_meta).mean() * 100
    print(f"  [Meta-Learner] Final Training Accuracy: {meta_acc:.1f}%")
    
    meta_path = models_dir / "meta_learner.pkl"
    joblib.dump(meta_learner, meta_path)
    print(f" Saved trained XGBoost meta-learner to: {meta_path}")
    
    # -------------------------------------------------------------------
    # Step 6: Alignment Verification & Summary
    # -------------------------------------------------------------------
    cnn_preds = (cnn_probs >= 0.5).astype(int)
    florence_preds = (florence_probs >= 0.5).astype(int)
    
    cnn_acc = (cnn_preds == y_meta).mean() * 100
    florence_acc = (florence_preds == y_meta).mean() * 100
    agree_rate = (cnn_preds == florence_preds).mean() * 100
    
    print("\n" + "=" * 60)
    print("=== MODEL ALIGNMENT & ACCURACY REPORT ===")
    print("=" * 60)
    print(f"Total Evaluated Signals      : {n_samples}")
    print(f"1-D CNN Accuracy             : {cnn_acc:.1f}%")
    print(f"Florence-2 VLM Accuracy      : {florence_acc:.1f}%")
    print(f"CNN-Florence Agreement Rate  : {agree_rate:.1f}%")
    print(f"XGBoost Meta-Learner Accuracy: {meta_acc:.1f}%")
    print("=" * 60)

if __name__ == "__main__":
    main()
