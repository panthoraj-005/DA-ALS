# Model artifacts go here

Copy these out of the notebook's `ALS_Saved_Models` folder on Google Drive
(`/content/drive/MyDrive/ALS_Saved_Models`) into this directory:

```
models/
├── cnn_best.pt              # REQUIRED. Best-val-loss CNN checkpoint.
│                            #   cnn_model.pt / emg_cnn.pt also load, but those
│                            #   are last-epoch weights — prefer cnn_best.pt.
├── meta_learner.pkl         # XGBoost fusion model (joblib)
├── florence_als_head.pt     # classifier head on the Florence embedding
└── florence_finetuned/      # fine-tuned Florence-2 (a whole HF directory)
    ├── config.json
    ├── model.safetensors    # or pytorch_model.bin
    ├── preprocessor_config.json
    └── ...
```

The server starts without them and tells you what is missing — check
`GET /health` or the "Model status" panel in the UI. What each missing file costs
you:

| Missing | Effect |
|---|---|
| `cnn_best.pt` | No predictions at all. `/predict` returns 503. |
| `florence_finetuned/` or `florence_als_head.pt` | No vision model, so no fusion and no SHAP. The CNN still scores. |
| `meta_learner.pkl` | No fusion and no SHAP. The CNN carries the decision. |

`florence_finetuned/` must contain a real weight file. A directory with only
`config.json` and tokenizer files is treated as invalid — the notebook hits the
same trap when a Drive save is interrupted.
