# 🔧 Modified Training Config (Vietnamese TTS)

This `config.json` file is customized for prosody-aware training in Vietnamese voice cloning.  
It is used with a modified So-VITS-SVC v4 system as part of our paper:

> **"Towards Cost-Effective Voice Cloning System for Vietnamese TTS: A Case Study at HCMUT"**

## 🔍 Key Modifications from Original
| Field | Value | Purpose |
|-------|-------|---------|
| `epochs` | `200` | Extended training for low-resource speech |
| `batch_size` | `6` | Tuned for Colab RAM (~12–16GB) |
| `segment_size` | `10240` | Balanced training speed and quality |
| `c_mel` | `45` | Increased for clearer pitch modeling |
| `lr_decay` | `0.999875` | Slower decay to preserve learning rate |
| `half_type` | `"fp16"` | Mixed precision support (for future) |

## 🧩 Usage
To use this config:
1. Clone original [so-vits-svc](https://github.com/svc-develop-team/so-vits-svc)
2. Replace `configs/config.json` with this file.
3. Run your modified training script or our Colab notebook.
