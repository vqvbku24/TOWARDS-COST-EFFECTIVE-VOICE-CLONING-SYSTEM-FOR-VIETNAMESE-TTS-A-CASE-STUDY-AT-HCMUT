# 🇻🇳 Vietnamese Voice Cloning Addon (PRAAT + So‑VITS‑SVC + TTS)

This repository contains **modified files and instructions** to extend the original [So‑VITS‑SVC v4](https://github.com/svc-develop-team/so-vits-svc) for Vietnamese tonal voice cloning.  
It supports **prosody‑aware voice cloning** with **pitch control**, **speaker embedding**, and a **Colab notebook for reproduction**.

> 📝 This work is part of our research paper:  
> **“Towards Cost‑Effective Voice Cloning System for Vietnamese TTS: A Case Study at HCMUT”**  
> 📄 Presented at *The 9th OISP Science and Technology Symposium for Students*


---

## ✨ What’s New in This Version?

| Feature                           | Description                                                      |
|-----------------------------------|------------------------------------------------------------------|
| 🔊 **Speaker Embedding Integration** | Use pretrained embeddings (SpeechBrain ECAPA‑TDNN) to preserve voice identity |
| 📈 **Prosody‑aware Cloning via PRAAT** | Extract F0 pitch contours using [Parselmouth](https://parselmouth.readthedocs.io/) |
| 🧠 **SpeechBrain Support**          | Integrate SpeechBrain to encode speaker and content             |
| 🗣️ **TTS Voice Injection**           | Use external TTS voices (Zalo AI, Edge TTS) as prompt input for cloning |
| 🧪 **Colab Notebook**               | Run training, embedding extraction and inference fully on Google Colab |
| 🛠️ **Bug Fixes**                    | Resolve conflicts with `numba`, `librosa`, `scipy`, etc.         |
| ⚙️ **Custom Training Config**       | Adjusted hyperparameters for Vietnamese tonal data              |


---

## 🔊 Two Inference Modes Supported

| Mode                            | Description                                                    | Command Example                                                                                               |
|---------------------------------|----------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------|
| 🎯 Default (cluster‑based)      | Speaker control via `--speaker`, KMeans clustering & retrieval | `python modified/inference_main.py -n input.wav -s speaker_1 --cluster_model_path logs/44k/kmeans_10000.pt`  |
| 🧬 Embedding (SpeechBrain ECAPA) | Direct `.npy` speaker embedding injection                       | `python modified/inference_with_embedding.py -n input.wav --use_embedding --embedding_path dataset_raw/embedding/voice1.npy` |

> Prior to using Mode 2, extract embeddings with `extract_spk_embedding.py` or via the Colab cell in `sovits4_for_colab__3_9.ipynb`.

---

## 🚀 Run on Google Colab

Open and execute the notebook to:
1. Train or fine‑tune the model  
2. Extract speaker embeddings  
3. Run either inference mode  

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](./modified/sovits4_for_colab__3_9.ipynb)

---

## 🔌 Integration with Original So‑VITS‑SVC Repository

1. Clone the base repo:  
   ```bash
   git clone https://github.com/svc-develop-team/so-vits-svc


## 📚 Citation

If you use this work, please cite our paper:

```bibtex
@inproceedings{vo2025tts,
  title={Towards Cost‑Effective Voice Cloning System for Vietnamese TTS: A Case Study at HCMUT},
  author={TTS-team et al.},
  booktitle={The 9th OISP Science and Technology Symposium for Students},
  year={2025}
}
```

---

## 🙏 Acknowledgements

- Base repo: So‑VITS‑SVC v4
- Pitch extraction: Parselmouth (Praat)
- Speaker embedding: SpeechBrain
- TTS prompts: Zalo AI, Edge TTS
