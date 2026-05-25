# Spotify Sentiment Analysis with IndoBERT

Klasifikasi sentimen ulasan Spotify Indonesia menggunakan Transformer (IndoBERT) dan dibandingkan dengan baseline TF-IDF + Logistic Regression.

Proyek ini dibuat sebagai tugas akhir mata kuliah Kecerdasan Buatan FTUI 2026.

---

## Overview

Model melakukan klasifikasi sentimen ulasan Spotify Google Play Store ke dalam 3 kelas:

- Negatif
- Netral
- Positif

Pendekatan utama:
- IndoBERT (Transformer)
- TF-IDF + Logistic Regression (Baseline)

Dataset berasal dari Kaggle:
Spotify Reviews Indonesia (Google Play Store)

---

## Model

### Main Model
- `indobenchmark/indobert-base-p1`
- Fine-tuning untuk multiclass sentiment classification

### Baseline
- TF-IDF (unigram + bigram)
- Logistic Regression

---

## Dataset

- Total dataset asli: 100.000 ulasan
- Balanced sampling:
  - 4.000 negatif
  - 4.000 netral
  - 4.000 positif
- Total dataset digunakan: 12.000 sampel

### Split Dataset

| Subset | Jumlah |
|---|---|
| Train | 8.400 |
| Validation | 1.800 |
| Test | 1.800 |

---

## Preprocessing

- Lowercase
- Remove URL
- Remove mention (@user)
- Remove hashtag
- Remove special characters
- Normalize whitespace

---

## Architecture Pipeline

```text
Dataset
→ Labeling
→ Text Cleaning
→ Balanced Sampling
→ Train/Validation/Test Split
→ Tokenization
→ IndoBERT Fine-tuning / TF-IDF
→ Evaluation
→ Confusion Matrix
→ Sentiment Prediction
```

---

## Results

| Model | Accuracy | F1-Macro |
|---|---|---|
| TF-IDF + Logistic Regression | 65.50% | 65.09% |
| IndoBERT (LR=2e-5) | 67.89% | 67.59% |
| IndoBERT (LR=1e-5) | 67.94% | 67.58% |

IndoBERT berhasil mengungguli baseline sekitar:
- +2.39% Accuracy
- +2.50% F1-Macro

---

## Tech Stack

- Python
- PyTorch
- HuggingFace Transformers
- Scikit-learn
- Pandas
- Matplotlib

---

## Project Structure

```text
├── notebooks/
│   ├── 01_eda_preprocessing.ipynb
│   ├── 02_baseline.ipynb
│   ├── 03_transformer.ipynb
│   └── 04_web_demo.ipynb
│
├── src/
│   └── preprocessing.py
│
├── models/
├── outputs/
├── README.md
└── metadata.json
```

---

## Team

- Dhafin Hamizan Setiawan — 2306267145
- Daffa Bagus Dhiananto — 2306250756
- Dimas Ananda Sutiardi — 2306250586
- Rafi Naufal Aryaputra — 2306250680

---

## References

- Devlin et al. — BERT: Pre-training of Deep Bidirectional Transformers
- Wilie et al. — IndoBERT / IndoNLU
- Vaswani et al. — Attention Is All You Need

---

## Dataset Source

```text
https://www.kaggle.com/datasets/pandaa12/spotify-reviews-indonesia-google-play-store
```
