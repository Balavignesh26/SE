**** Optimizing CVE Severity Prediction

This repository contains the implementation of our hybrid machine learning approach for **CVE (Common Vulnerabilities and Exposures) severity prediction**, combining **TF-IDF** and **DistilBERT** embeddings with an **attention-based deep learning model**.

-- Overview -- 

Cybersecurity depends on the timely and accurate assessment of vulnerabilities. This project introduces a scalable and efficient model to **automate the prediction of CVE severity levels (LOW, MEDIUM, HIGH, CRITICAL)** using hybrid feature extraction and deep learning.

-- Objectives --

- Automate the classification of CVE severity from vulnerability descriptions.
- Combine statistical and contextual NLP techniques.
- Improve prediction consistency across imbalanced classes.
- Achieve high accuracy with minimal computational overhead.

-- Model Architecture --

- **TF-IDF**: Extracts statistical significance of terms (5,000 features).
- **DistilBERT**: Provides contextual embeddings (768 features).
- **Attention Layer**: Focuses on the most relevant features.
- **Neural Network**:
  - Input: Concatenated 5,768-dim feature vector.
  - Hidden Layers: 512 ➝ 256 ➝ 128 units with ReLU, LayerNorm, Dropout.
  - Output: Softmax over 4 classes.

-- Performance --

| Metric              | Value      |
|---------------------|------------|
| Test Accuracy       | 93.52%     |
| Macro F1-Score      | 0.7926     |
| Weighted F1-Score   | 0.9314     |
| AUC (All Classes)   | 0.98–0.99  |

-- Techniques Used --

- **Natural Language Processing (NLP)**
  - TF-IDF Vectorizer
  - DistilBERT embeddings (`distilbert-base-uncased`)
- **Deep Learning**
  - Attention Mechanism
  - PyTorch with AdamW Optimizer
- **Imbalance Handling**
  - Class weighting (no oversampling or synthetic data)

-- Dataset --

- Source: [National Vulnerability Database (NVD)](https://nvd.nist.gov/)
- Cleaned Dataset: 48,452 entries
- Features: `CVE_ID`, `Description`, `Score`, `Severity`, etc.

-- Evaluation --

- Stratified Train-Validation-Test Split: 68%-15%-17%
- Evaluation metrics include accuracy, macro/weighted F1, and ROC curves.
- Visual analysis using loss and accuracy plots confirms strong generalization.

-- Future Work --

- Add "temporal CVSS metrics" for time-sensitive predictions.
- Explore "ensemble models" using CNNs or LSTMs.
- Expand to "exploitability" or "patch availability" predictions.
- "Real-time deployment" using NVD feeds.
