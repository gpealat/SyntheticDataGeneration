# Sig-WGAN: Synthetic Financial Time-Series Generation using Path Signatures

This project implements a **Signature Wasserstein GAN (Sig-WGAN)** to generate synthetic **log-return time-series data**, particularly for financial applications. It leverages **path signature features** from rough path theory to capture complex temporal dynamics in sequential data.

---

## 🧠 What is Sig-WGAN?

Sig-WGAN is a variant of the Wasserstein GAN (WGAN) tailored for time-series data. It introduces a **signature transform** layer in the discriminator to extract robust path-wise features of sequences. This allows the critic to better distinguish between real and fake sequences based on their underlying structure.

---

## 📈 Application: Synthetic Financial Data

This implementation focuses on **log returns** of financial assets (e.g., stock prices) to simulate realistic, stationary time-series. The model is trained to generate synthetic returns that can be transformed back into synthetic price paths.

---

## 🔧 Features

- Computes **log returns** from raw price data
- Uses **signature transforms** for robust sequence representation
- Generator produces synthetic return sequences
- Critic operates on **log-signatures** of sequences
- Supports **WGAN-GP** loss for stable training
- Synthetic returns can be **reconstructed into price paths**

---

## 🗂️ Project Structure

SYNTHETIC_DATA/
│
├── data/
│ └── price_data.csv # Raw financial data (closing prices with a Dates column)
│
├── models/
│ ├── generator.py # Generator network
│ ├── critic.py # Critic (discriminator) network
│
├── utils/
│ ├── signature_utils.py # Signature computation and helpers
│ └── data_loader.py # Preprocessing and batching
│
├── train.py # Main training loop
├── generate.py # Generate synthetic returns & prices
├── requirements.txt
└── README.md

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/gpealat/SyntheticDataGeneration.git
cd synthetic_data
```
### 2. Install dependencies

```bash
pip install -r requirements.txt
```

