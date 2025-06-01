# MEDPRIV-AI Framework 🏥🔐  
_Developed for the ISB–Cardiff Hackathon_

MEDPRIV-AI is a privacy-preserving machine learning framework tailored for sensitive healthcare datasets. It combines **Federated Learning** and **Differential Privacy (DP)** to ensure identity protection, regulatory compliance, and robust synthetic data generation.

---

## 🚀 Features

- 🧠 **Federated Learning Simulation** – Decentralized training without data sharing
- 🧬 **Differentially Private GAN** – Synthetic medical data generation with privacy guarantees
- 🧾 **Compliance-Focused** – Built with HIPAA and GDPR principles in mind
- 📈 **Evaluation Metrics** – Accuracy, F1-score, and classification reports included
- 🖥️ **Interactive Streamlit UI** – Easily explore, train, and visualize results

---

## 🛠 Technologies Used

- Python
- Streamlit
- PyTorch + Opacus
- Scikit-learn
- Pandas, NumPy
- Differential Privacy principles
- Federated Learning (simulated)

---
🧪 Privacy-Focused Design
✅ Federated Learning
Simulates hospital nodes training models independently

Aggregates model weights, never shares raw data

✅ Differential Privacy (via Opacus)
Adds calibrated noise to gradients

Tracks and displays ε (privacy budget) in training logs

Defends against Membership Inference Attacks

🧬 Synthetic Data
The DP-GAN component creates realistic but non-identifiable medical records, making it safe for:

Prototyping ML models

Sharing across institutions

Teaching and demonstrations

🎓 Use Cases
Secure healthcare ML pipelines

Privacy-compliant synthetic data generation

Academic research in federated learning and DP

Hospitals needing safe AI experimentation
