import streamlit as st
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from opacus import PrivacyEngine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report
import numpy as np

st.title("MEDPRIV-AI Framework")

st.markdown("""
### About MEDPRIV-AI Framework
This app demonstrates a **privacy-preserving machine learning framework for healthcare** that focuses on:
- Generating synthetic medical data with **Differential Privacy (DP)** guarantees to prevent identity leakage.
- Simulating **Federated Learning** where multiple clients collaboratively train a global model **without sharing raw patient data**.
- Ensuring compliance with healthcare data protection regulations such as **HIPAA and GDPR**.

---
""")

@st.cache_data
def load_data():
    df = pd.read_csv("synthetic_medical_data.csv")
    return df

df = load_data()

st.subheader("Dataset preview")
st.write(df.head())
st.write(f"Dataset shape: {df.shape}")

if st.checkbox("Show missing values count"):
    st.write(df.isnull().sum())

# Preprocessing for classification
@st.cache_data
def preprocess_classification(df):
    data = df.copy()
    data['gender'] = LabelEncoder().fit_transform(data['gender'])
    X = data.drop(columns=['diagnosis'])
    y = data['diagnosis']
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, y

X_scaled, y = preprocess_classification(df)

# Split into train/test
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

# --- Centralized Logistic Regression ---

def train_logistic_regression(X_train, y_train):
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train, y_train)
    return model

if st.button("Train Centralized Logistic Regression and Evaluate"):
    model = train_logistic_regression(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    st.write(f"Accuracy: {acc:.4f}")
    st.write(f"F1 Score: {f1:.4f}")
    st.text("Classification Report:")
    st.text(classification_report(y_test, y_pred))

# --- Federated Learning Simulation ---

st.subheader("Federated Learning Simulation")

st.markdown("""
**What is Federated Learning?**

Federated learning allows multiple healthcare institutions or devices to train machine learning models **locally on their own data** without sharing the raw sensitive data with a central server.  
Only model updates (like learned parameters) are shared and aggregated, preserving **patient privacy** and complying with data protection regulations such as HIPAA and GDPR.

Our simple simulation below splits the dataset across multiple clients, trains local models, and averages their parameters to form a global model.
""")

num_clients = st.slider("Number of federated clients", 2, 10, 5)

def federated_train(X_train, y_train, num_clients, rounds=5):
    client_data_size = len(X_train) // num_clients
    clients_X = [X_train[i*client_data_size:(i+1)*client_data_size] for i in range(num_clients)]
    clients_y = [y_train[i*client_data_size:(i+1)*client_data_size] for i in range(num_clients)]

    global_coef = np.zeros(X_train.shape[1])
    global_intercept = 0.0

    for r in range(rounds):
        local_coefs = []
        local_intercepts = []

        for c in range(num_clients):
            local_model = LogisticRegression(max_iter=1000, solver='lbfgs')
            local_model.fit(clients_X[c], clients_y[c])
            local_coefs.append(local_model.coef_.flatten())
            local_intercepts.append(local_model.intercept_[0])

        global_coef = np.mean(local_coefs, axis=0)
        global_intercept = np.mean(local_intercepts)
        st.write(f"Round {r+1}/{rounds} completed.")

    global_model = LogisticRegression()
    global_model.coef_ = global_coef.reshape(1, -1)
    global_model.intercept_ = np.array([global_intercept])
    global_model.classes_ = np.unique(y_train)
    return global_model

if st.button("Run Federated Learning Simulation and Evaluate"):
    global_model = federated_train(X_train, y_train, num_clients=num_clients, rounds=5)
    y_pred = global_model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    st.write(f"Federated Model Accuracy: {acc:.4f}")
    st.write(f"Federated Model F1 Score: {f1:.4f}")
    st.text("Classification Report:")
    st.text(classification_report(y_test, y_pred))

# --- DP-GAN Section ---

st.subheader("Train DP-GAN to Generate Synthetic Data")

st.markdown("""
**What is Differential Privacy (DP)?**

Differential Privacy is a mathematical framework that guarantees **individual data points in a dataset cannot be reverse-engineered or identified** from the model outputs.  
Our DP-GAN adds calibrated noise during training to prevent **identity leakage**, protecting sensitive patient information while allowing useful synthetic data generation.

This helps ensure compliance with data protection laws like HIPAA and GDPR, which require strict privacy guarantees.

Below you can train a DP-GAN and generate privacy-preserving synthetic medical data.
""")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

@st.cache_data
def prepare_gan_data(df):
    data = df.copy()
    data['gender'] = data['gender'].map({'F': 0, 'M': 1})
    X = torch.tensor(data.values, dtype=torch.float32)
    return X

X_gan = prepare_gan_data(df)

batch_size = st.slider("Batch size for GAN training", 16, 256, 64, step=16)
epochs = st.slider("Number of GAN training epochs", 10, 200, 50, step=10)
noise_multiplier = st.slider("Privacy noise multiplier", 0.5, 2.0, 1.2, step=0.1)
max_grad_norm = st.slider("Max gradient norm", 0.1, 5.0, 1.0, step=0.1)
noise_dim = 16
data_dim = X_gan.shape[1]

dataset = TensorDataset(X_gan)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

class Generator(nn.Module):
    def __init__(self, noise_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(noise_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim),
        )
    def forward(self, z):
        return self.net(z)

class Discriminator(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )
    def forward(self, x):
        return self.net(x)

def train_dp_gan(dataloader, epochs, noise_multiplier, max_grad_norm):
    G = Generator(noise_dim, data_dim).to(device)
    D = Discriminator(data_dim).to(device)
    optimizer_G = torch.optim.Adam(G.parameters(), lr=1e-3)
    optimizer_D = torch.optim.Adam(D.parameters(), lr=1e-3)
    criterion = nn.BCELoss()
    privacy_engine = PrivacyEngine()

    D, optimizer_D, dataloader_private = privacy_engine.make_private(
        module=D,
        optimizer=optimizer_D,
        data_loader=dataloader,
        noise_multiplier=noise_multiplier,
        max_grad_norm=max_grad_norm,
        poisson_sampling=False,
    )

    for epoch in range(epochs):
        for real_batch, in dataloader_private:
            real_batch = real_batch.to(device)
            batch_size_curr = real_batch.size(0)
            valid = torch.ones(batch_size_curr, 1).to(device)
            fake = torch.zeros(batch_size_curr, 1).to(device)

            # Train Discriminator
            optimizer_D.zero_grad()
            real_pred = D(real_batch)
            loss_real = criterion(real_pred, valid)
            z = torch.randn(batch_size_curr, noise_dim).to(device)
            fake_data = G(z).detach()
            fake_pred = D(fake_data)
            loss_fake = criterion(fake_pred, fake)
            loss_D = (loss_real + loss_fake) / 2
            loss_D.backward()
            optimizer_D.step()

            # Train Generator
            optimizer_G.zero_grad()
            z = torch.randn(batch_size_curr, noise_dim).to(device)
            gen_data = G(z)
            pred_gen = D(gen_data)
            loss_G = criterion(pred_gen, valid)
            loss_G.backward()
            optimizer_G.step()

        epsilon = privacy_engine.get_epsilon(delta=1e-5)
        st.write(f"Epoch {epoch+1}/{epochs} | Loss D: {loss_D.item():.4f} | Loss G: {loss_G.item():.4f} | ε (privacy budget): {epsilon:.2f}")

    G.eval()
    with torch.no_grad():
        z = torch.randn(1000, noise_dim).to(device)
        synthetic_samples = G(z).cpu().numpy()

    synthetic_df = pd.DataFrame(synthetic_samples, columns=df.columns)
    return synthetic_df

if st.button("Train DP-GAN and Generate Synthetic Data"):
    with st.spinner("Training DP-GAN... This may take a while!"):
        synthetic_df = train_dp_gan(dataloader, epochs, noise_multiplier, max_grad_norm)
        st.success("Training complete! Here are some synthetic samples:")
        st.write(synthetic_df.head())

        csv = synthetic_df.to_csv(index=False).encode()
        st.download_button(
            label="Download Synthetic Data as CSV",
            data=csv,
            file_name="synthetic_medical_data_generated.csv",
            mime="text/csv",
        )

st.markdown("""
---

### Additional Privacy Considerations

- **Membership Inference Attacks:** Attackers try to determine if a specific individual's data was used to train a model.  
- **How DP protects against these attacks:** By adding noise during training, Differential Privacy makes it statistically improbable to infer presence of any individual data point in the training set.  
- This is critical for **patient privacy and regulatory compliance**.

---

By combining **Federated Learning** and **Differential Privacy**, MEDPRIV-AI offers a robust framework for healthcare AI that **safeguards patient identity** and meets strict privacy regulations.

""")
