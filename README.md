# 🌟 CIFAR-10 Image Classification Web Application  
### **PyTorch · FastAPI · Docker · Hugging Face Spaces · MLOps-Ready**

A production-style, containerized deep-learning web application that performs **real-time image classification** on the CIFAR-10 dataset.  
Built with **PyTorch**, **FastAPI**, and **Docker**, deployed on **Hugging Face Spaces (Docker SDK)**.

This project demonstrates:
- End-to-end ML pipeline (training → inference → deployment)  
- Modular backend architecture  
- Production-style inference pipeline  
- Reproducibility with Docker  
- Logging and error handling (MLOps-friendly)

---

## 🚀 Live Demo
🔗 **https://huggingface.co/spaces/Smiyamaya/cifar10-cnn**

---

## 📸 About This Project

This application classifies images into 10 CIFAR‑10 classes:

> airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck

Includes:
- Clean web UI  
- Image upload + preview  
- Top prediction + confidence bar chart  
- FastAPI backend  
- PyTorch model inference  
- Full Docker deployment  

---

# 🧠 Model Architecture

### SimpleCNN (PyTorch)
- 3× Convolutional layers  
- ReLU activation  
- MaxPooling layers  
- Flatten  
- 2× Fully connected layers  
- Output: 10 classes  

Achieves **70–75% accuracy** on CIFAR‑10.

---

# 📁 Project Layout

```
cifar10_cnn_webapp/
│
├── src/
│   ├── app.py               # FastAPI app + HTML UI
│   ├── model.py             # CNN model architecture
│   ├── predictor.py         # Inference pipeline
│   └── utils.py             # Logging + device helpers
│
├── artifacts/
│   └── cnn_model.pth        # Trained model
│
├── notebooks/
│   └── cifar10_cnn_classification.ipynb
│
├── logs/
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── README.md
```

---

# 🛠️ Local Development

### Install dependencies
```bash
pip install -r requirements.txt
```

### Start app
```bash
uvicorn src.app:app --reload
```

Open: http://localhost:8000

---

# 🐳 Docker Deployment

### Build & run
```bash
docker compose up --build
```

App runs at:
👉 **http://localhost:8000**

---

# 🚀 Deploying on Hugging Face Spaces  
Already configured for **Docker SDK**.

### Required README header
```yaml
---
title: CIFAR-10 CNN Classifier
emoji: 🖼️
colorFrom: purple
colorTo: pink
sdk: docker
app_port: 8000
pinned: false
---
```

### Push to Space
```bash
huggingface-cli login
git remote add space https://huggingface.co/spaces/Smiyamaya/cifar10-cnn
git push --force space main
```

---

# 🔎 Logging & Monitoring
Logs stored in:
```
logs/app.log
```

Tracks uploads, predictions, and errors.

---

# 🛠️ Tech Stack

- PyTorch  
- FastAPI  
- Docker  
- Hugging Face Spaces  
- HTML5 / CSS3  
- Python 3.11  

---

# 🙌 Author
**Maya Murali**  
HuggingFace: https://huggingface.co/Smiyamaya  
