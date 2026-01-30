---
title: CIFAR-10 CNN Classifier
emoji: 🖼️
colorFrom: purple
colorTo: pink
sdk: docker
app_port: 8000
pinned: false
---


📘 CIFAR-10 Image Classification — PyTorch CNN Project

A beginner-friendly deep learning project using PyTorch, torchvision, and a custom Convolutional Neural Network (CNN) to classify images from the CIFAR-10 dataset.
This project includes:

✔ Dataset loading & normalization
✔ CNN architecture (Conv → ReLU → MaxPool → FC)
✔ Training & evaluation
✔ Visualization of predictions
✔ Saving & loading model
✔ Logging
✔ Virtual environment setup
✔ Clean project structure
✔ (Optional) FastAPI web app for real-time prediction


🔧 Project Structure
cifar10_cnn_webapp/
├── src/
│   ├── config.py
│   ├── model.py
│   ├── data.py
│   ├── train.py
│   ├── evaluate.py
│   ├── visualize.py
│   └── utils.py
│
├── artifacts/
│   └── cnn_model.pth
├── logs/
│   ├── train.log
├── requirements.txt
├── .gitignore
└── README.md

🧱 1. Setup Instructions
✅ Step 1 — Create Virtual Environment

Windows:

python -m venv env
env\Scripts\activate


Mac/Linux:

python3 -m venv env
source env/bin/activate


You should now see (env) before your terminal prompt.

✅ Step 2 — Install Dependencies
pip install -r requirements.txt


This installs PyTorch + FastAPI + matplotlib + other essentials.

✅ Step 3 — Train the CNN

Inside src/:

python train.py


This will:

download CIFAR-10

normalize & load dataset

train for the specified epochs

evaluate after each epoch

save best model → artifacts/cnn_model.pth

generate logs → logs/train.log

✅ Step 4 — Visualize Predictions
python visualize.py


This will display sample test images with:

actual label

predicted label

✅ Step 5 — (Optional) Run Web App for Inference

Coming next:

uvicorn app:app --reload


You will be able to upload an image → get model prediction.

📊 2. Results

Example output after training:

Epoch 1/10 | Batch 100 | Loss: 1.72
Epoch 1 Test Accuracy: 52.81%
Epoch 2 Test Accuracy: 63.14%
...
Model saved successfully at artifacts/cnn_model.pth

🧪 3. How to Load Trained Model (example)
net = SimpleCNN()
net.load_state_dict(torch.load("artifacts/cnn_model.pth"))
net.eval()

🔮 4. Future Improvements

Add more CNN layers

Use dropout for regularization

Add data augmentation

Replace CNN with ResNet-18

Deploy the FastAPI backend

Containerize using Docker