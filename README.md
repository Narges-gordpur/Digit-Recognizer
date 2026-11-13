# 🔢 Multi-Digit Recognition with Neural Networks

<div align="center">

<img src="https://img.shields.io/badge/Deep%20Learning-CRNN%20%2B%20CTC-8A2BE2?style=for-the-badge" />
<img src="https://img.shields.io/badge/Platform-Web%20Based-00A6A6?style=for-the-badge" />
<img src="https://img.shields.io/badge/Inference-ONNX-005CED?style=for-the-badge" />

### Intelligent Number Recognition Powered by Deep Learning

**Recognize multi-digit numbers from images with a custom-trained neural network — directly in your browser, without a backend.**

<br/>

![Python](https://img.shields.io/badge/Python-3.12-3776AB?style=flat-square&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)
![ONNX](https://img.shields.io/badge/ONNX-1.x-005CED?style=flat-square&logo=onnx&logoColor=white)
![JavaScript](https://img.shields.io/badge/JavaScript-ES6-F7DF1E?style=flat-square&logo=javascript&logoColor=black)
![HTML5](https://img.shields.io/badge/HTML5-E34F26?style=flat-square&logo=html5&logoColor=white)
![CSS3](https://img.shields.io/badge/CSS3-1572B6?style=flat-square&logo=css3&logoColor=white)


</div>

---

## 📖 About

A deep learning platform for automatic recognition of multi-digit numbers from images using a **CRNN + CTC neural network**.

The model is trained from scratch using synthetic data with PyTorch, exported to ONNX, and deployed directly in the browser.

## ✨ Features

- 🧠 CRNN architecture with CNN, BiLSTM, and CTC.
- 🔢 Recognition of 1–8 digit sequences.
- 🎨 Synthetic training data generation.
- ⚡ Browser-based inference using ONNX Runtime Web.
- 🚀 No backend or external API required.
- 🖼️ Image upload, drag & drop, and clipboard support.
- 📱 Modern responsive interface with animations.

## 🏗️ Architecture

```text
Input Image (32×256)
        │
        ▼
       CNN
Feature Extraction
        │
        ▼
      BiLSTM
Sequence Modeling
        │
        ▼
   CTC Decoder
        │
        ▼
 Recognized Number
```

| Component | Description |
|---|---|
| CNN | 5 convolutional layers for feature extraction |
| BiLSTM | 2 bidirectional LSTM layers, hidden size 128 |
| CTC | Sequence recognition without character segmentation |
| FC | 11 output classes (blank + digits 0–9) |

**Model Specifications:**
- Approximately 1.2M parameters
- Approximately 3 MB ONNX model
- Input size: 32 × 256
- Maximum sequence length: 8 digits

## 🚀 Installation & Usage

### 1. Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/digit-recognizer.git
cd digit-recognizer
```

### 2. Create a Virtual Environment

```bash
python -m venv .venv
```

Activate it:

```bash
# Windows
.venv\Scripts\Activate.ps1

# Linux / macOS
source .venv/bin/activate
```
### 3. Install Dependencies

```bash
pip install "numpy<2.4" torch onnx onnxscript pillow
```
### 4. Train the Model

```bash
python train.py
```
The trained model will be saved as `crnn_digits.pt`.

### 5. Export to ONNX

```bash
python export_onnx.py
```
### 6. Run the Web Application

```bash
python -m http.server 8000
```
Open in your browser:

```text
http://localhost:8000/
```
> Note: The application must run through HTTP rather than opening `index.html` directly.

## 📁 Project Structure

```text
digit-recognizer/
├── train.py
├── export_onnx.py
├── crnn_digits.pt
├── crnn_digits.onnx
├── index.html
└── README.md
```

## 📊 Training Results

| Epoch | Loss | Exact Match Accuracy |
|---|---:|---:|
| 1 | 2.75 | 0.0% |
| 3 | 0.22 | 88.7% |
| 7 | 0.11 | 93.3% |
| 15 | ~0.05 | ~96% |
| 25 | ~0.03 | ~97% |

*Approximate results. Accuracy is measured using exact sequence matching.*

## 🔬 Technical Details

**Image Preprocessing**
- Grayscale conversion
- Background inversion when needed
- Thresholding and bounding-box detection
- Cropping and proportional resizing
- Padding to 32 × 256
- Normalization to [-1, 1]

**Synthetic Data Generation**
- Random system fonts and font sizes
- Random digit sequences (1–8 digits)
- Rotation and translation
- Optional Gaussian noise and blur

## 🎯 Usage

1. Open the web application.
2. Upload or drag & drop an image.
3. Alternatively, paste an image using `Ctrl + V`.
4. Click **Detect**.
5. View the recognized number.

## ⚠️ Limitations

- Optimized primarily for printed digits.
- Handwritten digit recognition is not optimized.
- Multi-line recognition is not supported.
- Persian and Arabic digits are not currently supported.

---
<div align="center">

### 👨‍💻 Designed & Developed By : Narges Gordpur

Deep Learning · Neural Networks · Computer Vision

<br/>

[![Live Demo](https://img.shields.io/badge/demo-live-brightgreen?style=for-the-badge)](https://Narges-gordpur.github.io/Digit-Recognizer/)

</div>