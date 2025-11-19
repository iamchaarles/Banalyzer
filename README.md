# 🍌 Banalyzer
**A Banana Ripeness Classifier**

A deep learning project that classifies bananas into four ripeness categories: **Unripe**, **Ripe**, **Overripe**, and **Rotten** using transfer learning with MobileNetV2.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Features](#-features)
- [Project Structure](#-project-structure)
- [Installation](#-installation)
- [Dataset Setup](#-dataset-setup)
- [Usage](#-usage)
  - [Training the Model](#-training-the-model)
  - [Making Predictions](#-making-predictions)
  - [Running the Web App](#-running-the-web-app)
- [Model Performance](#-model-performance)
- [Deployment](#-deployment)
- [Future Improvements](#-future-improvements)
- [Contributing](#-contributing)
- [License](#-license)
- [Contact](#-contact)

---

## 🎯 Overview

This project uses computer vision and deep learning to automatically identify the ripeness stage of bananas from images. It's useful for:

- 🏪 Food quality control in grocery stores
- 📦 Inventory management and sorting
- ♻️ Reducing food waste by optimal usage timing
- 🎓 Learning about image classification and transfer learning
- 🔬 Research in agricultural automation

---

## ✨ Features

- **🤖 Transfer Learning**: Leverages pre-trained MobileNetV2 for efficient training
- **🎯 4-Class Classification**: Accurately categorizes as Unripe, Ripe, Overripe, or Rotten
- **📊 Data Augmentation**: Improves model generalization and robustness
- **💻 Command-Line Interface**: Simple prediction script for batch processing
- **🌐 Interactive Web App**: User-friendly Streamlit interface for instant classification
- **📈 Training Visualization**: Detailed plots of training metrics
- **🚀 Production Ready**: Easy to deploy on cloud platforms

---

## 🗂️ Project Structure

```
Banalyzer/
├── src/
│   ├── train.py              # Model training script
│   └── predict.py            # Image prediction script
├── helpers/                  # Utility functions (if any)
├── streamlitapp.py          # Streamlit web application
├── requirements.txt         # Python dependencies
├── DATASET_INFO.md          # Dataset documentation
├── Banana ripeness detection.pdf  # Project documentation
├── .gitignore              # Git ignore rules
└── README.md               # This file
```

**Note**: The `data/` and `models/` folders are not included in this repository due to size constraints.

---

## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- pip (Python package manager)
- Virtual environment (recommended)

### Setup Steps

1. **Clone the repository**

```bash
git clone https://github.com/iamchaarles/Banalyzer.git
cd Banalyzer
```

2. **Create and activate virtual environment**

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Mac/Linux
python -m venv venv
source venv/bin/activate
```

3. **Install dependencies**

```bash
pip install -r requirements.txt
```

---

## 📊 Dataset Setup

The training dataset is not included in this repository due to size constraints. To use this project:

1. Prepare your banana images organized by ripeness category
2. Follow the structure detailed in [DATASET_INFO.md](DATASET_INFO.md)
3. Organize images in the following directory structure:

```
data/
├── train/
│   ├── unripe/
│   ├── ripe/
│   ├── overripe/
│   └── rotten/
└── test/
    ├── unripe_test/
    ├── ripe_test/
    ├── over_ripe_test/
    └── rotten_test/
```

**Dataset Sources:**
- Collect your own banana images at different ripeness stages
- Use public datasets from Kaggle or similar platforms
- Ensure balanced representation across all four categories

---

## 💡 Usage

### 🎓 Training the Model

Train your own model with your dataset:

```bash
python src/train.py
```

**This will:**
- Load and augment your dataset
- Build a MobileNetV2-based transfer learning model
- Train for up to 20 epochs (with early stopping)
- Save the best model to `models/best_model.keras`
- Generate training history plots

**Training time:** 5-15 minutes on modern CPU, 2-5 minutes with GPU

---

### 🔮 Making Predictions

Test the model on individual images using the command line:

```bash
python src/predict.py path/to/your/banana.jpg
```

**Example Output:**

```
============================================================
PREDICTION RESULTS
============================================================

Predicted Class: RIPE
Confidence: 94.32%

All Class Probabilities:
----------------------------------------
unripe      :  2.15% █
ripe        : 94.32% ███████████████████████████████████████████████
overripe    :  3.21% █
rotten      :  0.32% 

============================================================
✓ High confidence prediction
```

---

### 🌐 Running the Web App

Launch the interactive Streamlit web application:

```bash
streamlit run streamlitapp.py
```

**Features:**
- 📤 Drag-and-drop image upload
- 📸 Use your device camera
- ⚡ Real-time classification
- 📊 Confidence score visualization
- 🎨 Beautiful, responsive UI

The app will open in your browser at `http://localhost:8501`

---

## 📈 Model Performance

| Metric | Value |
|--------|-------|
| **Architecture** | MobileNetV2 + Custom Head |
| **Input Size** | 640x640x3 RGB |
| **Total Parameters** | ~2.5M (trainable) |
| **Training Time** | ~10 min (CPU) / ~3 min (GPU) |
| **Validation Accuracy** | ~85-95% (dataset-dependent) |
| **Model Size** | ~15 MB |
| **Inference Time** | <100ms per image |

**Model Architecture:**
- Base: MobileNetV2 (pre-trained on ImageNet)
- Global Average Pooling
- Dropout (0.3)
- Dense Layer (128 units, ReLU)
- Dropout (0.2)
- Output Layer (4 units, Softmax)

---

## 🚀 Deployment

### Deploy on Streamlit Cloud

1. Fork this repository
2. Create a [Streamlit Cloud](https://share.streamlit.io) account
3. Click "New app" and connect your GitHub repository
4. Select `streamlitapp.py` as the main file
5. **Important:** Upload your trained model (`best_model.keras`) to the `models/` folder before deployment

### Deploy on Hugging Face Spaces

1. Create a [Hugging Face](https://huggingface.co) account
2. Create a new Space with Streamlit template
3. Upload all project files including the trained model
4. The app will automatically deploy

### Local Deployment with Docker (Optional)

```bash
# Coming soon - Docker support
```

---

## 🔮 Future Improvements

- [ ] Increase training dataset size and diversity
- [ ] Add multi-banana detection and counting
- [ ] Implement REST API (Flask/FastAPI)
- [ ] Create mobile application (iOS/Android)
- [ ] Add nutritional information based on ripeness
- [ ] Real-time video classification
- [ ] Export to TensorFlow Lite for mobile deployment
- [ ] Add explainability features (Grad-CAM visualizations)
- [ ] Support for other fruits

---

## 🤝 Contributing

Contributions are welcome! Here's how you can help:

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

Please ensure your code follows the project's coding standards and includes appropriate tests.

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- TensorFlow and Keras teams for the amazing deep learning framework
- Google for the MobileNetV2 architecture
- Streamlit for the intuitive web app framework
- The open-source community for inspiration and support

---

## 📧 Contact

**Charles** - [@iamchaarles](https://github.com/iamchaarles)

**Project Link**: [https://github.com/iamchaarles/Banalyzer](https://github.com/iamchaarles/Banalyzer)

---

<div align="center">

⭐ **If you found this project helpful, please consider giving it a star!** ⭐

Made with ❤️ and 🍌

</div>
