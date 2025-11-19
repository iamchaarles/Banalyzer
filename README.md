# 🍌Banalyzer
'A Banana Ripeness Classifier'

A deep learning project that classifies bananas into four ripeness categories: **Unripe**, **Ripe**, **Overripe**, and **Rotten** using transfer learning with MobileNetV2.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

## 📋 Project Overview

This project uses computer vision and deep learning to automatically identify the ripeness stage of bananas from images. It's useful for:
- Food quality control
- Grocery store inventory management
- Reducing food waste
- Learning about image classification and transfer learning

## 🎯 Features

- **Transfer Learning**: Uses pre-trained MobileNetV2 for efficient training
- **4-Class Classification**: Unripe, Ripe, Overripe, Rotten
- **Data Augmentation**: Improves model generalization
- **Easy to Use**: Simple command-line interface
- **Web Interface**: Upload and classify bananas in your browser

## 🏗️ Project Structure

```
banana-ripeness-classifier/
├── data/
│   ├── train/           # Training images
│   └── test/            # Test images
├── models/              # Saved models
├── src/
│   ├── train.py         # Training script
│   └── predict.py       # Prediction script
├── web/                 # Web interface
├── requirements.txt     # Python dependencies
└── README.md
```

## 🚀 Getting Started

### Prerequisites

- Python 3.8 or higher
- pip (Python package manager)
- Virtual environment (recommended)

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/banana-ripeness-classifier.git
cd banana-ripeness-classifier
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

### Dataset Setup

Organize your banana images in the following structure:

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


## 🎓 Training the Model

Run the training script:

```bash
python src/train.py
```

This will:
- Load and augment your dataset
- Build a MobileNetV2-based model
- Train for 20 epochs (with early stopping)
- Save the best model to `models/best_model.keras`
- Generate training history plots

**Training typically takes**: 5-15 minutes on a modern CPU, 2-5 minutes with GPU.

## 🔮 Making Predictions

Test the model on a single image:

```bash
python src/predict.py path/to/your/banana.jpg
```

Example output:
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

## 📊 Model Performance

| Metric | Value |
|--------|-------|
| Architecture | MobileNetV2 + Custom Head |
| Input Size | 224x224x3 |
| Parameters | ~2.5M trainable |
| Training Time | ~10 min (CPU) |
| testing Accuracy | ~85-95% (depends on dataset) |

## 🌐 Web Interface

The project includes a simple web interface for easy testing:

1. Open `web/index.html` in your browser
2. Upload a banana image or use your camera
3. Get instant classification results

## 📈 Future Improvements

- [ ] Add more training data
- [ ] Implement backend API (Flask/FastAPI)
- [ ] Deploy to cloud (Heroku, AWS, Google Cloud)
- [ ] Add multi-banana detection
- [ ] Create mobile app version
- [ ] Add nutritional information based on ripeness

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- TensorFlow and Keras teams for the amazing framework
- MobileNetV2 architecture from Google
- The open-source community

## 📧 Contact

Your Name - 

Project Link: 

---

⭐ If you found this project helpful, please consider giving it a star!