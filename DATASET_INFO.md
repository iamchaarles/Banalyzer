# 📊 Dataset Information

## Overview

This document describes the dataset structure and requirements for training the Banalyzer banana ripeness classifier.

---

## 📁 Directory Structure

The dataset should be organized in the following structure:

```
data/
├── train/                    # Training images (80% of dataset)
│   ├── unripe/              # Green bananas
│   ├── ripe/                # Yellow bananas (ready to eat)
│   ├── overripe/            # Brown spotted bananas
│   └── rotten/              # Black/heavily damaged bananas
│
└── test/                     # Testing/Validation images (20% of dataset)
    ├── unripe_test/
    ├── ripe_test/
    ├── over_ripe_test/
    └── rotten_test/
```

---

## 🍌 Class Definitions

### 1. **Unripe** 🟢
- **Color**: Bright green to greenish-yellow
- **Characteristics**: 
  - Firm texture
  - No brown spots
  - Starch hasn't converted to sugar yet
- **Stage**: Not ready to eat
- **Recommended images**: 30-50 per category

### 2. **Ripe** 🟡
- **Color**: Bright yellow
- **Characteristics**:
  - Soft but not mushy
  - Few or no brown spots
  - Sweet and flavorful
- **Stage**: Perfect for eating
- **Recommended images**: 30-50 per category

### 3. **Overripe** 🟤
- **Color**: Yellow with brown spots
- **Characteristics**:
  - Very soft texture
  - Multiple brown spots (leopard spotting)
  - Very sweet
- **Stage**: Best for baking/smoothies
- **Recommended images**: 30-50 per category

### 4. **Rotten** ⚫
- **Color**: Mostly black or dark brown
- **Characteristics**:
  - Mushy or decaying
  - Strong odor
  - Not suitable for consumption
- **Stage**: Should be discarded
- **Recommended images**: 30-50 per category

---

## 📸 Image Requirements

### Technical Specifications

| Requirement | Specification |
|------------|---------------|
| **Format** | JPG, JPEG, PNG |
| **Resolution** | Minimum 640x640 pixels (will be resized) |
| **Color Space** | RGB |
| **File Size** | 100KB - 5MB per image |
| **Background** | Varied (white, wooden, countertop, etc.) |

### Quality Guidelines

✅ **DO:**
- Use good lighting conditions
- Capture bananas from different angles
- Include single and bunch of bananas
- Vary backgrounds and settings
- Ensure clear, focused images
- Include different banana sizes

❌ **DON'T:**
- Use blurry or out-of-focus images
- Include watermarks or text overlays
- Use heavily filtered images
- Mix multiple ripeness stages in one image

---

## 📊 Dataset Statistics

### Recommended Dataset Size

| Split | Images per Class | Total Images |
|-------|------------------|--------------|
| **Training** | 40-80 | 160-320 |
| **Testing** | 10-20 | 40-80 |
| **Total** | 50-100 | 200-400 |

**Note**: More images lead to better model performance. Aim for at least 50 images per class.

### Current Dataset Balance

If you're using your own dataset, document it here:

```
Training Set:
- Unripe: X images
- Ripe: X images
- Overripe: X images
- Rotten: X images
Total: X images

Testing Set:
- Unripe: X images
- Ripe: X images
- Overripe: X images
- Rotten: X images
Total: X images
```

---

## 🔍 Data Collection Methods

### Option 1: Collect Your Own Images
1. Purchase bananas at different ripeness stages
2. Take photos over several days as they ripen
3. Use consistent lighting and backgrounds
4. Capture 10-15 images per banana at each stage

### Option 2: Web Scraping (with permission)
- Use image search engines with proper attribution
- Ensure images are free to use (Creative Commons)
- Verify image quality before adding to dataset

### Option 3: Public Datasets
- Check Kaggle for banana/fruit datasets
- Search academic repositories
- GitHub repositories with open datasets

**Suggested Kaggle Datasets:**
- [Banana Quality Dataset](https://www.kaggle.com/search?q=banana+ripeness)
- [Fruit Classification Dataset](https://www.kaggle.com/search?q=fruit+classification)

---

## 🛠️ Data Preprocessing

The training script automatically applies:

### Training Augmentation
- Rescaling: 1/255 (normalization)
- Rotation: ±20 degrees
- Width/Height shift: ±20%
- Horizontal flip: Yes
- Zoom: ±20%
- Brightness: 80-120%

### Validation Preprocessing
- Rescaling only: 1/255

**No manual preprocessing required** - the model handles this automatically!

---

## 📥 Downloading Sample Dataset

If you don't have a dataset, you can:

### Create Your Own Mini Dataset

1. **Week 1**: Buy green bananas (Unripe)
   - Take 20-30 photos
   
2. **Week 1 (Day 3-4)**: Bananas turn yellow (Ripe)
   - Take 20-30 photos
   
3. **Week 2**: Bananas develop spots (Overripe)
   - Take 20-30 photos
   
4. **Week 2 (End)**: Bananas turn brown/black (Rotten)
   - Take 20-30 photos

### Using External Sources

**Important**: Always respect copyright and usage rights!

```bash
# Example: Download from Kaggle (requires Kaggle account)
kaggle datasets download -d <dataset-name>
unzip <dataset-name>.zip -d data/
```

---

## ✅ Dataset Checklist

Before training, ensure:

- [ ] Dataset follows the correct directory structure
- [ ] Each class has at least 30 images
- [ ] Images are in JPG/PNG format
- [ ] No corrupt or unreadable images
- [ ] Classes are balanced (similar number of images)
- [ ] Images represent variety in lighting, angles, and backgrounds
- [ ] Train/test split is approximately 80/20

---

## 🐛 Troubleshooting

### Common Issues

**Issue**: "Found 0 images"
- **Solution**: Check folder structure matches exactly as shown above
- Ensure images are in correct subdirectories

**Issue**: Low accuracy
- **Solution**: Add more diverse images to each class
- Ensure clear distinction between classes

**Issue**: Overfitting
- **Solution**: Add more training images
- Increase data augmentation

---

## 📞 Need Help?

If you need help with dataset preparation:
1. Open an issue in the GitHub repository
2. Check the [README.md](README.md) for training instructions
3. Review the training script: [src/train.py](src/train.py)

---

## 📝 Dataset Attribution

If you're using images from external sources, list them here:

```
Image Sources:
- Source 1: [Name] - [URL] - [License]
- Source 2: [Name] - [URL] - [License]
- Personal collection: X images
```

---

## 📊 Example File Names

Good naming conventions help organize your dataset:

```
train/
  unripe/
    - unripe_001.jpg
    - unripe_002.jpg
    - green_banana_01.jpg
  
  ripe/
    - ripe_001.jpg
    - yellow_banana_01.jpg
    - perfect_banana_01.jpg
  
  overripe/
    - overripe_001.jpg
    - spotted_banana_01.jpg
    - brown_spots_01.jpg
  
  rotten/
    - rotten_001.jpg
    - black_banana_01.jpg
    - decayed_01.jpg
```

---

**Last Updated**: November 2025

**Maintained by**: [@iamchaarles](https://github.com/iamchaarles)
