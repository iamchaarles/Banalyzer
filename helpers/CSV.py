import os
import pandas as pd
from PIL import Image
import hashlib
from datetime import datetime

# Configuration
DATA_DIR = 'data'
OUTPUT_CSV = 'banana_ripeness_dataset.csv'
SPLITS = ['train', 'test']  # test folder is used as validation during training
CLASSES = ['unripe', 'ripe', 'over_ripe', 'rotten']  # Base class names
# Actual folder names include the split suffix
FOLDER_MAPPING = {
    'train': {
        'unripe': 'unripe_train',
        'ripe': 'ripe_train',
        'over_ripe': 'over_ripe_train',
        'rotten': 'rotten_train'
    },
    'test': {
        'unripe': 'unripe_test',
        'ripe': 'ripe_test',
        'over_ripe': 'over_ripe_test',
        'rotten': 'rotten_test'
    }
}

def get_image_info(filepath):
    """Extract metadata from image file"""
    try:
        with Image.open(filepath) as img:
            width, height = img.size
            file_size = os.path.getsize(filepath)
            
            # Generate a unique hash for the image
            with open(filepath, 'rb') as f:
                file_hash = hashlib.md5(f.read()).hexdigest()
            
            return {
                'width': width,
                'height': height,
                'file_size_kb': round(file_size / 1024, 2),
                'format': img.format,
                'mode': img.mode,
                'hash': file_hash
            }
    except Exception as e:
        print(f"   Error processing {filepath}: {e}")
        return {
            'width': None,
            'height': None,
            'file_size_kb': None,
            'format': None,
            'mode': None,
            'hash': None
        }

def create_dataset_csv():
    """Create comprehensive CSV file for the banana dataset"""
    
    print("="*70)
    print("Banana Ripeness Dataset - CSV Generator")
    print("="*70)
    
    # Check if data directory exists
    if not os.path.exists(DATA_DIR):
        print(f"\nError: '{DATA_DIR}' directory not found!")
        print(f"   Current directory: {os.getcwd()}")
        print(f"   Please make sure you're running this script from the correct location.")
        return None
    
    print(f"\nFound data directory: {os.path.abspath(DATA_DIR)}")
    
    data_rows = []
    
    # Iterate through all splits and classes
    for split in SPLITS:
        split_dir = os.path.join(DATA_DIR, split)
        
        if not os.path.exists(split_dir):
            print(f"\nWarning: Directory '{split_dir}' not found, skipping...")
            continue
        
        print(f"\nProcessing {split} split...")
        
        # List actual folders in the directory
        actual_folders = [f for f in os.listdir(split_dir) if os.path.isdir(os.path.join(split_dir, f))]
        print(f"   Found folders: {actual_folders}")
        
        for class_name in CLASSES:
            # Get the actual folder name with suffix
            folder_name = FOLDER_MAPPING[split][class_name]
            class_dir = os.path.join(split_dir, folder_name)
            
            if not os.path.exists(class_dir):
                print(f"   Warning: Class directory '{class_dir}' not found!")
                print(f"      Looking for: {folder_name}")
                print(f"      Full path: {os.path.abspath(class_dir)}")
                continue
            
            # Get all image files (including both .png and .jpg)
            all_files = os.listdir(class_dir)
            image_files = [f for f in all_files 
                          if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))]
            
            if len(image_files) == 0:
                print(f"   Warning: {class_name}: No images found!")
                print(f"       Files in directory: {all_files[:5]}")  # Show first 5 files
                continue
            
            print(f"   [OK] {class_name}: {len(image_files)} images")
            
            for idx, filename in enumerate(image_files, 1):
                filepath = os.path.join(class_dir, filename)
                
                # Get image metadata
                img_info = get_image_info(filepath)
                
                # Create relative path for portability
                relative_path = os.path.join(split, folder_name, filename).replace('\\', '/')
                
                # Create row
                row = {
                    'image_id': f"{split}_{class_name}_{idx}",
                    'filename': filename,
                    'relative_path': relative_path,
                    'absolute_path': os.path.abspath(filepath),
                    'class': class_name,
                    'class_id': CLASSES.index(class_name),
                    'split': split,
                    'width': img_info['width'],
                    'height': img_info['height'],
                    'file_size_kb': img_info['file_size_kb'],
                    'format': img_info['format'],
                    'color_mode': img_info['mode'],
                    'file_hash': img_info['hash']
                }
                
                data_rows.append(row)
    
    # Check if we found any images
    if len(data_rows) == 0:
        print("\nError: No images found in any directory!")
        print("\nExpected structure:")
        print("data/")
        print("├── train/")
        print("│   ├── unripe_train/")
        print("│   ├── ripe_train/")
        print("│   ├── over_ripe_train/")
        print("│   └── rotten_train/")
        print("└── test/")
        print("    ├── unripe_test/")
        print("    ├── ripe_test/")
        print("    ├── over_ripe_test/")
        print("    └── rotten_test/")
        return None
    
    # Create DataFrame
    df = pd.DataFrame(data_rows)
    
    # Sort by split and class for better organization
    df = df.sort_values(['split', 'class', 'filename']).reset_index(drop=True)
    
    # Save to CSV
    df.to_csv(OUTPUT_CSV, index=False)
    
    # Print statistics
    print("\n" + "="*70)
    print("📊 Dataset Statistics")
    print("="*70)
    print(f"\nTotal images: {len(df)}")
    print(f"\nBy split:")
    for split in df['split'].unique():
        count = len(df[df['split'] == split])
        print(f"   {split}: {count} images")
    
    print(f"\nBy class:")
    for class_name in CLASSES:
        count = len(df[df['class'] == class_name])
        print(f"   {class_name}: {count} images")
    
    print(f"\nBy split and class:")
    for split in df['split'].unique():
        print(f"\n   {split}:")
        for class_name in CLASSES:
            count = len(df[(df['split'] == split) & (df['class'] == class_name)])
            print(f"      {class_name}: {count} images")
    
    # Image statistics
    valid_images = df[df['width'].notna()]
    if len(valid_images) > 0:
        print(f"\nImage dimensions:")
        print(f"   Average: {valid_images['width'].mean():.0f} x {valid_images['height'].mean():.0f} pixels")
        print(f"   Min: {valid_images['width'].min():.0f} x {valid_images['height'].min():.0f} pixels")
        print(f"   Max: {valid_images['width'].max():.0f} x {valid_images['height'].max():.0f} pixels")
        print(f"\nFile sizes:")
        print(f"   Average: {valid_images['file_size_kb'].mean():.2f} KB")
        print(f"   Total: {valid_images['file_size_kb'].sum() / 1024:.2f} MB")
    
    print(f"\n[SUCCESS] CSV file saved: {OUTPUT_CSV}")
    print("="*70)
    
    # Create a simplified version for Kaggle
    kaggle_df = df[['image_id', 'relative_path', 'class', 'class_id', 'split']]
    kaggle_csv = 'banana_ripeness_labels.csv'
    kaggle_df.to_csv(kaggle_csv, index=False)
    print(f"[SUCCESS] Simplified CSV for Kaggle saved: {kaggle_csv}")
    
    # Create dataset info file
    create_dataset_info(df)
    
    return df

def create_dataset_info(df):
    """Create a README-style info file for the dataset"""
    
    info_file = 'DATASET_INFO.md'
    
    with open(info_file, 'w') as f:
        f.write("# Banana Ripeness Classification Dataset\n\n")
        f.write(f"**Created:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("## Dataset Overview\n\n")
        f.write(f"- **Total Images:** {len(df)}\n")
        f.write(f"- **Classes:** {len(CLASSES)} ({', '.join(CLASSES)})\n")
        f.write(f"- **Splits:** train (80%), test/validation (20%)\n\n")
        
        f.write("## Dataset Structure\n\n")
        f.write("```\n")
        f.write("data/\n")
        f.write("├── train/     (80% - 1000 images for training)\n")
        for class_name in CLASSES:
            folder_name = FOLDER_MAPPING['train'][class_name]
            count = len(df[(df['split'] == 'train') & (df['class'] == class_name)])
            f.write(f"│   ├── {folder_name}/  ({count} images)\n")
        f.write("│\n")
        f.write("└── test/      (20% - 200 images for validation)\n")
        for class_name in CLASSES:
            folder_name = FOLDER_MAPPING['test'][class_name]
            count = len(df[(df['split'] == 'test') & (df['class'] == class_name)])
            f.write(f"    ├── {folder_name}/  ({count} images)\n")
        f.write("```\n\n")
        f.write("**Note:** The 'test' folder is used as validation data during model training.\n\n")
        
        f.write("## Class Distribution\n\n")
        f.write("| Split | Class | Count |\n")
        f.write("|-------|-------|-------|\n")
        for split in ['train', 'test']:
            for class_name in CLASSES:
                count = len(df[(df['split'] == split) & (df['class'] == class_name)])
                f.write(f"| {split} | {class_name} | {count} |\n")
        
        f.write("\n## Class Labels\n\n")
        for idx, class_name in enumerate(CLASSES):
            f.write(f"- **{idx}:** {class_name}\n")
        
        valid_images = df[df['width'].notna()]
        if len(valid_images) > 0:
            f.write("\n## Image Specifications\n\n")
            f.write(f"- **Average Dimensions:** {valid_images['width'].mean():.0f} x {valid_images['height'].mean():.0f} pixels\n")
            
            formats = valid_images['format'].value_counts()
            if len(formats) > 0:
                f.write(f"- **Formats:** {', '.join([f'{fmt} ({cnt})' for fmt, cnt in formats.items()])}\n")
            
            modes = valid_images['color_mode'].value_counts()
            if len(modes) > 0:
                f.write(f"- **Color Modes:** {', '.join([f'{mode} ({cnt})' for mode, cnt in modes.items()])}\n")
            
            f.write(f"- **Average File Size:** {valid_images['file_size_kb'].mean():.2f} KB\n")
        
        f.write("\n## CSV Files\n\n")
        f.write("### banana_ripeness_dataset.csv\n")
        f.write("Complete dataset information with metadata:\n")
        f.write("- `image_id`: Unique identifier\n")
        f.write("- `filename`: Image filename\n")
        f.write("- `relative_path`: Relative path from data root\n")
        f.write("- `absolute_path`: Absolute file path\n")
        f.write("- `class`: Class name (unripe/ripe/overripe/rotten)\n")
        f.write("- `class_id`: Numeric class label (0-3)\n")
        f.write("- `split`: Dataset split (train/test)\n")
        f.write("- `width`, `height`: Image dimensions\n")
        f.write("- `file_size_kb`: File size in KB\n")
        f.write("- `format`: Image format (PNG/JPEG)\n")
        f.write("- `color_mode`: Color mode (RGB/RGBA)\n")
        f.write("- `file_hash`: MD5 hash for deduplication\n\n")
        
        f.write("### banana_ripeness_labels.csv\n")
        f.write("Simplified version for training:\n")
        f.write("- `image_id`: Unique identifier\n")
        f.write("- `relative_path`: Path to image\n")
        f.write("- `class`: Class name\n")
        f.write("- `class_id`: Numeric label\n")
        f.write("- `split`: Dataset split\n\n")
        
        f.write("## Usage Example\n\n")
        f.write("```python\n")
        f.write("import pandas as pd\n")
        f.write("from PIL import Image\n\n")
        f.write("# Load labels\n")
        f.write("df = pd.read_csv('banana_ripeness_labels.csv')\n\n")
        f.write("# Get training data\n")
        f.write("train_df = df[df['split'] == 'train']\n\n")
        f.write("# Load an image\n")
        f.write("img_path = train_df.iloc[0]['relative_path']\n")
        f.write("img = Image.open(f'data/{img_path}')\n")
        f.write("label = train_df.iloc[0]['class']\n")
        f.write("print(f'Loaded {label} banana')\n")
        f.write("```\n\n")
        
        f.write("## Model Training Example\n\n")
        f.write("```python\n")
        f.write("from tensorflow.keras.preprocessing.image import ImageDataGenerator\n\n")
        f.write("# Using folder structure (recommended)\n")
        f.write("train_datagen = ImageDataGenerator(rescale=1./255)\n")
        f.write("train_gen = train_datagen.flow_from_directory(\n")
        f.write("    'data/train/',\n")
        f.write("    target_size=(224, 224),\n")
        f.write("    batch_size=32\n")
        f.write(")\n\n")
        f.write("# OR using CSV\n")
        f.write("df = pd.read_csv('banana_ripeness_labels.csv')\n")
        f.write("train_df = df[df['split'] == 'train']\n")
        f.write("train_gen = train_datagen.flow_from_dataframe(\n")
        f.write("    train_df,\n")
        f.write("    directory='data/',\n")
        f.write("    x_col='relative_path',\n")
        f.write("    y_col='class',\n")
        f.write("    target_size=(224, 224),\n")
        f.write("    batch_size=32\n")
        f.write(")\n")
        f.write("```\n\n")
        
        f.write("## License\n\n")
        f.write("Please add your license information here (e.g., CC BY 4.0, MIT, etc.).\n\n")
        
        f.write("## Author\n\n")
        f.write("Please add your information here.\n\n")
        
        f.write("## Acknowledgments\n\n")
        f.write("If you use this dataset, please cite:\n")
        f.write("```\n")
        f.write("[Your Name]. (2024). Banana Ripeness Classification Dataset.\n")
        f.write("Retrieved from [Kaggle URL]\n")
        f.write("```\n")
    
    print(f"[SUCCESS] Dataset info file created: {info_file}")

if __name__ == "__main__":
    try:
        print(f"Current working directory: {os.getcwd()}\n")
        
        df = create_dataset_csv()
        
        if df is not None:
            print("\nNext steps:")
            print("   1. Review the generated CSV files")
            print("   2. Edit DATASET_INFO.md with your details (license, author, etc.)")
            print("   3. Create a Kaggle account at kaggle.com")
            print("   4. Click 'New Dataset' and upload:")
            print("      - data/ folder (with train/ and test/ subfolders)")
            print("      - banana_ripeness_labels.csv")
            print("      - banana_ripeness_dataset.csv")
            print("      - DATASET_INFO.md (use as description)")
            print("   5. Add relevant tags: computer-vision, classification, food, agriculture")
        
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()