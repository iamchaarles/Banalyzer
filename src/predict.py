import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from tensorflow import keras
import numpy as np
from PIL import Image
import sys
import os

# Configuration
IMG_SIZE = 640
CLASS_NAMES = ['overripe', 'ripe', 'rotten', 'unripe']
MODEL_PATH = 'models/best_model.keras'

def load_and_preprocess_image(image_path):
    """Load and preprocess a single image"""
    
    # Load image
    img = Image.open(image_path).convert('RGB')
    
    # Resize to model input size
    img = img.resize((IMG_SIZE, IMG_SIZE))
    
    # Convert to array and normalize
    img_array = np.array(img) / 255.0
    
    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)
    
    return img_array

def predict_ripeness(image_path, model_path=MODEL_PATH):
    """Predict banana ripeness from image"""
    
    # Check if model exists
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        print("Please train the model first using train.py")
        return
    
    # Check if image exists
    if not os.path.exists(image_path):
        print(f"Error: Image not found at {image_path}")
        return
    
    print(f"\nLoading model from {model_path}...")
    model = keras.models.load_model(model_path)
    
    print(f"Processing image: {image_path}")
    img_array = load_and_preprocess_image(image_path)
    
    print("Making prediction...")
    predictions = model.predict(img_array, verbose=0)
    
    # Get predicted class and confidence
    predicted_class_idx = np.argmax(predictions[0])
    confidence = predictions[0][predicted_class_idx]
    predicted_class = CLASS_NAMES[predicted_class_idx]
    
    # Display results
    print("\n" + "="*60)
    print("PREDICTION RESULTS")
    print("="*60)
    print(f"\nPredicted Class: {predicted_class.upper()}")
    print(f"Confidence: {confidence*100:.2f}%")
    print("\nAll Class Probabilities:")
    print("-"*40)
    
    for i, class_name in enumerate(CLASS_NAMES):
        prob = predictions[0][i] * 100
        bar = "█" * int(prob / 2)
        print(f"{class_name:12s}: {prob:5.2f}% {bar}")
    
    print("="*60)
    
    # Interpretation
    if confidence > 0.8:
        print("\n✓ High confidence prediction")
    elif confidence > 0.6:
        print("\n⚠ Moderate confidence - consider getting more training data")
    else:
        print("\n⚠ Low confidence - model uncertain")
    
    return predicted_class, confidence, predictions[0]

def main():
    """Main function"""
    
    if len(sys.argv) < 2:
        print("\nUsage: python predict.py <path_to_image>")
        print("\nExample:")
        print("  python predict.py test_banana.jpg")
        print("  python predict.py data/testing/ripe/banana_01.jpg")
        return
    
    image_path = sys.argv[1]
    predict_ripeness(image_path)

if __name__ == "__main__":
    main()