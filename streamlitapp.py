import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow import keras
import os

# Page configuration
st.set_page_config(
    page_title="🍌 Banalyzer - Banana Ripeness Classifier",
    page_icon="🍌",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stButton>button {
        width: 100%;
        background-color: #FFD700;
        color: #333;
        font-weight: bold;
        border-radius: 10px;
        padding: 0.75rem;
        border: none;
        font-size: 1.1rem;
    }
    .stButton>button:hover {
        background-color: #FFC700;
        border: none;
    }
    .upload-box {
        border: 2px dashed #FFD700;
        border-radius: 10px;
        padding: 2rem;
        text-align: center;
        background-color: #FFFEF0;
    }
    .result-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        margin: 1rem 0;
    }
    .error-card {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        margin: 1rem 0;
        text-align: center;
    }
    .confidence-high {
        color: #00C853;
        font-weight: bold;
    }
    .confidence-medium {
        color: #FFA726;
        font-weight: bold;
    }
    .confidence-low {
        color: #EF5350;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)

# Configuration
IMG_SIZE_DETECTOR = 224
IMG_SIZE_CLASSIFIER = 640
CLASS_NAMES = ['overripe', 'ripe', 'rotten', 'unripe']

DETECTOR_PATH = 'models/banana_detector_final.keras'
CLASSIFIER_PATH = 'models/banana_classifier_best_model.keras'

# Detection threshold (tune this based on your testing)
DETECTION_THRESHOLD = 0.5

# Class descriptions
CLASS_INFO = {
    'unripe': {
        'emoji': '💚',
        'description': 'Green banana - Not ready to eat yet',
        'tip': 'Wait 2-4 days for optimal ripeness'
    },
    'ripe': {
        'emoji': '💛',
        'description': 'Perfect for eating - Sweet and soft',
        'tip': 'Best time to enjoy! Eat within 1-2 days'
    },
    'overripe': {
        'emoji': '🫥',
        'description': 'Very ripe - Great for baking',
        'tip': 'Perfect for banana bread or smoothies'
    },
    'rotten': {
        'emoji': '❌',
        'description': 'Past its prime - Not recommended',
        'tip': 'Consider composting'
    }
}

@st.cache_resource
def load_detector():
    """Load binary banana detector"""
    if not os.path.exists(DETECTOR_PATH):
        return None
    try:
        model = keras.models.load_model(DETECTOR_PATH)
        return model
    except Exception as e:
        st.error(f"Error loading detector: {e}")
        return None

@st.cache_resource
def load_classifier():
    """Load ripeness classifier"""
    if not os.path.exists(CLASSIFIER_PATH):
        st.error(f"⚠️ Classifier not found at {CLASSIFIER_PATH}")
        return None
    try:
        model = keras.models.load_model(CLASSIFIER_PATH)
        return model
    except Exception as e:
        st.error(f"Error loading classifier: {e}")
        return None

def detect_banana(image, detector_model):
    """
    Detect if image contains a banana using binary classifier
    Returns: (is_banana, confidence, raw_score)
    """
    if detector_model is None:
        # If no detector, assume it's a banana (fallback mode)
        return True, 1.0, 0.0
    
    # Preprocess for detector (224x224)
    img = image.resize((IMG_SIZE_DETECTOR, IMG_SIZE_DETECTOR))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    # Predict
    raw_prediction = detector_model.predict(img_array, verbose=0)[0][0]
    
    # Interpret: Lower score = banana, Higher score = not banana
    # (This depends on how Keras ordered your classes during training)
    is_banana = raw_prediction < DETECTION_THRESHOLD
    confidence = (1 - raw_prediction) if is_banana else raw_prediction
    
    return is_banana, confidence, raw_prediction

def classify_ripeness(image, classifier_model):
    """Classify banana ripeness"""
    # Preprocess for classifier (640x640)
    img = image.resize((IMG_SIZE_CLASSIFIER, IMG_SIZE_CLASSIFIER))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    # Predict
    predictions = classifier_model.predict(img_array, verbose=0)
    
    predicted_idx = np.argmax(predictions[0])
    confidence = predictions[0][predicted_idx]
    predicted_class = CLASS_NAMES[predicted_idx]
    
    return predicted_class, confidence, predictions[0]

def get_confidence_level(confidence):
    """Get confidence level styling"""
    if confidence > 0.8:
        return "High", "confidence-high", "✅"
    elif confidence > 0.6:
        return "Moderate", "confidence-medium", "⚠️"
    else:
        return "Low", "confidence-low", "⚠️"

# Header
st.markdown("""
    <div style='text-align: center; padding: 1rem 0 2rem 0;'>
        <h1>🍌 Banalyzer</h1>
        <p style='font-size: 1.2rem; color: #666;'>AI-Powered Banana Ripeness Classifier</p>
        <p style='color: #888;'>Upload a banana image or take a photo to analyze its ripeness</p>
    </div>
""", unsafe_allow_html=True)

# Load models
detector = load_detector()
classifier = load_classifier()

# Check if models are loaded
if classifier is None:
    st.error("⚠️ Ripeness classifier not found! Please ensure models/best_model.keras exists.")
    st.stop()

# Show detector status
if detector is not None:
    st.success("✅ Two-stage detection active: Binary Detector + Ripeness Classifier")
else:
    st.warning("⚠️ Binary detector not found. Running in classifier-only mode.")
    st.info("To enable banana detection, train the binary classifier: `python train_binary_classifier.py`")

# Create tabs for different input methods
tab1, tab2 = st.tabs(["📁 Upload Image", "📸 Take Photo"])

uploaded_file = None

with tab1:
    st.markdown('<div class="upload-box">', unsafe_allow_html=True)
    uploaded_file = st.file_uploader(
        "Drag and drop or click to upload",
        type=['jpg', 'jpeg', 'png'],
        label_visibility="collapsed"
    )
    st.markdown("Supported formats: JPG, JPEG, PNG")
    st.markdown('</div>', unsafe_allow_html=True)

with tab2:
    camera_photo = st.camera_input("Take a photo of your banana")
    if camera_photo:
        uploaded_file = camera_photo

# Process the image if uploaded
if uploaded_file is not None:
    # Display the uploaded image
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        image = Image.open(uploaded_file).convert('RGB')
        st.image(image, caption="Your Image", use_container_width=True)
    
    # Add analyze button
    if st.button("🔍 Analyze Ripeness"):
        
        # STEP 1: Banana Detection
        if detector is not None:
            with st.spinner("🔍 Step 1/2: Checking for banana..."):
                is_banana, detection_conf, raw_score = detect_banana(image, detector)
            
            # Debug info (can be removed later)
            with st.expander("🔧 Detection Debug Info"):
                st.write(f"Raw score: {raw_score:.4f}")
                st.write(f"Threshold: {DETECTION_THRESHOLD}")
                st.write(f"Is banana: {is_banana}")
                st.write(f"Confidence: {detection_conf*100:.1f}%")
        else:
            is_banana = True
            detection_conf = 1.0
        
        # Check if banana detected
        if not is_banana:
            # NOT A BANANA - Show error
            st.markdown(f"""
                <div class="error-card">
                    <h2>🚫 No Banana Detected!</h2>
                    <p style='font-size: 1.2rem; margin: 1rem 0;'>
                        This doesn't appear to be a banana image.
                    </p>
                    <p style='font-size: 1rem;'>
                        Confidence: {detection_conf*100:.1f}%
                    </p>
                </div>
            """, unsafe_allow_html=True)
            
            st.error("❌ **Detection Failed:** The AI determined this is not a banana")
            
            st.info("""
                **Tips for better detection:**
                - 📸 Take a clear, well-lit photo of the banana
                - 🍌 Make sure the banana fills most of the frame
                - 💡 Use natural lighting when possible
                - 🎯 Avoid cluttered backgrounds
                - 🔄 Try a different angle
            """)
            
            st.warning("Only banana images will be analyzed for ripeness.")
        
        else:
            # BANANA DETECTED - Proceed with ripeness classification
            if detector is not None:
                st.success(f"✅ Step 1/2: Banana detected! (Confidence: {detection_conf*100:.1f}%)")
            
            # STEP 2: Ripeness Classification
            with st.spinner("🔬 Step 2/2: Analyzing ripeness..."):
                predicted_class, confidence, all_probs = classify_ripeness(image, classifier)
            
            confidence_level, confidence_class, confidence_icon = get_confidence_level(confidence)
            
            # Display results
            st.markdown("---")
            st.markdown("## 📊 Ripeness Analysis Results")
            
            # Main prediction card
            class_info = CLASS_INFO[predicted_class]
            st.markdown(f"""
                <div class="result-card">
                    <h2 style='text-align: center; margin: 0;'>
                        {class_info['emoji']} {predicted_class.upper()}
                    </h2>
                    <p style='text-align: center; font-size: 1.1rem; margin: 0.5rem 0;'>
                        {class_info['description']}
                    </p>
                    <p style='text-align: center; font-size: 2rem; font-weight: bold; margin: 0.5rem 0;'>
                        {confidence*100:.1f}%
                    </p>
                </div>
            """, unsafe_allow_html=True)
            
            # Confidence indicator
            st.markdown(f"""
                <div style='text-align: center; padding: 1rem;'>
                    <span class='{confidence_class}'>
                        {confidence_icon} {confidence_level} Confidence
                    </span>
                </div>
            """, unsafe_allow_html=True)
            
            # Recommendation
            st.info(f"💡 **Tip:** {class_info['tip']}")
            
            # All probabilities
            st.markdown("### 📈 Detailed Probabilities")
            
            for i, class_name in enumerate(CLASS_NAMES):
                prob = all_probs[i] * 100
                st.markdown(f"**{CLASS_INFO[class_name]['emoji']} {class_name.capitalize()}**")
                st.progress(float(all_probs[i]))
                st.markdown(f"<p style='text-align: right; margin-top: -1rem;'>{prob:.2f}%</p>", unsafe_allow_html=True)
            
            # Additional info based on confidence
            if confidence_level == "Low":
                st.warning("⚠️ The model is uncertain about this prediction. Consider taking a clearer photo with better lighting.")
            elif confidence_level == "Moderate":
                st.info("ℹ️ Moderate confidence. The banana might be in a transition stage between ripeness levels.")

else:
    # Show example/instructions when no image is uploaded
    st.markdown("---")
    st.markdown("### 📖 How to use:")
    st.markdown("""
    1. **Upload** a banana image or **take a photo** using the tabs above
    2. Click the **Analyze Ripeness** button
    3. Get instant results with confidence scores
    
    ### 🎯 Features:
    """)
    
    if detector is not None:
        st.markdown("""
        - 🔍 **Smart Detection**: Automatically rejects non-banana images
        - 🍌 **Ripeness Classification**: Identifies 4 ripeness stages
        - 💡 **Helpful Tips**: Recommendations for each stage
        - 📊 **Detailed Analysis**: Probability breakdown for all classes
        """)
    else:
        st.markdown("""
        - 🍌 **Ripeness Classification**: Identifies 4 ripeness stages  
        - 💡 **Helpful Tips**: Recommendations for each stage
        - 📊 **Detailed Analysis**: Probability breakdown for all classes
        """)
    
    st.markdown("""
    ### 💡 Tips for best results:
    - Use good lighting
    - Capture the banana clearly in the frame
    - Avoid blurry images
    - Show the banana's color clearly
    - Keep backgrounds simple (if using detector)
    """)

# Footer
st.markdown("---")
st.markdown("""
    <div style='text-align: center; color: #888; padding: 2rem 0;'>
        <p>Made with ❤️ using TensorFlow & Streamlit</p>
        <p style='font-size: 0.9rem;'>Binary Detector + MobileNetV2 Transfer Learning</p>
    </div>
""", unsafe_allow_html=True)
