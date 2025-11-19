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

# Custom CSS for better UI
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
IMG_SIZE = 640
CLASS_NAMES = ['overripe', 'ripe', 'rotten', 'unripe']
MODEL_PATH = 'models/best_model.keras'

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
def load_model():
    """Load the trained model"""
    if not os.path.exists(MODEL_PATH):
        return None
    return keras.models.load_model(MODEL_PATH)

def preprocess_image(image):
    """Preprocess image for model prediction"""
    # Resize to model input size
    img = image.resize((IMG_SIZE, IMG_SIZE))
    
    # Convert to array and normalize
    img_array = np.array(img) / 255.0
    
    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)
    
    return img_array

def get_confidence_level(confidence):
    """Get confidence level and styling"""
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

# Load model
model = load_model()

if model is None:
    st.error("⚠️ Model not found! Please train the model first using train.py")
    st.info("Run: `python train.py` to train the model")
    st.stop()

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
        st.image(image, caption="Your Banana", use_container_width=True)
    
    # Add analyze button
    if st.button("🔍 Analyze Ripeness"):
        with st.spinner("Analyzing banana ripeness..."):
            # Preprocess and predict
            img_array = preprocess_image(image)
            predictions = model.predict(img_array, verbose=0)
            
            # Get results
            predicted_idx = np.argmax(predictions[0])
            confidence = predictions[0][predicted_idx]
            predicted_class = CLASS_NAMES[predicted_idx]
            
            confidence_level, confidence_class, confidence_icon = get_confidence_level(confidence)
            
            # Display results
            st.markdown("---")
            st.markdown("## 📊 Analysis Results")
            
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
                prob = predictions[0][i] * 100
                st.markdown(f"**{CLASS_INFO[class_name]['emoji']} {class_name.capitalize()}**")
                st.progress(float(predictions[0][i]))
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
    
    ### 🎯 Tips for best results:
    - Use good lighting
    - Capture the banana clearly in the frame
    - Avoid blurry images
    - Show the banana's color clearly
    """)

# Footer
st.markdown("---")
st.markdown("""
    <div style='text-align: center; color: #888; padding: 2rem 0;'>
        <p>Made with ❤️ using TensorFlow & Streamlit</p>
        <p style='font-size: 0.9rem;'>Model: MobileNetV2 Transfer Learning</p>
    </div>
""", unsafe_allow_html=True)