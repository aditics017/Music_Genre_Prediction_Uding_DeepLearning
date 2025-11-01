"""
Music Genre Classifier - Streamlit Web App
Run with: streamlit run app.py
"""

import streamlit as st
import numpy as np
import librosa
import tensorflow as tf
from tensorflow.image import resize
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from PIL import Image
import io
import warnings
warnings.filterwarnings('ignore')

# ============================================
# PAGE CONFIG
# ============================================
st.set_page_config(
    page_title="Music Genre Classifier",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================
# CUSTOM CSS - MODERN & BEAUTIFUL DESIGN
# ============================================
st.markdown("""
<style>
    /* Main background with dark animated gradient */
    .stApp {
        background: linear-gradient(135deg, #0f0c29 0%, #302b63 25%, #24243e 50%, #1a1a2e 75%, #16213e 100%);
        background-size: 400% 400%;
        animation: gradientShift 15s ease infinite;
        min-height: 100vh;
    }
    
    @keyframes gradientShift {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    
    /* Main container with glassmorphism - more transparent to show dark bg */
    .main .block-container {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(20px);
        border-radius: 25px;
        padding: 3rem;
        box-shadow: 0 20px 60px rgba(0, 0, 0, 0.5);
        margin-top: 2rem;
        margin-bottom: 2rem;
        border: 1px solid rgba(102, 126, 234, 0.3);
    }
    
    /* Headers with stunning gradient */
    .main-header {
        font-size: 4.5rem;
        font-weight: 900;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        text-align: center;
        margin-bottom: 0.5rem;
        text-shadow: 0 0 30px rgba(102, 126, 234, 0.5);
        animation: glow 2s ease-in-out infinite alternate;
        letter-spacing: -2px;
    }
    
    @keyframes glow {
        from { filter: drop-shadow(0 0 10px rgba(102, 126, 234, 0.5)); }
        to { filter: drop-shadow(0 0 20px rgba(118, 75, 162, 0.8)); }
    }
    
    .sub-header {
        font-size: 1.5rem;
        color: #ffffff !important;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: 400;
        opacity: 0.95;
    }
    
    /* Prediction box - Glassmorphism with gradient border */
    .prediction-box {
        background: linear-gradient(135deg, rgba(255, 255, 255, 0.9) 0%, rgba(255, 255, 255, 0.95) 100%);
        backdrop-filter: blur(20px);
        padding: 50px;
        border-radius: 30px;
        border: 3px solid transparent;
        background-clip: padding-box;
        margin: 30px 0;
        box-shadow: 
            0 20px 60px rgba(0, 0, 0, 0.15),
            inset 0 1px 0 rgba(255, 255, 255, 0.6);
        position: relative;
        overflow: hidden;
        animation: slideIn 0.5s ease-out;
    }
    
    .prediction-box::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        border-radius: 30px;
        padding: 3px;
        background: linear-gradient(135deg, #667eea, #764ba2, #f093fb);
        -webkit-mask: linear-gradient(#fff 0 0) content-box, linear-gradient(#fff 0 0);
        -webkit-mask-composite: xor;
        mask-composite: exclude;
        z-index: -1;
    }
    
    @keyframes slideIn {
        from {
            opacity: 0;
            transform: translateY(20px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    .prediction-box h1 {
        color: #1a202c !important;
        font-weight: 900 !important;
        font-size: 3.5rem !important;
        text-shadow: 2px 2px 8px rgba(102, 126, 234, 0.2);
        margin: 0;
        animation: pulse 2s ease-in-out infinite;
    }
    
    @keyframes pulse {
        0%, 100% { transform: scale(1); }
        50% { transform: scale(1.02); }
    }
    
    .prediction-box h2 {
        color: #667eea !important;
        font-weight: 700 !important;
        margin-top: 20px !important;
        font-size: 2rem !important;
    }
    
    /* Buttons - Modern 3D effect */
    .stButton>button {
        width: 100%;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        font-weight: 700;
        border: none;
        padding: 18px 30px;
        border-radius: 15px;
        font-size: 1.2rem;
        box-shadow: 
            0 10px 30px rgba(102, 126, 234, 0.4),
            inset 0 1px 0 rgba(255, 255, 255, 0.3);
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        position: relative;
        overflow: hidden;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    .stButton>button::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.3), transparent);
        transition: left 0.5s;
    }
    
    .stButton>button:hover::before {
        left: 100%;
    }
    
    .stButton>button:hover {
        transform: translateY(-3px) scale(1.02);
        box-shadow: 0 15px 40px rgba(102, 126, 234, 0.6);
    }
    
    .stButton>button:active {
        transform: translateY(-1px) scale(0.98);
    }
    
    /* Sidebar - Modern glass effect */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, rgba(102, 126, 234, 0.9) 0%, rgba(118, 75, 162, 0.9) 100%);
        backdrop-filter: blur(20px);
        border-right: 1px solid rgba(255, 255, 255, 0.2);
    }
    
    section[data-testid="stSidebar"] * {
        color: #ffffff !important;
    }
    
    /* Metrics - Beautiful cards */
    div[data-testid="stMetricValue"] {
        color: #667eea !important;
        font-size: 1.8rem !important;
        font-weight: 900 !important;
    }
    
    div[data-testid="stMetricLabel"] {
        color: #4a5568 !important;
        font-weight: 600 !important;
    }
    
    [data-testid="stMetricContainer"] {
        background: rgba(255, 255, 255, 0.7);
        padding: 20px;
        border-radius: 15px;
        backdrop-filter: blur(10px);
        border: 1px solid rgba(102, 126, 234, 0.2);
        box-shadow: 0 5px 15px rgba(0, 0, 0, 0.1);
    }
    
    /* File uploader - Modern design */
    .uploadedFile {
        background: rgba(255, 255, 255, 0.8) !important;
        border-radius: 15px !important;
        padding: 20px !important;
        border: 2px dashed #667eea !important;
        backdrop-filter: blur(10px);
    }
    
    /* Genre cards - 3D effect */
    .genre-card {
        padding: 30px;
        border-radius: 20px;
        text-align: center;
        color: white;
        font-weight: bold;
        box-shadow: 
            0 15px 35px rgba(0, 0, 0, 0.3),
            inset 0 1px 0 rgba(255, 255, 255, 0.3);
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
        position: relative;
        overflow: hidden;
    }
    
    .genre-card::before {
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: radial-gradient(circle, rgba(255, 255, 255, 0.3) 0%, transparent 70%);
        opacity: 0;
        transition: opacity 0.4s;
    }
    
    .genre-card:hover::before {
        opacity: 1;
    }
    
    .genre-card:hover {
        transform: translateY(-10px) scale(1.05) rotate(2deg);
        box-shadow: 0 20px 50px rgba(0, 0, 0, 0.4);
    }
    
    /* Info boxes - Glass effect with dark text */
    .stInfo, .stSuccess, .stWarning, .stError {
        background: rgba(255, 255, 255, 0.9) !important;
        border-radius: 15px !important;
        backdrop-filter: blur(10px);
        border-left: 4px solid;
        padding: 20px !important;
        color: #1a202c !important;
    }
    
    .stInfo p, .stSuccess p, .stWarning p, .stError p,
    .stInfo div, .stSuccess div, .stWarning div, .stError div {
        color: #1a202c !important;
    }
    
    /* Markdown text - WHITE on dark background (main content) */
    .stMarkdown {
        color: #ffffff !important;
    }
    
    /* Main content headings - WHITE on dark background */
    .main h1, .main h2, .main h3, .main h4 {
        color: #ffffff !important;
    }
    
    /* All headings in main content area - WHITE */
    h1, h2, h3, h4, h5, h6 {
        color: #ffffff !important;
    }
    
    /* Paragraph text in main content - WHITE */
    .main p, p {
        color: #ffffff !important;
    }
    
    /* Sections - sidebar text stays white on dark background */
    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3,
    section[data-testid="stSidebar"] h4,
    section[data-testid="stSidebar"] p,
    section[data-testid="stSidebar"] li,
    section[data-testid="stSidebar"] div,
    section[data-testid="stSidebar"] span {
        color: #ffffff !important;
    }
    
    /* Text in RESULT AREAS (white backgrounds) - DARK text */
    .prediction-box,
    .prediction-box h1,
    .prediction-box h2,
    .prediction-box p,
    .prediction-box div {
        color: #1a202c !important;
    }
    
    /* White background sections - dark text */
    div[style*="background: rgba(255, 255, 255"],
    div[style*="background:linear-gradient"][style*="255, 255, 255"],
    div[style*="background: rgba(255,255,255"] {
        color: #1a202c !important;
    }
    
    /* Text in white background boxes */
    [style*="background: rgba(255,255,255,0.7)"],
    [style*="background: rgba(255,255,255,0.9)"],
    [style*="background: rgba(255,255,255,0.95)"] {
        color: #1a202c !important;
    }
    
    [style*="background: rgba(255,255,255,0.7)"] *,
    [style*="background: rgba(255,255,255,0.9)"] *,
    [style*="background: rgba(255,255,255,0.95)"] * {
        color: #1a202c !important;
    }
    
    /* Scrollbar styling */
    ::-webkit-scrollbar {
        width: 10px;
    }
    
    ::-webkit-scrollbar-track {
        background: rgba(255, 255, 255, 0.1);
        border-radius: 10px;
    }
    
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(135deg, #667eea, #764ba2);
        border-radius: 10px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: linear-gradient(135deg, #764ba2, #f093fb);
    }
    
    /* Input fields */
    .stTextInput>div>div>input {
        background: rgba(255, 255, 255, 0.9);
        border-radius: 10px;
        border: 2px solid rgba(102, 126, 234, 0.3);
    }
    
    /* Audio player styling */
    audio {
        width: 100%;
        border-radius: 10px;
    }
    
    /* Ensure all list items have WHITE text */
    .main ul li, .main ol li {
        color: #ffffff !important;
    }
    
    /* All text elements in main content - WHITE */
    .main span, .main div, .main label {
        color: #ffffff !important;
    }
    
    /* File uploader text - WHITE */
    .stFileUploader label, .stFileUploader div, .stFileUploader span {
        color: #ffffff !important;
    }
    
    /* Ensure spinner text is WHITE */
    .stSpinner label {
        color: #ffffff !important;
    }
    
    /* Metrics labels - WHITE */
    div[data-testid="stMetricLabel"] {
        color: #ffffff !important;
    }
    
    /* Metrics values - keep colorful but readable */
    div[data-testid="stMetricValue"] {
        color: #667eea !important;
    }
</style>
""", unsafe_allow_html=True)

# ============================================
# CONSTANTS
# ============================================
GENRES = ['blues', 'classical', 'country', 'disco', 'hiphop', 
          'jazz', 'metal', 'pop', 'reggae', 'rock']

GENRE_EMOJIS = {
    'blues': '🎸',
    'classical': '🎻',
    'country': '🤠',
    'disco': '🕺',
    'hiphop': '🎤',
    'jazz': '🎺',
    'metal': '🤘',
    'pop': '🎤',
    'reggae': '🌴',
    'rock': '🎸'
}

GENRE_COLORS = {
    'blues': '#4A90E2',
    'classical': '#8B572A',
    'country': '#D4A574',
    'disco': '#E91E63',
    'hiphop': '#9C27B0',
    'jazz': '#FF9800',
    'metal': '#424242',
    'pop': '#FF4081',
    'reggae': '#4CAF50',
    'rock': '#F44336'
}

# ============================================
# LOAD MODEL
# ============================================
@st.cache_resource
def load_model():
    """Load the trained model"""
    try:
        model = tf.keras.models.load_model("best_genre_model_final.keras")
        return model, None
    except Exception as e:
        try:
            model = tf.keras.models.load_model("best_genre_model_final.h5")
            return model, None
        except:
            return None, str(e)

# ============================================
# AUDIO PROCESSING FUNCTIONS
# ============================================
def process_audio_chunk(audio_data, sample_rate, target_shape=(150, 150)):
    """Process a single audio chunk and extract features"""
    try:
        # Extract mel-spectrogram
        mel_spectrogram = librosa.feature.melspectrogram(
            y=audio_data, 
            sr=sample_rate
        )
        
        # Resize to target shape
        mel_spectrogram = resize(
            np.expand_dims(mel_spectrogram, axis=-1), 
            target_shape
        )
        
        return mel_spectrogram.numpy()
    except Exception as e:
        st.error(f"Error processing audio: {e}")
        return None

def predict_genre(model, audio_file, duration=30):
    """Predict genre from uploaded audio file"""
    try:
        # Load audio file
        audio_data, sample_rate = librosa.load(audio_file, sr=None, duration=duration)
        
        # Process audio into chunks (same as training)
        chunk_duration = 4
        overlap_duration = 2
        chunk_samples = int(chunk_duration * sample_rate)
        overlap_samples = int(overlap_duration * sample_rate)
        
        num_chunks = int(np.ceil((len(audio_data) - chunk_samples) / 
                                (chunk_samples - overlap_samples))) + 1
        
        predictions = []
        spectrograms = []
        
        # Process each chunk
        for i in range(min(num_chunks, 5)):  # Process max 5 chunks for speed
            start = i * (chunk_samples - overlap_samples)
            end = start + chunk_samples
            
            if end > len(audio_data):
                chunk = audio_data[start:]
                chunk = np.pad(chunk, (0, chunk_samples - len(chunk)), mode='constant')
            else:
                chunk = audio_data[start:end]
            
            # Process chunk
            mel_spec = process_audio_chunk(chunk, sample_rate)
            if mel_spec is not None:
                spectrograms.append(mel_spec)
                
                # Predict
                mel_spec_batch = np.expand_dims(mel_spec, axis=0)
                pred = model.predict(mel_spec_batch, verbose=0)
                predictions.append(pred[0])
        
        # Average predictions
        avg_prediction = np.mean(predictions, axis=0)
        predicted_class = np.argmax(avg_prediction)
        confidence = avg_prediction[predicted_class] * 100
        
        return {
            'genre': GENRES[predicted_class],
            'confidence': confidence,
            'all_probabilities': avg_prediction,
            'spectrograms': spectrograms,
            'audio_data': audio_data,
            'sample_rate': sample_rate
        }
        
    except Exception as e:
        st.error(f"Error during prediction: {e}")
        return None

# ============================================
# VISUALIZATION FUNCTIONS - FIXED
# ============================================
def plot_waveform(audio_data, sample_rate):
    """Plot audio waveform"""
    fig, ax = plt.subplots(figsize=(10, 3), facecolor='#1a1a2e')
    ax.set_facecolor('#16213e')
    time = np.linspace(0, len(audio_data) / sample_rate, num=len(audio_data))
    ax.plot(time, audio_data, color='#667eea', linewidth=0.5)
    ax.set_xlabel('Time (s)', fontsize=10, color='white')
    ax.set_ylabel('Amplitude', fontsize=10, color='white')
    ax.set_title('Audio Waveform', fontsize=12, fontweight='bold', color='white')
    ax.tick_params(colors='white')
    ax.grid(True, alpha=0.2, color='white')
    plt.tight_layout()
    return fig

def plot_spectrogram(spectrogram):
    """Plot mel-spectrogram"""
    fig, ax = plt.subplots(figsize=(8, 4), facecolor='#1a1a2e')
    ax.set_facecolor('#16213e')
    img = ax.imshow(spectrogram[:, :, 0], aspect='auto', origin='lower', cmap='viridis')
    ax.set_xlabel('Time', fontsize=10, color='white')
    ax.set_ylabel('Mel Frequency', fontsize=10, color='white')
    ax.set_title('Mel-Spectrogram', fontsize=12, fontweight='bold', color='white')
    ax.tick_params(colors='white')
    cbar = plt.colorbar(img, ax=ax)
    cbar.ax.tick_params(colors='white')
    plt.tight_layout()
    return fig

def plot_prediction_bars(probabilities):
    """Create interactive bar chart for predictions - FIXED"""
    sorted_indices = np.argsort(probabilities)[::-1]
    sorted_genres = [GENRES[i] for i in sorted_indices]
    sorted_probs = [probabilities[i] * 100 for i in sorted_indices]
    sorted_colors = [GENRE_COLORS[genre] for genre in sorted_genres]
    
    fig = go.Figure(data=[
        go.Bar(
            x=sorted_probs,
            y=sorted_genres,
            orientation='h',
            marker=dict(
                color=sorted_colors,
                line=dict(color='white', width=2)
            ),
            text=[f'{p:.1f}%' for p in sorted_probs],
            textposition='auto',
        )
    ])
    
    fig.update_layout(
        title={
            'text': 'Genre Prediction Probabilities',
            'font': {'color': 'white', 'size': 16, 'family': 'Arial Black'}
        },
        xaxis_title='Confidence (%)',
        yaxis_title='Genre',
        height=400,
        showlegend=False,
        plot_bgcolor='#16213e',
        paper_bgcolor='#1a1a2e',
        font=dict(color='white'),
        xaxis=dict(
            range=[0, 100],
            gridcolor='rgba(255,255,255,0.1)',
            color='white'
        ),
        yaxis=dict(
            gridcolor='rgba(255,255,255,0.1)',
            color='white'
        )
    )
    
    return fig

# ============================================
# MAIN APP
# ============================================
def main():
    # Header with enhanced design
    st.markdown('<p class="main-header">🎵 Music Genre Classifier</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">🎶 Upload an audio file and let AI predict the genre with stunning accuracy! 🎶</p>', unsafe_allow_html=True)
    
    # Load model
    with st.spinner('🔄 Loading AI model... Please wait...'):
        model, error = load_model()
    
    if model is None:
        st.error(f"❌ Failed to load model: {error}")
        st.info("💡 Please make sure 'best_genre_model_final.keras' or 'best_genre_model_final.h5' is in the same directory.")
        return
    
    # Success message with better styling
    st.markdown("""
    <div style="background: linear-gradient(135deg, rgba(76, 175, 80, 0.2), rgba(56, 142, 60, 0.2)); 
                padding: 20px; border-radius: 15px; border-left: 4px solid #4CAF50; margin-bottom: 20px;">
        <h3 style="color: #2e7d32; margin: 0;">✅ Model Loaded Successfully!</h3>
        <p style="color: #1a202c !important; margin: 10px 0 0 0; font-weight: 500;">Ready to classify your music! 🚀</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Enhanced Sidebar
    with st.sidebar:
        st.markdown("""
        <div style="text-align: center; padding: 20px 0;">
            <h1 style="color: white; font-size: 2rem; margin: 0;">🎼 AI Classifier</h1>
            <p style="color: rgba(255,255,255,0.8); margin: 10px 0;">Powered by Deep Learning</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        st.markdown("### 📊 Model Information")
        
        # Genre display with cards
        st.markdown(f"""
        <div style="background: rgba(255,255,255,0.1); padding: 15px; border-radius: 10px; margin-bottom: 20px;">
            <p style="color: white; font-size: 1.1rem; font-weight: bold; margin: 0;">
                Supported Genres: <span style="color: #f093fb;">{len(GENRES)}</span>
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("**🎵 Genres List:**")
        # Display genres in a grid
        cols = st.columns(2)
        for idx, genre in enumerate(GENRES):
            with cols[idx % 2]:
                st.markdown(f"""
                <div style="background: rgba(255,255,255,0.1); padding: 10px; 
                            border-radius: 8px; margin: 5px 0; text-align: center;">
                    <span style="font-size: 1.2rem;">{GENRE_EMOJIS[genre]}</span>
                    <strong style="color: white;">{genre.title()}</strong>
                </div>
                """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Model details in styled box
        st.markdown("### 🔧 Model Details")
        st.markdown("""
        <div style="background: rgba(255,255,255,0.1); padding: 15px; border-radius: 10px;">
            <p style="color: white; margin: 5px 0;">🏗️ <strong>Architecture:</strong> Deep CNN</p>
            <p style="color: white; margin: 5px 0;">📊 <strong>Input:</strong> Mel-Spectrograms</p>
            <p style="color: white; margin: 5px 0;">🎯 <strong>Accuracy:</strong> ~89%</p>
            <p style="color: white; margin: 5px 0;">📚 <strong>Dataset:</strong> GTZAN</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        st.markdown("### 📁 Supported Formats")
        st.markdown("""
        <div style="background: rgba(255,255,255,0.15); padding: 15px; border-radius: 10px; 
                    border: 2px dashed rgba(255,255,255,0.3); text-align: center;">
            <p style="color: white; font-weight: bold; margin: 5px 0;">🎵 WAV</p>
            <p style="color: white; font-weight: bold; margin: 5px 0;">🎵 MP3</p>
            <p style="color: white; font-weight: bold; margin: 5px 0;">🎵 OGG</p>
            <p style="color: white; font-weight: bold; margin: 5px 0;">🎵 FLAC</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Main content with enhanced styling
    st.markdown("""
    <div style="background: linear-gradient(135deg, rgba(102, 126, 234, 0.1), rgba(118, 75, 162, 0.1)); 
                padding: 25px; border-radius: 20px; margin: 20px 0;">
        <h2 style="color: #ffffff !important; text-align: center; margin: 0;">
            📁 Upload Your Audio File
        </h2>
        <p style="color: #ffffff !important; text-align: center; margin: 10px 0 0 0; font-weight: 500;">
            Drag & drop or click to browse
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # File uploader
    uploaded_file = st.file_uploader(
        "🎵 Choose an audio file...",
        type=['wav', 'mp3', 'ogg', 'flac', 'au'],
        help="Upload a music file to classify its genre",
        label_visibility="collapsed"
    )
    
    if uploaded_file is not None:
        # Display file info in styled cards
        st.markdown("### 📋 File Information")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("📄 File Name", uploaded_file.name.split('/')[-1] if '/' in uploaded_file.name else uploaded_file.name)
        with col2:
            file_size_mb = uploaded_file.size / (1024 * 1024)
            if file_size_mb < 1:
                st.metric("💾 File Size", f"{uploaded_file.size / 1024:.1f} KB")
            else:
                st.metric("💾 File Size", f"{file_size_mb:.2f} MB")
        with col3:
            st.metric("🎼 File Type", uploaded_file.type.split('/')[-1] if '/' in uploaded_file.type else uploaded_file.type)
        
        # Audio player with better styling
        st.markdown("### 🎧 Audio Player")
        st.markdown("""
        <div style="background: rgba(102, 126, 234, 0.1); padding: 20px; border-radius: 15px; margin: 10px 0;">
        </div>
        """, unsafe_allow_html=True)
        st.audio(uploaded_file, format='audio/wav')
        
        # Enhanced predict button
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("🚀 Classify Genre", use_container_width=True):
            # Show progress with status messages
            status_container = st.container()
            with status_container:
                status_placeholder = st.empty()
                
            with st.spinner('🎵 Analyzing audio features... This may take a moment...'):
                status_placeholder.info('🔄 Processing audio file...')
                result = predict_genre(model, uploaded_file)
                
                if result:
                    status_placeholder.success('✅ Analysis complete!')
                    import time
                    time.sleep(0.5)  # Brief pause for UX
                    status_placeholder.empty()
            
            if result:
                # Display prediction with animation
                st.markdown("---")
                st.markdown("""
                <div style="text-align: center; margin: 30px 0;">
                    <h2 style="color: #ffffff !important; font-size: 2.5rem; margin: 0;">
                        🎯 Prediction Results
                    </h2>
                    <p style="color: #ffffff !important; font-size: 1.1rem; margin: 10px 0; font-weight: 500;">
                        Discover what genre your music belongs to!
                    </p>
                </div>
                """, unsafe_allow_html=True)
                
                # Main prediction card - WHITE BACKGROUND WITH DARK TEXT
                genre = result['genre']
                confidence = result['confidence']
                emoji = GENRE_EMOJIS[genre]
                
                st.markdown(f"""
                <div class="prediction-box">
                    <h1 style="text-align: center; margin: 0; color: #1a1a2e;">
                        {emoji} {genre.upper()} {emoji}
                    </h1>
                    <h2 style="text-align: center; color: #667eea; margin-top: 10px;">
                        Confidence: {confidence:.1f}%
                    </h2>
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown("---")
                
                # Visualizations section with beautiful headers
                st.markdown("""
                <div style="background: linear-gradient(135deg, rgba(102, 126, 234, 0.1), rgba(118, 75, 162, 0.1)); 
                            padding: 20px; border-radius: 15px; margin: 20px 0;">
                    <h2 style="color: #ffffff !important; text-align: center; margin: 0;">📊 Visualizations</h2>
                    <p style="color: #ffffff !important; text-align: center; margin: 10px 0 0 0; font-weight: 500;">Explore the analysis in detail</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Two columns for visualizations
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("""
                    <div style="background: rgba(102, 126, 234, 0.05); padding: 15px; border-radius: 10px; margin-bottom: 10px;">
                        <h3 style="color: #ffffff !important; margin: 0; text-align: center;">📊 All Genre Probabilities</h3>
                    </div>
                    """, unsafe_allow_html=True)
                    fig_bars = plot_prediction_bars(result['all_probabilities'])
                    st.plotly_chart(fig_bars, use_container_width=True)
                
                with col2:
                    st.markdown("""
                    <div style="background: rgba(118, 75, 162, 0.05); padding: 15px; border-radius: 10px; margin-bottom: 10px;">
                        <h3 style="color: #ffffff !important; margin: 0; text-align: center;">🌊 Audio Waveform</h3>
                    </div>
                    """, unsafe_allow_html=True)
                    fig_wave = plot_waveform(result['audio_data'], result['sample_rate'])
                    st.pyplot(fig_wave)
                
                # Spectrogram with enhanced styling
                if result['spectrograms']:
                    st.markdown("""
                    <div style="background: linear-gradient(135deg, rgba(240, 147, 251, 0.1), rgba(79, 172, 254, 0.1)); 
                                padding: 15px; border-radius: 10px; margin: 20px 0 10px 0;">
                        <h3 style="color: #ffffff !important; margin: 0; text-align: center;">🎨 Mel-Spectrogram Visualization</h3>
                        <p style="color: #ffffff !important; text-align: center; margin: 5px 0 0 0; font-size: 0.9rem; font-weight: 500;">
                            Frequency representation of your audio
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
                    fig_spec = plot_spectrogram(result['spectrograms'][0])
                    st.pyplot(fig_spec)
                
                # Top 3 predictions with enhanced styling
                st.markdown("""
                <div style="background: linear-gradient(135deg, rgba(102, 126, 234, 0.15), rgba(118, 75, 162, 0.15)); 
                            padding: 25px; border-radius: 20px; margin: 30px 0;">
                    <h2 style="color: #ffffff !important; text-align: center; margin: 0;">🏆 Top 3 Predictions</h2>
                    <p style="color: #ffffff !important; text-align: center; margin: 10px 0 0 0; font-weight: 500;">
                        See the most confident predictions
                    </p>
                </div>
                """, unsafe_allow_html=True)
                
                sorted_indices = np.argsort(result['all_probabilities'])[::-1][:3]
                
                cols = st.columns(3)
                for i, idx in enumerate(sorted_indices):
                    with cols[i]:
                        genre_name = GENRES[idx]
                        prob = result['all_probabilities'][idx] * 100
                        rank_emoji = ["🥇", "🥈", "🥉"][i]
                        st.markdown(f"""
                        <div class="genre-card" style="background: linear-gradient(135deg, {GENRE_COLORS[genre_name]}, {GENRE_COLORS[genre_name]}dd);">
                            <div style="font-size: 1.5rem; margin-bottom: 10px;">{rank_emoji}</div>
                            <h3 style="margin: 0; font-size: 3rem;">{GENRE_EMOJIS[genre_name]}</h3>
                            <h4 style="margin: 15px 0 10px 0; font-size: 1.3rem;">{genre_name.title()}</h4>
                            <h2 style="margin: 0; font-size: 2.5rem; text-shadow: 2px 2px 4px rgba(0,0,0,0.3);">{prob:.1f}%</h2>
                            <p style="margin: 10px 0 0 0; font-size: 0.9rem; opacity: 0.9;">Confidence</p>
                        </div>
                        """, unsafe_allow_html=True)
    
    else:
        # Beautiful empty state
        st.markdown("""
        <div style="text-align: center; padding: 60px 20px; background: rgba(255,255,255,0.7); 
                    border-radius: 20px; border: 3px dashed #667eea; margin: 40px 0;">
            <h1 style="font-size: 4rem; margin: 0; color: #667eea;">🎵</h1>
            <h2 style="color: #000000 !important; margin: 20px 0 10px 0; font-weight: 700;">Ready to Classify Your Music?</h2>
            <p style="color: #000000 !important; font-size: 1.1rem; font-weight: 600;">Upload an audio file above to get started!</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Instructions in styled cards
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            <div style="background: rgba(102, 126, 234, 0.15); padding: 25px; border-radius: 15px; 
                        border-left: 4px solid #667eea; height: 100%;">
                <h3 style="color: #ffffff !important; margin-top: 0;">📖 How to Use:</h3>
                <ol style="color: #ffffff !important; line-height: 2; font-weight: 500;">
                    <li>Click <strong style="color: #ffffff !important;">Browse files</strong> or drag & drop</li>
                    <li>Wait for file upload</li>
                    <li>Click <strong style="color: #ffffff !important;">Classify Genre</strong> button</li>
                    <li>View results & visualizations!</li>
                </ol>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div style="background: rgba(118, 75, 162, 0.15); padding: 25px; border-radius: 15px; 
                        border-left: 4px solid #764ba2; height: 100%;">
                <h3 style="color: #ffffff !important; margin-top: 0;">⚡ Pro Tips:</h3>
                <ul style="color: #ffffff !important; line-height: 2; font-weight: 500;">
                    <li>Use clear, high-quality audio</li>
                    <li>Works best with songs >10 seconds</li>
                    <li>Try different genres for fun!</li>
                    <li>Supports WAV, MP3, OGG, FLAC</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)

# ============================================
# RUN APP
# ============================================
if __name__ == "__main__":
    main()