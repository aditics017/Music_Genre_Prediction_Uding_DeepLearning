# 🎵 Music Genre Classification using Deep Learning

A sophisticated web application that automatically classifies music genres using advanced deep learning techniques. Built with TensorFlow/Keras and Streamlit, featuring a beautiful, modern UI with glassmorphism design.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-Latest-red.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

---

## ✨ Features

- 🎼 **10 Genre Classification**: Blues, Classical, Country, Disco, Hip-hop, Jazz, Metal, Pop, Reggae, Rock
- 🤖 **Deep CNN Architecture**: Advanced convolutional neural network with SE (Squeeze-and-Excitation) attention blocks
- 📊 **Interactive Visualizations**: Real-time waveform, mel-spectrogram, and probability charts
- 🎨 **Beautiful Modern UI**: Dark gradient theme with glassmorphism effects
- ⚡ **Real-time Prediction**: Fast audio processing and classification
- 📱 **Multiple Format Support**: WAV, MP3, OGG, FLAC, AU
- 📈 **High Accuracy**: ~89% test accuracy on GTZAN dataset

---

## 🖼️ Screenshots

### Main Interface
![Main Interface](SS/Screenshot%202025-11-01%20235002.png)

### File Upload
![File Upload](SS/Screenshot%202025-11-01%20235042.png)

### Prediction Results
![Prediction Results](SS/Screenshot%202025-11-01%20235056.png)

### Visualizations
![Visualizations](SS/Screenshot%202025-11-01%20235110.png)

### Detailed Analysis
![Detailed Analysis](SS/Screenshot%202025-11-01%20235328.png)

---

## 🚀 Getting Started

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Installation

1. **Clone the repository** (or download the project)
   ```bash
   git clone <repository-url>
   cd "Music Genre Prediction"
   ```

2. **Install required packages**
   ```bash
   pip install streamlit tensorflow librosa numpy matplotlib plotly pillow
   ```

   Or install from a requirements.txt file:
   ```bash
   pip install -r requirements.txt
   ```

3. **Ensure model files are present**
   - Make sure `best_genre_model_final.keras` or `best_genre_model_final.h5` is in the project directory
   - These files contain the pre-trained model weights

---

## 💻 Usage

### Running the Web Application

1. **Start the Streamlit app**
   ```bash
   streamlit run app.py
   ```

2. **Open in browser**
   - The app will automatically open in your default browser
   - Default URL: `http://localhost:8501`

3. **Upload and classify**
   - Click "Browse files" or drag & drop an audio file
   - Wait for the file to upload
   - Click "🚀 Classify Genre" button
   - View the prediction results and visualizations!

### Training Your Own Model

1. **Open the Jupyter Notebook**
   ```bash
   jupyter notebook Music_Genre_Classifier.ipynb
   ```

2. **Run all cells** to:
   - Load and preprocess audio data
   - Extract mel-spectrogram features
   - Train the deep learning model
   - Evaluate and save the model

3. **Model will be saved** as `best_genre_model_final.keras`

---

## 🏗️ Architecture

### Model Architecture

The model uses a sophisticated **VGG-style CNN** with:

- **4 Convolutional Blocks**: 
  - Block 1: 64 filters
  - Block 2: 128 filters
  - Block 3: 256 filters
  - Block 4: 512 filters

- **SE Attention Blocks**: Squeeze-and-Excitation mechanism for feature enhancement
- **Global Average Pooling**: Instead of flattening for better generalization
- **Dense Layers**: 512 → 256 → 10 (output classes)
- **Regularization**: Batch Normalization, Dropout, L2 regularization

### Input Processing

- **Audio Format**: 30-second clips, 22050 Hz sample rate
- **Feature Extraction**: Mel-spectrograms (128 mel bins, 130 time frames)
- **Normalization**: Min-max scaling to [0, 1] range

---

## 📁 Project Structure

```
Music Genre Prediction/
│
├── app.py                          # Streamlit web application
├── Music_Genre_Classifier.ipynb    # Jupyter notebook for training
├── best_genre_model_final.keras    # Trained model (Keras format)
├── best_genre_model_final.h5       # Trained model (H5 format)
│
├── genres_original/                # GTZAN dataset audio files
│   ├── blues/
│   ├── classical/
│   ├── country/
│   ├── disco/
│   ├── hiphop/
│   ├── jazz/
│   ├── metal/
│   ├── pop/
│   ├── reggae/
│   └── rock/
│
├── SS/                             # Screenshots folder
│   ├── Screenshot 2025-11-01 235002.png
│   ├── Screenshot 2025-11-01 235042.png
│   ├── Screenshot 2025-11-01 235056.png
│   ├── Screenshot 2025-11-01 235110.png
│   └── Screenshot 2025-11-01 235328.png
│
├── images_original/                # Generated spectrogram images
├── features_3_sec.csv              # Extracted features
├── file.csv                        # Dataset metadata
├── training_history.json             # Training logs
│
├── confusion_matrix_final.png      # Confusion matrix visualization
├── accuracy_visualization.png       # Accuracy plot
├── loss_visualization.png           # Loss plot
└── training_curves.png              # Training curves

```

---

## 🔧 Technologies Used

- **Deep Learning**: TensorFlow 2.x, Keras
- **Web Framework**: Streamlit
- **Audio Processing**: librosa
- **Data Processing**: NumPy, Pandas
- **Visualization**: Matplotlib, Plotly
- **UI Design**: Custom CSS with glassmorphism effects

---

## 📊 Model Performance

- **Training Accuracy**: ~99%
- **Validation Accuracy**: ~89%
- **Test Accuracy**: ~89%
- **Dataset**: GTZAN Dataset (10 genres, ~1000 songs)
- **Input Shape**: (128, 130, 1) - Mel-spectrogram
- **Output Classes**: 10 genres

### Confusion Matrix
The model shows excellent performance across most genres, with slight confusion between similar genres (e.g., rock and metal).

---

## 🎯 Supported Genres

| Genre | Emoji | Description |
|-------|-------|-------------|
| Blues | 🎸 | Soulful, emotional music with guitar focus |
| Classical | 🎻 | Orchestral, instrumental compositions |
| Country | 🤠 | Folk-influenced American music |
| Disco | 🕺 | Dance music from the 1970s |
| Hip-hop | 🎤 | Rap and rhythmic beats |
| Jazz | 🎺 | Improvisational, swing music |
| Metal | 🤘 | Heavy, aggressive rock music |
| Pop | 🎤 | Popular, mainstream music |
| Reggae | 🌴 | Jamaican rhythm and blues |
| Rock | 🎸 | Electric guitar-driven music |

---

## 🔍 Key Features Explained

### Audio Processing
- **Mel-Spectrograms**: Convert audio to 2D frequency-time representation
- **Chunk Processing**: Split long audio into 4-second chunks with overlap
- **Normalization**: Scale features for optimal model performance

### Prediction Process
1. Load audio file (up to 30 seconds)
2. Extract mel-spectrogram features
3. Process in chunks (max 5 chunks)
4. Average predictions for final result
5. Display top 3 genres with confidence scores

### Visualization
- **Waveform**: Time-domain audio representation
- **Mel-Spectrogram**: Frequency-time heatmap
- **Probability Chart**: Interactive bar chart of all genre probabilities
- **Top 3 Predictions**: Visual cards with confidence percentages

---

## 🐛 Troubleshooting

### Model Not Loading
- Ensure `best_genre_model_final.keras` or `best_genre_model_final.h5` exists
- Check file permissions
- Verify TensorFlow version compatibility

### Audio Processing Errors
- Ensure audio file is in supported format
- Check file size (recommended: < 200MB)
- Verify audio is not corrupted

### Import Errors
- Install all required packages: `pip install -r requirements.txt`
- Check Python version (3.8+)
- Verify TensorFlow installation

---

## 📝 License

This project is open source and available under the [MIT License](LICENSE).

---

## 👥 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 🙏 Acknowledgments

- **GTZAN Dataset**: For providing the audio dataset
- **TensorFlow/Keras**: For deep learning framework
- **Streamlit**: For web application framework
- **librosa**: For audio processing capabilities

---

## 📧 Contact

For questions or suggestions, please open an issue in the repository.

---

## ⭐ Show Your Support

If you find this project useful, please give it a ⭐ on GitHub!

---

**Made with ❤️ using Deep Learning and Streamlit**

