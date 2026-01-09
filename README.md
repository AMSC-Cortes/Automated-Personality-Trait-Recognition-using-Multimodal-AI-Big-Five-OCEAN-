# Automated-Personality-Trait-Recognition-using-Multimodal-AI-Big-Five-OCEAN-
This project develops a data-driven, automated system to identify Big Five (OCEAN) personality traits using multimodal machine learning. It analyzes visual and audio signals from interviews to overcome bias in traditional questionnaires. Built on the ChaLearn dataset, it supports HR, psychology, education, and mental health applications.

Multimodal Personality Analysis (OCEAN Model) 🧠📽️🎙️
This repository contains a Deep Learning system designed to predict the Big Five personality traits (Openness, Conscientiousness, Extraversion, Agreeableness, and Neuroticism) through multimodal analysis of video and audio signals.

🚀 Overview
Traditional personality assessments rely on self-report questionnaires, which can be subjective or biased. This project implements a data-driven approach using Computer Vision and Digital Signal Processing to provide objective, real-time insights into human behavior.

🏗️ Technical Architecture
The system processes data through two specialized branches before fusing them for final prediction:

1. Visual Branch (Video Analysis)
Frame Extraction: Frames are sampled every 1.5 seconds to capture temporal changes in facial expressions and body language.

Feature Extraction: Uses EfficientNetB0 (pre-trained on ImageNet) to convert raw pixels into 1,280 abstract visual features per frame.

Temporal Modeling: An LSTM (Long Short-Term Memory) network processes the sequence of frames to understand motion and behavior over time.

2. Audio Branch (Voice Analysis)
Acoustic Processing: Audio is extracted and converted into Mel-Spectrograms (128 frequency bands).

Normalization: Features are normalized using Z-score to ensure model stability across different recording environments.

Architecture: Implements a simplified Audio Spectrogram Transformer (AST) using Patch Embeddings and Multi-Head Self-Attention to identify vocal energy, pitch, and speaking patterns.

3. Fusion & Prediction
The model merges visual and auditory features through a dense network with Dropout (0.3) to prevent overfitting.

Output: 5 neurons with Sigmoid activation, representing the normalized scores (0 to 1) for each OCEAN trait.

📊 Performance
Loss Function: Mean Squared Error (MSE).

Evaluation Metric: Mean Absolute Error (MAE).

The model achieves high precision by identifying "Peak Signals" in video frames where facial expressions are most discriminative.

📂 Project Structure
Plaintext

├── personalitybig5.py      # Core script (Data cleaning, Model Architecture, Training)

├── features_extraction/    # Extracted .npy features (Visual & Audio)

├── annotations/            # Labels for Train/Val/Test sets

├── best_personality_model.keras  # Saved trained weights

└── README.md
⚠️ Important Note on Data
The dataset is not included in this repository. This project uses the ChaLearn Looking at People dataset.

Privacy: To protect the privacy of the participants (biometric data), you must request access to the original dataset from the official providers.

Setup: Once downloaded, place the videos in the designated folders and run the preprocessing blocks in personalitybig5.py to generate the necessary .npy feature files.

🛠️ Requirements
Python 3.x

TensorFlow / Keras

OpenCV & Librosa

FFmpeg (for audio extraction)

Why not upload the dataset? (Answer to your question)
As established in the README, you should not upload the raw data (videos/audios) for several reasons:

Ethics/Privacy: The videos contain faces and voices of real people. Uploading them to a public repo violates privacy standards.

GitHub Limits: GitHub is for code. A single video dataset can easily exceed 50GB, while GitHub has a 1GB limit per repository (and 100MB per file).

Reproducibility: Professional projects share the code and weights. Other researchers are expected to download the dataset from the original source (ChaLearn) to verify your results.

Results overview

![Results](assets/ResultsExample1.PNG)
![Results](assets/ResultsExample2.PNG)
