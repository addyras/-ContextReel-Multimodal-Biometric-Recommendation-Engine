# -ContextReel-Multimodal-Biometric-Recommendation-Engine
A Zero-Shot recommendation engine that solves the 'Cold Start' problem using real-time facial emotion recognition and Reinforcement Learning. Your face is your login, your mood is the algorithm

# ContextReel: Multimodal Biometric Recommendation Engine

**Student:** Aditya Rastogi
**Student Code:** iitrpr_ai_25010952
**Course:** Minor in Artificial Intelligence (IIT Ropar)
**Module:** E (AI Applications)

---

## 📌 Project Overview
**ContextReel** is a multimodal recommendation engine designed to solve the "Context Blindness" of traditional streaming platforms. While apps like Netflix rely on months of history to know what you like, ContextReel uses Computer Vision to understand how you feel *right now*.

By analyzing facial cues (Age, Gender, Emotion) in real-time, it builds an immediate "Context Vector" to serve relevant content instantly—no login history required. As you interact, a Reinforcement Learning agent takes over, seamlessly transitioning from biometric heuristics to personalized adaptation.

### Key Features
* **Biometric Login:** No passwords. Your face is your context (Age, Gender, Emotion Detection).
* **Mood Regulation:** Detects negative emotions (e.g., Sad) and suggests mood-lifting content (e.g., Comedy).
* **Reinforcement Learning:** Uses an Epsilon-Greedy bandit algorithm to adapt to user preferences (Likes/Dislikes) in real-time.
* **Explainable AI:** Displays exactly *why* a video was recommended (e.g., *"Detected Sadness -> Suggesting Comedy"*).

---

## 📂 Repository Structure
* **`app.py`**: The main interactive Web Application (Streamlit).
* **`ContextReel_Walkthrough.ipynb`**: The core logic walkthrough and evaluation notebook (Primary Submission Artifact).
* **`user_profile.json`**: Persistent storage for RL model weights.
* **`test_face.jpg`**: Sample image used for demonstrating the biometric pipeline in the notebook.
* **`requirements.txt`**: List of Python dependencies.

---

## 🚀 Setup & Installation

### 1. Clone the Repository
```bash
git clone [https://github.com/addyras/-ContextReel-Multimodal-Biometric-Recommendation-Engine.git](https://github.com/addyras/-ContextReel-Multimodal-Biometric-Recommendation-Engine.git)
cd -ContextReel-Multimodal-Biometric-Recommendation-Engine
