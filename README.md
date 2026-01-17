# 🎬 ContextReel  
### Multimodal Biometric Recommendation Engine  
**Your Face is the Login. Your Mood is the Algorithm.**

---

![ContextReel Banner](https://img.shields.io/badge/AI-Computer%20Vision%20%7C%20Reinforcement%20Learning-blue)
![Streamlit](https://img.shields.io/badge/Frontend-Streamlit-red)
![Status](https://img.shields.io/badge/Status-Working%20Prototype-success)

---

## 🧠 What is ContextReel?

**ContextReel** is a next-generation recommendation engine that solves the **Cold Start** and **Context Blindness** problems in traditional streaming platforms.

Unlike Netflix or YouTube—which require weeks of interaction history—ContextReel understands **who you are and how you feel right now**, using **real-time facial analysis** and **Reinforcement Learning**.

> No login history.  
> No preferences asked.  
> Just instant personalization.

---

## 🚀 Core Idea

Traditional systems ask:
> *“What did you like in the past?”*

ContextReel asks:
> **“Who are you right now?”**

It builds a **Context Vector** using:
- 👤 **Demographics** (Age, Gender)
- 🙂 **Emotion** (Real-time facial expression)
- 🧠 **Reinforcement Learning** (Likes & Dislikes)

Then smoothly transitions from **biometric heuristics → personalized intelligence**.

---

## ✨ Key Features

### 🔐 Biometric Cold Start
- Face scan at startup (no history required)
- Determines **initial content category**
- Solves the *first-user problem*

---

### 🙂 Emotion-Aware Recommendations
- Real-time facial emotion detection
- Supports **Mood Regulation** & **Mood Congruency**
- Example:
  > *Sad → Comedy*  
  > *Angry → Relax*  

---

### 🧠 Reinforcement Learning (Bandit Model)
- **Epsilon-Greedy Strategy**
- Likes/Dislikes update category weights
- Smart epsilon decay with **mood-shock override**
- Learns continuously during the session

---

### 🔍 Explainable AI (XAI)
Every recommendation explains itself:
Detected SAD → Regulation Strategy → COMEDY
Decision Source: Biometric (Mood Shift)

yaml
Copy code

No black-box behavior.

---

### 📊 Live Analytics Dashboard
- User preference weights (normalized)
- Emotion radar chart
- Mood timeline over session

---

## 🏗️ System Architecture

Camera Input
↓
DeepFace (Age | Gender | Emotion)
↓
Context Vector
↓
Decision Engine
(Biometric Logic + RL)
↓
Video Recommendation
↓
User Feedback (Like / Dislike)
↓
Policy Update

yaml
Copy code

---

## 🧪 Technologies Used

| Layer | Tools |
|-----|------|
| Frontend | Streamlit |
| Computer Vision | DeepFace, OpenCV |
| ML Logic | Reinforcement Learning (Epsilon-Greedy Bandit) |
| Data | Pandas, NumPy |
| Visualization | Plotly |
| Persistence | JSON |

---

## 📂 Repository Structure

ContextReel/
│
├── app.py # Streamlit Web App
├── ContextReel_Walkthrough.ipynb # Logic & Evaluation Notebook
├── user_profile.json # Persistent RL Memory
├── test_face.jpg # Demo face image
├── requirements.txt
└── videos/ # Local video database
├── comedy/
├── tech/
├── gym/
├── gaming/
├── relax/
├── food/
├── makeup/
└── news/

yaml
Copy code

---

## ⚙️ Installation & Setup

### 1️⃣ Clone Repository
```bash
git clone https://github.com/addyras/ContextReel-Multimodal-Biometric-Recommendation-Engine.git
cd ContextReel-Multimodal-Biometric-Recommendation-Engine
2️⃣ Install Dependencies
bash
Copy code
pip install -r requirements.txt
3️⃣ Add Video Dataset
Due to GitHub file size limits, videos are not included.

Create this structure:

Copy code
videos/
 ├── comedy/   (3–4 videos)
 ├── tech/
 ├── gym/
 ├── gaming/
 ├── relax/
 ├── food/
 ├── makeup/
 └── news/
4️⃣ Run the Application
bash
Copy code
streamlit run app.py
5️⃣ Run Evaluation Notebook
bash
Copy code
jupyter notebook ContextReel_Walkthrough.ipynb

