import streamlit as st
import cv2
from deepface import DeepFace
import random
import os
import json
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import time

# --- PAGE CONFIGURATION (Compact Mode) ---
st.set_page_config(
    page_title="ContextReel: Final UI",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- CSS: COMPACT UI & FIT SCREEN ---
st.markdown("""
<style>
    /* 1. Tighten the top whitespace */
    .block-container { padding-top: 0.5rem; padding-bottom: 0rem; }
    
    /* 2. Video Player: 'contain' shows full video, no cropping */
    video { 
        max-height: 65vh !important; 
        width: 100% !important; 
        border-radius: 12px; 
        object-fit: contain !important; 
        box-shadow: 0px 4px 10px rgba(0,0,0,0.4);
    }
    
    /* 3. Make Buttons Compact */
    div.stButton > button {
        width: 100%; height: 45px; font-size: 16px; font-weight: bold; border-radius: 8px; border: none;
        margin-bottom: 0px;
    }
    .dislike-btn > button { background-color: #FF4B4B; color: white; }
    .like-btn > button { background-color: #00CC00; color: white; }
    
    /* Next/Prev Buttons */
    .nav-btn > button { background-color: #1E88E5; color: white; height: 60px; font-size: 18px; }
    
    /* 4. Text & Headers - Reduce Size & Margins */
    h3 { font-size: 20px !important; margin-bottom: 5px !important; padding-top: 0px !important; }
    h4 { font-size: 16px !important; margin-bottom: 0px !important; margin-top: 10px !important; }
    p { font-size: 14px !important; margin-bottom: 5px !important; }
    
    /* 5. Metrics */
    div[data-testid="stMetricValue"] { font-size: 18px; color: #00FF00; }
    
    /* 6. Explainable AI Text Box */
    .xai-box {
        background-color: #1E1E1E;
        padding: 10px;
        border-radius: 8px;
        border-left: 5px solid #00CC00;
        margin-bottom: 10px;
        font-family: monospace;
        font-size: 14px;
    }
</style>
""", unsafe_allow_html=True)

# --- SIDEBAR: LOGIC SETTINGS ---
with st.sidebar:
    st.header("🧠 Logic Controls")
    mood_strategy = st.radio(
        "Recommendation Strategy:",
        ["Cheer Me Up (Regulation)", "Match My Vibe (Congruency)"],
        index=0,
        help="Regulation tries to fix negative moods. Congruency matches them."
    )
    st.info("Dynamic Epsilon is Active: The system trusts Face ID more at the start (or during mood swings), and History more over time.")
    
    st.markdown("---")
    # FACTORY RESET BUTTON (For Demo Purposes)
    if st.button("🗑️ Factory Reset Memory"):
        if os.path.exists("user_profile.json"):
            os.remove("user_profile.json")
        st.session_state.clear()
        st.rerun()

# --- GLOBAL RESOURCES (Shared Camera) ---
@st.cache_resource
def get_camera():
    # Returns the single shared camera instance to avoid "Resource Busy" errors
    return cv2.VideoCapture(0)

# --- DEMOGRAPHIC SCANNER LOGIC ---
def get_initial_demographic_category():
    """
    Scans the camera ONCE at startup to determine Age and Gender.
    Returns: 'gaming', 'news', 'makeup', or 'tech' (fallback).
    """
    cap = get_camera()
    
    # Camera Warmup to allow auto-exposure
    for _ in range(5): 
        cap.read()
        
    ret, frame = cap.read()
    # Note: We do NOT release the cap here because it is cached globally.
    
    start_category = "tech" # Default fallback
    
    if ret:
        try:
            # Enforce detection False prevents crash if face not found immediately
            result = DeepFace.analyze(frame, actions=['age', 'gender'], enforce_detection=False, silent=True)
            gender = result[0]['dominant_gender'] 
            age = result[0]['age']
            
            if gender == "Woman":
                start_category = "makeup"
            elif gender == "Man":
                if age >= 35: start_category = "news"
                else: start_category = "gaming"
            print(f"Startup Scan: Detected {gender}, Age {age} -> Starting with {start_category}")    
        except Exception as e:
            print(f"Startup Scan Failed: {e}")
            
    return start_category

# --- DATABASE LOADER ---
@st.cache_data
def load_local_videos():
    db = []
    base_path = "videos"
    if not os.path.exists(base_path): return []
    video_id = 1
    categories = ["comedy", "gym", "relax", "food", "tech", "gaming", "makeup", "news"]
    
    for cat in categories:
        cat_path = os.path.join(base_path, cat)
        if os.path.exists(cat_path):
            files = [f for f in os.listdir(cat_path) if f.endswith(('.mp4', '.mov', '.avi'))]
            for file in files:
                db.append({
                    "id": video_id, 
                    "title": f"{cat.upper()} #{video_id}", 
                    "path": os.path.join(cat_path, file), 
                    "category": cat
                })
                video_id += 1
    return db

video_db = load_local_videos()

# --- PERSISTENT BRAIN ---
PROFILE_FILE = "user_profile.json"

def load_profile():
    if os.path.exists(PROFILE_FILE):
        try:
            with open(PROFILE_FILE, "r") as f: return json.load(f)
        except: return None
    return None

def save_profile(weights):
    data = {
        "weights": weights,
        "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    with open(PROFILE_FILE, "w") as f:
        json.dump(data, f)

# --- STATE INITIALIZATION ---
saved_data = load_profile()
if 'mood_log' not in st.session_state: st.session_state['mood_log'] = pd.DataFrame(columns=["Time", "Score", "Emotion"])
if 'last_emotion' not in st.session_state: st.session_state['last_emotion'] = "Neutral"
if 'last_emotion_dict' not in st.session_state: st.session_state['last_emotion_dict'] = {"neutral": 100}
if 'explanation' not in st.session_state: st.session_state['explanation'] = "System Initialized. Waiting for input..."

# Default weights
default_weights = {
    "comedy": 1.0, "gym": 1.0, "relax": 1.0, "food": 1.0, 
    "tech": 1.0, "gaming": 1.0, "makeup": 1.0, "news": 1.0
}

# 1. LOAD WEIGHTS (Persistent Preference)
if 'user_weights' not in st.session_state:
    if saved_data:
        st.session_state['user_weights'] = saved_data["weights"]
        for cat in default_weights:
            if cat not in st.session_state['user_weights']:
                st.session_state['user_weights'][cat] = 1.0
    else:
        st.session_state['user_weights'] = default_weights

# 2. INITIALIZE SESSION HISTORY (Always Fresh)
if 'viewed_ids' not in st.session_state: 
    st.session_state['viewed_ids'] = []

# 3. INITIALIZE HISTORY POINTER
if 'history_index' not in st.session_state:
    st.session_state['history_index'] = -1

# 4. INITIALIZE MOOD TRACKER (For Logic Fix)
if 'prev_mood' not in st.session_state:
    st.session_state['prev_mood'] = "neutral"

# --- COLD START LOGIC ---
if 'current_video' not in st.session_state: 
    
    # Run the Demographic Scan
    with st.spinner("🤖 Scanning Face for Personalization..."):
        detected_category = get_initial_demographic_category()
        st.toast(f"Personalization Active: Starting with {detected_category.upper()}", icon="👤")
        st.session_state['explanation'] = f"ℹ️ **Cold Start:** Detected Demographic -> **{detected_category.upper()}**"
    
    # Filter for videos matching the detected demographic
    start_vids = [v for v in video_db if v['category'] == detected_category]
    initial = random.choice(start_vids) if start_vids else (random.choice(video_db) if video_db else None)
    
    st.session_state['current_video'] = initial
    
    # Add First Video to History Immediately
    if initial:
        st.session_state['viewed_ids'].append(initial['id'])
        st.session_state['history_index'] = 0 

# --- CORE LOGIC ---
def get_valence_score(emotion):
    scores = {"happy": 10, "surprise": 8, "neutral": 5, "fear": 3, "sad": 2, "angry": 1, "disgust": 0}
    return scores.get(emotion, 5)

def get_next_video(emotion):
    # Toggleable Strategy
    if "Cheer Me Up" in mood_strategy:
        # Regulation Strategy (Fix the mood)
        target_map = {
            "sad": "comedy", 
            "fear": "comedy", 
            "happy": "gym",      # Happy -> Gym (Use Energy)
            "surprise": "gym",
            "angry": "relax", 
            "disgust": "relax", 
            "neutral": "tech"
        }
        strategy_name = "Regulation"
    else:
        # Congruency Strategy (Match the mood)
        target_map = {
            "sad": "relax", 
            "fear": "relax", 
            "happy": "comedy",   # UPDATED: Happy -> Comedy (Thematic Match)
            "surprise": "tech",
            "angry": "gym", 
            "disgust": "news",   # UPDATED: Disgust -> News (Informative/Neutral)
            "neutral": "news"
        }
        strategy_name = "Congruency"

    bio_target = target_map.get(emotion, "tech")
    weights = st.session_state['user_weights']
    
    # --- LOGIC FIX 1: Smart Epsilon (Mood Shock) ---
    # If the user's mood shifts significantly, TRUST THE FACE immediately.
    # Otherwise, rely on history (RL) as the session gets longer.
    
    prev_mood = st.session_state['prev_mood']
    is_mood_swing = (emotion != prev_mood) and (emotion not in ['neutral', prev_mood])
    
    history_len = len(st.session_state['viewed_ids'])
    
    if is_mood_swing:
        epsilon = 0.95 # Force Biometric
        decision_source = "Biometric (Mood Shift)"
        why_text = f"User shifted from {prev_mood} to {emotion}."
    else:
        # Standard Decay: 0.8 -> 0.2 over time
        epsilon = max(0.2, 0.8 * (0.95 ** history_len))
        decision_source = "History (RL)"
    
    # Update Mood Tracker
    st.session_state['prev_mood'] = emotion

    # --- DECISION MAKING ---
    if random.random() < epsilon or is_mood_swing:
        final_category = bio_target
        decision_source = "Biometric (Face)" if not is_mood_swing else "Biometric (Mood Shift)"
        explanation = f"🤖 **AI Logic:** Detected **{emotion.upper()}**. Strategy ({strategy_name}) chose **{final_category.upper()}**."
    else:
        decision_source = "History (RL)"
        # Weighted Random Choice based on user profile
        safe_weights = [max(0.01, w) for w in weights.values()]
        final_category = random.choices(list(weights.keys()), weights=safe_weights, k=1)[0]
        explanation = f"🧠 **AI Logic:** Biometrics ignored (Epsilon {epsilon:.2f}). Based on history, you love **{final_category.upper()}**."

    st.toast(f"Next: {final_category.upper()} (via {decision_source})", icon="⏭️")
    st.session_state['explanation'] = explanation

    candidates = [v for v in video_db if v['category'] == final_category]
    if not candidates: candidates = video_db
    
    # Filter unseen
    unseen_candidates = [v for v in candidates if v['id'] not in st.session_state['viewed_ids']]
    if not unseen_candidates:
        unseen_candidates = candidates # Reset pool if exhausted
        st.toast(f"Refreshed pool for {final_category.upper()}", icon="🔄")

    winner = random.choice(unseen_candidates)
    
    # Update State
    st.session_state['viewed_ids'].append(winner['id'])
    save_profile(st.session_state['user_weights'])
    
    return winner

# --- LOGIC FIX 2: Normalization Helper ---
def normalize_weights(weights):
    """Keeps total weight sum constant (e.g., 10.0) so one category doesn't dominate."""
    total = sum(weights.values())
    if total == 0: return weights
    # Normalize so they sum to 10
    return {k: round((v / total) * 10, 2) for k, v in weights.items()}

def handle_feedback(feedback_type):
    current_cat = st.session_state['current_video']['category']
    weights = st.session_state['user_weights']
    
    if feedback_type == "like":
        weights[current_cat] += 2.0 # Stronger signal
        msg, icon = f"Liked {current_cat.upper()} (RL Update)", "📈"
    elif feedback_type == "dislike":
        weights[current_cat] = max(0.1, weights[current_cat] - 2.0)
        msg, icon = f"Disliked {current_cat.upper()} (RL Update)", "📉"
    
    # Apply Normalization
    st.session_state['user_weights'] = normalize_weights(weights)
    
    save_profile(st.session_state['user_weights'])
    st.toast(msg, icon=icon)

def scan_and_switch():
    cap = get_camera()
    # Quick warmup for re-scan
    for _ in range(2): cap.read()
    ret, frame = cap.read()
    
    if ret:
        try:
            result = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False, silent=True)
            dominant = result[0]['dominant_emotion']
            emotions = result[0]['emotion']
            st.session_state['last_emotion'] = dominant
            st.session_state['last_emotion_dict'] = emotions
            new_entry = {"Time": datetime.now().strftime("%H:%M:%S"), "Score": get_valence_score(dominant), "Emotion": dominant}
            st.session_state['mood_log'] = pd.concat([st.session_state['mood_log'], pd.DataFrame([new_entry])], ignore_index=True)
            
            # Pass emotion to get_next_video
            st.session_state['current_video'] = get_next_video(dominant)
        except: pass

# --- NAVIGATION LOGIC ---
def nav_previous():
    """Go back in history."""
    if st.session_state['history_index'] > 0:
        st.session_state['history_index'] -= 1
        prev_id = st.session_state['viewed_ids'][st.session_state['history_index']]
        vid = next((v for v in video_db if v['id'] == prev_id), None)
        if vid: 
            st.session_state['current_video'] = vid
            st.session_state['explanation'] = "⏮️ **Replay:** You are watching a previous video from History."

def nav_next():
    """Go forward in history OR generate new if at end."""
    if st.session_state['history_index'] < len(st.session_state['viewed_ids']) - 1:
        st.session_state['history_index'] += 1
        next_id = st.session_state['viewed_ids'][st.session_state['history_index']]
        vid = next((v for v in video_db if v['id'] == next_id), None)
        if vid: 
            st.session_state['current_video'] = vid
            st.session_state['explanation'] = "⏩ **Forward:** Viewing pre-loaded video from History."
    else:
        scan_and_switch()
        st.session_state['history_index'] = len(st.session_state['viewed_ids']) - 1

# --- COMPACT UI LAYOUT ---
col_video, col_controls = st.columns([7, 3])

with col_video:
    if st.session_state['current_video']:
        st.markdown(f"**Playing:** {st.session_state['current_video']['title']}")
        st.video(st.session_state['current_video']['path'], autoplay=True, loop=True)
        # FEATURE: Explainable AI Text
        st.markdown(f"<div class='xai-box'>{st.session_state['explanation']}</div>", unsafe_allow_html=True)

with col_controls:
    st.markdown("### 🎛️ Controls")
    
    # Feedback Row
    c1, c2 = st.columns(2)
    with c1:
        st.markdown('<div class="like-btn">', unsafe_allow_html=True)
        if st.button("👍 LIKE"): handle_feedback("like")
        st.markdown('</div>', unsafe_allow_html=True)
        
    with c2:
        st.markdown('<div class="dislike-btn">', unsafe_allow_html=True)
        if st.button("👎 DISLIKE"): handle_feedback("dislike")
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("---")
    
    # Navigation Row
    nav1, nav2 = st.columns(2)
    with nav1:
        st.markdown('<div class="nav-btn">', unsafe_allow_html=True)
        disable_prev = st.session_state['history_index'] <= 0
        if st.button("⏮️ PREV", disabled=disable_prev, use_container_width=True):
            nav_previous()
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)
        
    with nav2:
        st.markdown('<div class="nav-btn">', unsafe_allow_html=True)
        at_end = st.session_state['history_index'] == len(st.session_state['viewed_ids']) - 1
        label = "⏭️ NEW" if at_end else "▶️ NEXT"
        
        if st.button(label, use_container_width=True):
            nav_next()
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

    # --- IMPROVED GRAPHS ---
    
 # 1. Weights Bar (Fixed UI)
    st.markdown("#### 🧠 User Weights (Normalized)")
    weights = st.session_state['user_weights']
    df_weights = pd.DataFrame(list(weights.items()), columns=["Category", "Weight"])
    
    fig_w = px.bar(df_weights, x="Category", y="Weight", height=180, 
                   color="Weight", color_continuous_scale=["red", "yellow", "green"])
    
    fig_w.update_layout(
        margin=dict(t=10, b=50, l=10, r=10), 
        paper_bgcolor='rgba(0,0,0,0)', 
        plot_bgcolor='rgba(0,0,0,0)', 
        font=dict(color="white"), 
        coloraxis_showscale=False,
        xaxis=dict(title=None, tickangle=-45), 
        yaxis=dict(title=None, showgrid=False) 
    )
    st.plotly_chart(fig_w, use_container_width=True)

    # 2. Emotion Radar
    st.markdown("#### 📡 Last Emotion")
    emo_data = st.session_state['last_emotion_dict']
    fig_radar = go.Figure(go.Scatterpolar(r=list(emo_data.values()), theta=list(emo_data.keys()), fill='toself', line_color='#00CC00'))
    fig_radar.update_layout(polar=dict(radialaxis=dict(visible=False)), margin=dict(t=10, b=10, l=20, r=20), height=120, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color="white", size=10))
    st.plotly_chart(fig_radar, use_container_width=True)
    
    # 3. Mood History
    st.markdown("#### 📉 Mood History")
    if not st.session_state['mood_log'].empty:
        fig_line = px.line(st.session_state['mood_log'], x="Time", y="Score", markers=True, height=120)
        fig_line.update_layout(margin=dict(t=10, b=10, l=0, r=0), 
                               paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', 
                               font=dict(color="white"), yaxis=dict(range=[0, 10]))
        st.plotly_chart(fig_line, use_container_width=True)