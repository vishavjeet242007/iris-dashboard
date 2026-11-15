import os
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

# ==========================================
# 1. PAGE CONFIGURATION
# ==========================================
st.set_page_config(
    page_title="Iris Flower AI",
    page_icon="🌸",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================
# 2. AESTHETIC PURPLE THEME, GLASSMORPHISM & EFFECTS
# ==========================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Poppins', sans-serif;
}

/* 1. KEYFRAME ANIMATIONS */
@keyframes gradientDrift {
    0% { background-position: 0% 50%; }
    50% { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
}

@keyframes floatEffect {
    0% { transform: translateY(0px) rotate(0deg); }
    50% { transform: translateY(-10px) rotate(3deg); }
    100% { transform: translateY(0px) rotate(0deg); }
}

@keyframes orbFloat {
    0% { transform: translate(0px, 0px) scale(1); }
    50% { transform: translate(25px, -30px) scale(1.08); }
    100% { transform: translate(0px, 0px) scale(1); }
}

@keyframes textShimmer {
    0% { background-position: 0% 50%; }
    50% { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
}

@keyframes pulseGlow {
    0% { box-shadow: 0 0 18px rgba(168, 85, 247, 0.35), inset 0 0 18px rgba(168, 85, 247, 0.08); }
    50% { box-shadow: 0 0 38px rgba(217, 70, 239, 0.65), inset 0 0 24px rgba(244, 114, 182, 0.14); }
    100% { box-shadow: 0 0 18px rgba(168, 85, 247, 0.35), inset 0 0 18px rgba(168, 85, 247, 0.08); }
}

@keyframes fadeInUp {
    from { opacity: 0; transform: translateY(18px); }
    to { opacity: 1; transform: translateY(0); }
}

/* 2. ANIMATED MESH BACKGROUND */
[data-testid="stAppViewContainer"],
.stApp,
.main {
    background: linear-gradient(-45deg, #13002B, #2A0845, #4C1D95, #6D28D9, #3B0764) !important;
    background-size: 300% 300% !important;
    animation: gradientDrift 18s ease infinite !important;
    color: #FAF5FF !important;
}

[data-testid="stHeader"] {
    background: rgba(19, 0, 43, 0.35) !important;
    backdrop-filter: blur(10px) !important;
}

/* AMBIENT FLOATING ORBS */
.ambient-orb {
    position: fixed;
    border-radius: 50%;
    filter: blur(85px);
    opacity: 0.32;
    pointer-events: none;
    z-index: 0;
    animation: orbFloat 12s ease-in-out infinite;
}
.orb-1 {
    width: 340px;
    height: 340px;
    background: #D946EF;
    top: -60px;
    right: 6%;
}
.orb-2 {
    width: 280px;
    height: 280px;
    background: #818CF8;
    bottom: 5%;
    left: 18%;
    animation-delay: -5s;
}

/* CUSTOM SCROLLBAR */
::-webkit-scrollbar { width: 8px; height: 8px; }
::-webkit-scrollbar-track { background: #14002B; }
::-webkit-scrollbar-thumb {
    background: linear-gradient(180deg, #9333EA, #EC4899);
    border-radius: 10px;
}

/* 3. SIDEBAR IN ROYAL PURPLE */
[data-testid="stSidebar"],
[data-testid="stSidebar"] > div:first-child {
    background: linear-gradient(180deg, rgba(15, 0, 34, 0.96) 0%, rgba(36, 5, 66, 0.96) 100%) !important;
    border-right: 1px solid rgba(192, 132, 252, 0.25) !important;
    backdrop-filter: blur(14px);
}
[data-testid="stSidebar"] * {
    color: #F3E8FF !important;
}

/* 4. TYPOGRAPHY */
h1, h2, h3, h4, h5, h6 {
    color: #FFFFFF !important;
    font-weight: 700 !important;
}
p, span, label, .stMarkdown {
    color: #F3E8FF !important;
}

/* 5. HERO BANNER WITH FLOATING & SHIMMER EFFECTS */
.animated-hero {
    background: linear-gradient(135deg, rgba(147, 51, 234, 0.38) 0%, rgba(76, 29, 149, 0.68) 100%);
    border: 1px solid rgba(216, 180, 254, 0.42);
    padding: 36px 26px;
    border-radius: 26px;
    backdrop-filter: blur(18px);
    -webkit-backdrop-filter: blur(18px);
    text-align: center;
    animation: pulseGlow 4.5s infinite ease-in-out, fadeInUp 0.65s ease-out;
    margin-bottom: 25px;
    position: relative;
    overflow: hidden;
}

.shimmer-title {
    font-size: 2.75rem;
    font-weight: 800;
    margin-bottom: 8px;
    background: linear-gradient(90deg, #FFFFFF, #E9D5FF, #F472B6, #C084FC, #FFFFFF);
    background-size: 300% 300%;
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    animation: textShimmer 5.5s ease infinite;
}

.floating-flower {
    display: inline-block;
    animation: floatEffect 3.5s ease-in-out infinite;
    font-size: 2.5rem;
}

/* 6. GLOWING METRIC CARDS WITH HOVER LIFT */
.animated-metric {
    background: linear-gradient(135deg, rgba(124, 58, 237, 0.45) 0%, rgba(192, 38, 211, 0.38) 100%);
    border: 1px solid rgba(216, 180, 254, 0.38);
    border-radius: 20px;
    padding: 22px 12px;
    text-align: center;
    backdrop-filter: blur(14px);
    box-shadow: 0 8px 24px rgba(0, 0, 0, 0.35);
    transition: all 0.35s cubic-bezier(0.175, 0.885, 0.32, 1.275);
    animation: fadeInUp 0.7s ease-out;
}
.animated-metric:hover {
    transform: translateY(-8px) scale(1.03);
    box-shadow: 0 14px 34px rgba(217, 70, 239, 0.55);
    border-color: rgba(244, 114, 182, 0.85);
}
.animated-metric h2 {
    color: #FFFFFF !important;
    font-size: 2.2rem;
    margin: 0;
    font-weight: 700;
}
.animated-metric p {
    color: #E9D5FF !important;
    font-size: 0.9rem;
    margin-top: 4px;
    margin-bottom: 0;
    text-transform: uppercase;
    letter-spacing: 1.2px;
}

/* 7. FROSTED GLASS CARDS */
.glass-card {
    background: rgba(255, 255, 255, 0.07);
    backdrop-filter: blur(18px);
    -webkit-backdrop-filter: blur(18px);
    border: 1px solid rgba(216, 180, 254, 0.25);
    border-radius: 20px;
    padding: 24px;
    box-shadow: 0 10px 34px 0 rgba(18, 0, 38, 0.42);
    margin-bottom: 22px;
    transition: border-color 0.3s ease, box-shadow 0.3s ease;
    animation: fadeInUp 0.7s ease-out;
}
.glass-card:hover {
    border-color: rgba(216, 180, 254, 0.48);
}

/* 8. SPECIES SHOWCASE CARDS */
.species-card {
    background: rgba(255, 255, 255, 0.07);
    border: 1px solid rgba(216, 180, 254, 0.25);
    border-radius: 18px;
    padding: 22px 18px;
    text-align: center;
    backdrop-filter: blur(14px);
    transition: transform 0.35s cubic-bezier(0.175, 0.885, 0.32, 1.275), box-shadow 0.35s ease, border-color 0.35s ease;
}
.species-card:hover {
    transform: translateY(-7px);
    box-shadow: 0 12px 30px rgba(168, 85, 247, 0.5);
    border-color: rgba(244, 114, 182, 0.7);
}
.badge {
    display: inline-block;
    padding: 5px 14px;
    border-radius: 14px;
    font-size: 0.8rem;
    font-weight: 600;
    margin-top: 10px;
}

/* 9. VIBRANT PURPLE-PINK ACTION BUTTONS WITH SHINE SWEEP */
div.stButton > button {
    position: relative;
    overflow: hidden;
    background: linear-gradient(90deg, #9333EA 0%, #D946EF 50%, #EC4899 100%) !important;
    color: #FFFFFF !important;
    font-weight: 600 !important;
    font-size: 1.05rem !important;
    border: 1px solid rgba(255, 255, 255, 0.2) !important;
    border-radius: 14px !important;
    padding: 12px 28px !important;
    box-shadow: 0 6px 22px rgba(217, 70, 239, 0.45) !important;
    transition: all 0.3s ease !important;
}
div.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 10px 30px rgba(217, 70, 239, 0.75) !important;
}

/* 10. STYLED TABS & SLIDERS */
.stTabs [data-baseweb="tab-list"] {
    gap: 10px;
    background: rgba(255, 255, 255, 0.05);
    padding: 6px;
    border-radius: 14px;
    border: 1px solid rgba(216, 180, 254, 0.2);
}
.stTabs [data-baseweb="tab"] {
    border-radius: 10px;
    color: #E9D5FF !important;
    font-weight: 600;
    padding: 8px 18px;
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(90deg, #9333EA, #D946EF) !important;
    color: #FFFFFF !important;
}

/* TECH PILLS */
.tech-pill {
    display: inline-block;
    background: rgba(168, 85, 247, 0.25);
    border: 1px solid rgba(216, 180, 254, 0.4);
    color: #FAF5FF;
    padding: 6px 14px;
    border-radius: 20px;
    font-size: 0.85rem;
    font-weight: 600;
    margin: 4px;
}
</style>

<div class="ambient-orb orb-1"></div>
<div class="ambient-orb orb-2"></div>
""", unsafe_allow_html=True)

# ==========================================
# 3. DATA LOADING & MODEL TRAINING
# ==========================================
@st.cache_data
def get_dataset():
    candidate_files = [
        "IRIS FLOWER CLASSIFICATION.csv.xlsx",
        "IRIS FLOWER CLASSIFICATION.csv",
        "iris.csv",
        "IRIS.csv",
        "dataset.csv"
    ]
    df = None
    for fname in candidate_files:
        if os.path.exists(fname):
            try:
                df = pd.read_excel(fname) if fname.endswith(".xlsx") else pd.read_csv(fname)
                break
            except Exception:
                continue

    if df is None:
        for f in os.listdir("."):
            if f.endswith((".csv", ".xlsx")) and "iris" in f.lower():
                try:
                    df = pd.read_excel(f) if f.endswith(".xlsx") else pd.read_csv(f)
                    break
                except Exception:
                    pass

    if df is None:
        raw = load_iris()
        df = pd.DataFrame(raw.data, columns=['sepal_length', 'sepal_width', 'petal_length', 'petal_width'])
        df['species'] = pd.Series(raw.target).map({0: 'Setosa', 1: 'Versicolor', 2: 'Virginica'})
    else:
        df.columns = [c.strip().lower().replace(" ", "_").replace("(cm)", "").strip("_") for c in df.columns]
        if 'species' in df.columns:
            df['species'] = df['species'].astype(str).str.replace('Iris-', '', regex=False).str.strip().str.capitalize()

    feature_cols = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
    return df, feature_cols

@st.cache_resource
def train_model(df_data):
    X = df_data[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']]
    y = df_data['species']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    return clf, X, y, X_train, X_test, y_train, y_test, y_pred, acc

df, feature_names = get_dataset()
model, X, y, X_train, X_test, y_train, y_test, y_pred_model, accuracy = train_model(df)
species_counts = df['species'].value_counts().to_dict()

flower_info = {
    "Setosa": {
        "image": "https://upload.wikimedia.org/wikipedia/commons/a/a7/Iris_setosa_-_Botanischer_Garten_M%C3%BCnchen-Nymphenburg_-_DSC09255.JPG",
        "desc": "🌸 **Iris Setosa** is characterized by small, delicate petals and wide sepals. It is distinct and cleanly separable from the other two species."
    },
    "Versicolor": {
        "image": "https://upload.wikimedia.org/wikipedia/commons/4/41/Iris_versicolor_3.jpg",
        "desc": "🌸 **Iris Versicolor** (Blue Flag) features medium-sized petals with soft violet and indigo tones, bridging the gap between Setosa and Virginica."
    },
    "Virginica": {
        "image": "https://upload.wikimedia.org/wikipedia/commons/9/9f/Iris_virginica.jpg",
        "desc": "🌸 **Iris Virginica** has the longest petals and sepals among the three species, showcasing rich, deep purple-violet tones."
    }
}

def apply_purple_theme(fig):
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#FAF5FF", family="Poppins"),
        title_font=dict(color="#FAF5FF", size=16),
        legend=dict(font=dict(color="#FAF5FF"))
    )
    fig.update_xaxes(gridcolor="rgba(216, 180, 254, 0.14)", zerolinecolor="rgba(216, 180, 254, 0.2)")
    fig.update_yaxes(gridcolor="rgba(216, 180, 254, 0.14)", zerolinecolor="rgba(216, 180, 254, 0.2)")
    return fig

# ==========================================
# 4. SIDEBAR NAVIGATION
# ==========================================
st.sidebar.markdown("## 🌸 **Iris AI Hub**")
st.sidebar.markdown("Intelligent flower classification suite.")
st.sidebar.markdown(
    f"<div style='background: rgba(168, 85, 247, 0.22); border: 1px solid rgba(216, 180, 254, 0.35); "
    f"border-radius: 12px; padding: 10px 14px; margin: 10px 0 18px 0; font-size: 0.85rem;'>"
    f"<b>Dataset:</b> {len(df):,} samples<br>"
    f"<b>Accuracy:</b> {accuracy * 100:.2f}%</div>",
    unsafe_allow_html=True
)

page = st.sidebar.radio(
    "Select View:",
    [
        "🏠 Home",
        "🔮 Prediction",
        "📊 Dashboard",
        "ℹ️ About"
    ]
)

# ==========================================
# PAGE 1: 🏠 ANIMATED AESTHETIC HOME
# ==========================================
if page == "🏠 Home":
    st.markdown("""
    <div class="animated-hero">
        <span class="floating-flower">🌸</span>
        <h1 class="shimmer-title">Iris Flower AI Classifier</h1>
        <p style="font-size: 1.12rem; color: #DDD6FE; margin: 0 auto; max-width: 660px;">
            An intelligent machine learning platform for botanical recognition, morphology analysis, and real-time species classification.
        </p>
        <div style="margin-top: 16px;">
            <span style="background: rgba(168, 85, 247, 0.4); padding: 6px 16px; border-radius: 20px; border: 1px solid rgba(216, 180, 254, 0.45); font-size: 0.85rem; color: #FAF5FF; font-weight: 600;">
                ⚡ Model Status: Active & Trained (Random Forest)
            </span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.markdown(f"""
        <div class="animated-metric">
            <h2>{len(df):,}</h2>
            <p>Total Flowers</p>
        </div>
        """, unsafe_allow_html=True)
    with m2:
        st.markdown(f"""
        <div class="animated-metric">
            <h2>{df['species'].nunique()}</h2>
            <p>Species</p>
        </div>
        """, unsafe_allow_html=True)
    with m3:
        st.markdown(f"""
        <div class="animated-metric">
            <h2>{len(feature_names)}</h2>
            <p>Features</p>
        </div>
        """, unsafe_allow_html=True)
    with m4:
        st.markdown(f"""
        <div class="animated-metric">
            <h2>{accuracy * 100:.2f}%</h2>
            <p>Model Accuracy</p>
        </div>
        """, unsafe_allow_html=True)

    st.write("")

    col_img, col_info = st.columns([1.2, 1])
    with col_img:
        st.image(
            "https://upload.wikimedia.org/wikipedia/commons/4/41/Iris_versicolor_3.jpg",
            caption="Specimen: Iris Versicolor (Northern Blue Flag)",
            use_container_width=True
        )
    with col_info:
        st.markdown(f"""
        <div class="glass-card" style="height: 100%;">
            <h3 style="margin-top: 0;">✨ Botanical Intelligence</h3>
            <p>Trained across <b>{len(df):,} botanical records</b>, this model extracts patterns from four physical features:</p>
            <ul style="color: #E9D5FF; line-height: 1.85;">
                <li><b>🌿 Sepal Dimensions:</b> Outer protective calyx measurements.</li>
                <li><b>🌺 Petal Dimensions:</b> Core distinguishing floral proportions.</li>
            </ul>
            <hr style="border: 0; border-top: 1px solid rgba(216, 180, 254, 0.25); margin: 18px 0;">
            <p style="margin-bottom: 6px;"><b>🚀 Quick Navigation:</b></p>
            <p style="font-size: 0.95rem; color: #DDD6FE; margin: 0;">
                Navigate to <b>🔮 Prediction</b> to test custom inputs, or <b>📊 Dashboard</b> to inspect the Confusion Matrix and feature correlations!
            </p>
        </div>
        """, unsafe_allow_html=True)

    st.write("")
    st.markdown("<h3 style='text-align: center; margin-bottom: 20px;'>🌸 Meet the Three Iris Species</h3>", unsafe_allow_html=True)
    s1, s2, s3 = st.columns(3)

    with s1:
        st.markdown(f"""
        <div class="species-card">
            <h4 style="color: #F472B6 !important; margin: 0 0 6px 0;">Iris Setosa</h4>
            <p style="font-size: 0.9rem; color: #DDD6FE; margin: 0;">Smallest petals, high sepal width. Cleanest separation boundary.</p>
            <span class="badge" style="background: rgba(244, 114, 182, 0.28); color: #FCE7F3; border: 1px solid #F472B6;">
                {species_counts.get('Setosa', 0):,} Samples
            </span>
        </div>
        """, unsafe_allow_html=True)

    with s2:
        st.markdown(f"""
        <div class="species-card">
            <h4 style="color: #C084FC !important; margin: 0 0 6px 0;">Iris Versicolor</h4>
            <p style="font-size: 0.9rem; color: #DDD6FE; margin: 0;">Medium petal and sepal sizes. Distinct blue-violet coloration.</p>
            <span class="badge" style="background: rgba(192, 132, 252, 0.28); color: #F3E8FF; border: 1px solid #C084FC;">
                {species_counts.get('Versicolor', 0):,} Samples
            </span>
        </div>
        """, unsafe_allow_html=True)

    with s3:
        st.markdown(f"""
        <div class="species-card">
            <h4 style="color: #818CF8 !important; margin: 0 0 6px 0;">Iris Virginica</h4>
            <p style="font-size: 0.9rem; color: #DDD6FE; margin: 0;">Longest petals and sepals. Deep violet-purple structures.</p>
            <span class="badge" style="background: rgba(129, 140, 248, 0.28); color: #E0E7FF; border: 1px solid #818CF8;">
                {species_counts.get('Virginica', 0):,} Samples
            </span>
        </div>
        """, unsafe_allow_html=True)

# ==========================================
# PAGE 2: 🔮 PREDICTION
# ==========================================
elif page == "🔮 Prediction":
    st.markdown("""
    <div class="animated-hero" style="padding: 24px; margin-bottom: 20px;">
        <h1 class="shimmer-title" style="font-size: 2.2rem;">🔮 Predict Flower Species</h1>
        <p style="color: #DDD6FE; margin: 0;">Adjust the morphological sliders to classify your Iris flower in real time</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### 🌿 Sepal Dimensions")
        sepal_length = st.slider(
            "Sepal Length (cm)",
            min_value=float(round(X['sepal_length'].min() - 0.2, 1)),
            max_value=float(round(X['sepal_length'].max() + 0.2, 1)),
            value=float(round(X['sepal_length'].mean(), 1)),
            step=0.1
        )
        sepal_width = st.slider(
            "Sepal Width (cm)",
            min_value=float(round(X['sepal_width'].min() - 0.2, 1)),
            max_value=float(round(X['sepal_width'].max() + 0.2, 1)),
            value=float(round(X['sepal_width'].mean(), 1)),
            step=0.1
        )

    with col2:
        st.markdown("#### 🌺 Petal Dimensions")
        petal_length = st.slider(
            "Petal Length (cm)",
            min_value=float(round(X['petal_length'].min() - 0.2, 1)),
            max_value=float(round(X['petal_length'].max() + 0.2, 1)),
            value=float(round(X['petal_length'].mean(), 1)),
            step=0.1
        )
        petal_width = st.slider(
            "Petal Width (cm)",
            min_value=float(round(X['petal_width'].min() - 0.1, 1)),
            max_value=float(round(X['petal_width'].max() + 0.2, 1)),
            value=float(round(X['petal_width'].mean(), 1)),
            step=0.1
        )
    st.markdown('</div>', unsafe_allow_html=True)

    predict_btn = st.button("🌸 Classify Flower Now", use_container_width=True)

    if predict_btn:
        input_sample = pd.DataFrame(
            [[sepal_length, sepal_width, petal_length, petal_width]],
            columns=X.columns
        )
        prediction = model.predict(input_sample)[0]
        probabilities = model.predict_proba(input_sample)[0]
        top_conf = max(probabilities) * 100

        st.balloons()

        st.markdown(f"""
        <div style="background: linear-gradient(90deg, #9333EA, #C026D3, #EC4899); padding: 20px; border-radius: 18px; text-align: center; margin: 20px 0; box-shadow: 0 8px 28px rgba(192, 38, 211, 0.5);">
            <h2 style="color: white !important; margin: 0;">🎉 Identified Species: Iris {prediction} ({top_conf:.1f}% Confidence)</h2>
        </div>
        """, unsafe_allow_html=True)

        res_left, res_right = st.columns([1, 1])
        info = flower_info.get(prediction, {
            "image": flower_info["Versicolor"]["image"],
            "desc": f"🌸 **Iris {prediction}** classified by the Random Forest model."
        })

        with res_left:
            st.image(
                info["image"],
                caption=f"Identified: Iris {prediction}",
                use_container_width=True
            )
            st.markdown(f"""
            <div class="glass-card" style="padding: 16px;">
                {info["desc"]}
            </div>
            """, unsafe_allow_html=True)

        with res_right:
            st.markdown("### 📊 Prediction Probabilities")
            prob_df = pd.DataFrame({
                "Species": model.classes_,
                "Confidence (%)": [f"{p * 100:.1f}%" for p in probabilities],
                "Probability": probabilities
            })

            fig_prob = px.bar(
                prob_df,
                x="Probability",
                y="Species",
                orientation='h',
                color="Species",
                color_discrete_map={
                    "Setosa": "#F472B6",
                    "Versicolor": "#C084FC",
                    "Virginica": "#818CF8"
                },
                text="Confidence (%)",
                range_x=[0, 1]
            )
            fig_prob.update_layout(
                showlegend=False,
                height=300,
                xaxis_title="Probability",
                yaxis_title=""
            )
            apply_purple_theme(fig_prob)
            st.plotly_chart(fig_prob, use_container_width=True)

# ==========================================
# PAGE 3: 📊 DASHBOARD & CONFUSION MATRIX
# ==========================================
elif page == "📊 Dashboard":
    st.markdown("""
    <div class="animated-hero" style="padding: 24px; margin-bottom: 20px;">
        <h1 class="shimmer-title" style="font-size: 2.2rem;">📊 Analysis Graphs & Model Evaluation</h1>
        <p style="color: #DDD6FE; margin: 0;">Interactive confusion matrix, correlation heatmap, 3D clusters, and feature distributions</p>
    </div>
    """, unsafe_allow_html=True)

    # Top KPI Row
    k1, k2, k3 = st.columns(3)
    with k1:
        st.markdown(f"""
        <div class="animated-metric">
            <h2>{len(df):,}</h2>
            <p>Total Samples</p>
        </div>
        """, unsafe_allow_html=True)
    with k2:
        st.markdown(f"""
        <div class="animated-metric">
            <h2>{len(X_train):,}</h2>
            <p>Training Samples</p>
        </div>
        """, unsafe_allow_html=True)
    with k3:
        st.markdown(f"""
        <div class="animated-metric">
            <h2>{len(X_test):,}</h2>
            <p>Testing Samples</p>
        </div>
        """, unsafe_allow_html=True)

    st.write("")

    tab1, tab2, tab3 = st.tabs(["🎯 Model Evaluation", "🧬 Correlations & 3D View", "📈 Feature Distributions"])

    with tab1:
        c1, c2 = st.columns([1, 1])
        labels = sorted(y.unique())
        cm = confusion_matrix(y_test, y_pred_model, labels=labels)

        with c1:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.markdown("#### Confusion Matrix")
            fig_cm = px.imshow(
                cm,
                text_auto=True,
                x=labels,
                y=labels,
                color_continuous_scale="Purples",
                labels=dict(x="Predicted Species", y="Actual Species", color="Count")
            )
            fig_cm.update_layout(height=380)
            apply_purple_theme(fig_cm)
            st.plotly_chart(fig_cm, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

        with c2:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.markdown("#### Feature Importance")
            imp_df = pd.DataFrame({
                "Feature": [f.replace("_", " ").title() for f in feature_names],
                "Importance": model.feature_importances_
            }).sort_values("Importance", ascending=True)

            fig_imp = px.bar(
                imp_df,
                x="Importance",
                y="Feature",
                orientation="h",
                color="Importance",
                color_continuous_scale="Magenta",
                text_auto=".3f"
            )
            fig_imp.update_layout(height=380, showlegend=False)
            apply_purple_theme(fig_imp)
            st.plotly_chart(fig_imp, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("#### Classification Report")
        report_dict = classification_report(y_test, y_pred_model, output_dict=True)
        report_df = pd.DataFrame(report_dict).transpose().drop(index="accuracy", errors="ignore").round(3)
        st.dataframe(report_df, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with tab2:
        c3, c4 = st.columns([1, 1.2])
        with c3:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.markdown("#### Feature Correlation Heatmap")
            corr = X.corr().round(2)
            fig_corr = px.imshow(
                corr,
                text_auto=True,
                color_continuous_scale="RdPu"
            )
            fig_corr.update_layout(height=420)
            apply_purple_theme(fig_corr)
            st.plotly_chart(fig_corr, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

        with c4:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.markdown("#### 3D Species Cluster")
            plot_df = df.sample(min(len(df), 1000), random_state=42)
            fig_3d = px.scatter_3d(
                plot_df,
                x="sepal_length",
                y="petal_length",
                z="petal_width",
                color="species",
                color_discrete_map={
                    "Setosa": "#F472B6",
                    "Versicolor": "#C084FC",
                    "Virginica": "#818CF8"
                },
                opacity=0.85
            )
            fig_3d.update_layout(height=420, margin=dict(l=0, r=0, b=0, t=20))
            apply_purple_theme(fig_3d)
            st.plotly_chart(fig_3d, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

    with tab3:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        selected_feat = st.selectbox(
            "Choose Feature to Inspect:",
            feature_names,
            format_func=lambda f: f.replace("_", " ").title()
        )
        fig_box = px.violin(
            df,
            x="species",
            y=selected_feat,
            color="species",
            box=True,
            color_discrete_map={
                "Setosa": "#F472B6",
                "Versicolor": "#C084FC",
                "Virginica": "#818CF8"
            }
        )
        fig_box.update_layout(height=430, showlegend=False)
        apply_purple_theme(fig_box)
        st.plotly_chart(fig_box, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

# ==========================================
# PAGE 4: ℹ️ ABOUT
# ==========================================
elif page == "ℹ️ About":
    st.markdown("""
    <div class="animated-hero" style="padding: 24px; margin-bottom: 20px;">
        <h1 class="shimmer-title" style="font-size: 2.2rem;">ℹ️ About Iris Flower AI</h1>
        <p style="color: #DDD6FE; margin: 0;">Architecture, dataset statistics, and technology stack</p>
    </div>
    """, unsafe_allow_html=True)

    a1, a2 = st.columns([1.1, 1])
    with a1:
        st.markdown("""
        <div class="glass-card">
            <h3 style="margin-top: 0;">🧠 Project Overview</h3>
            <p>
                <b>Iris Flower AI</b> is an interactive machine learning web application built with Streamlit and Scikit-learn.
                It trains a <b>Random Forest Classifier</b> (100 decision trees) on botanical sepal and petal measurements
                to identify <i>Iris Setosa</i>, <i>Iris Versicolor</i>, and <i>Iris Virginica</i> in real time.
            </p>
            <hr style="border: 0; border-top: 1px solid rgba(216, 180, 254, 0.25); margin: 16px 0;">
            <h4>🛠️ Tech Stack</h4>
            <div>
                <span class="tech-pill">Python</span>
                <span class="tech-pill">Streamlit</span>
                <span class="tech-pill">Scikit-Learn</span>
                <span class="tech-pill">Plotly</span>
                <span class="tech-pill">Pandas</span>
                <span class="tech-pill">NumPy</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with a2:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("#### 📋 Dataset Summary Statistics")
        st.dataframe(df[feature_names].describe().round(2), use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown("#### 🔍 Sample Records from Loaded Dataset")
    st.dataframe(df.head(10), use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)
