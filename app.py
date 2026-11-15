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
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================
# 2. MIDNIGHT EMERALD & BOTANICAL TEAL THEME
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
    50% { transform: translateY(-10px) rotate(4deg); }
    100% { transform: translateY(0px) rotate(0deg); }
}

@keyframes orbFloat {
    0% { transform: translate(0px, 0px) scale(1); }
    50% { transform: translate(28px, -28px) scale(1.08); }
    100% { transform: translate(0px, 0px) scale(1); }
}

@keyframes textShimmer {
    0% { background-position: 0% 50%; }
    50% { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
}

@keyframes pulseGlow {
    0% { box-shadow: 0 0 18px rgba(16, 185, 129, 0.28), inset 0 0 18px rgba(6, 182, 212, 0.08); }
    50% { box-shadow: 0 0 38px rgba(45, 212, 191, 0.55), inset 0 0 24px rgba(16, 185, 129, 0.14); }
    100% { box-shadow: 0 0 18px rgba(16, 185, 129, 0.28), inset 0 0 18px rgba(6, 182, 212, 0.08); }
}

@keyframes fadeInUp {
    from { opacity: 0; transform: translateY(18px); }
    to { opacity: 1; transform: translateY(0); }
}

/* 2. ANIMATED MESH BACKGROUND (DEEP EMERALD & OCEAN TEAL) */
[data-testid="stAppViewContainer"],
.stApp,
.main {
    background: linear-gradient(-45deg, #021412, #052622, #083832, #0B463E, #041D24) !important;
    background-size: 300% 300% !important;
    animation: gradientDrift 18s ease infinite !important;
    color: #ECFDF5 !important;
}

[data-testid="stHeader"] {
    background: rgba(2, 20, 18, 0.45) !important;
    backdrop-filter: blur(12px) !important;
}

/* AMBIENT FLOATING ORBS */
.ambient-orb {
    position: fixed;
    border-radius: 50%;
    filter: blur(90px);
    opacity: 0.28;
    pointer-events: none;
    z-index: 0;
    animation: orbFloat 13s ease-in-out infinite;
}
.orb-1 {
    width: 340px;
    height: 340px;
    background: #10B981;
    top: -60px;
    right: 6%;
}
.orb-2 {
    width: 290px;
    height: 290px;
    background: #06B6D4;
    bottom: 4%;
    left: 16%;
    animation-delay: -5s;
}

/* CUSTOM SCROLLBAR */
::-webkit-scrollbar { width: 8px; height: 8px; }
::-webkit-scrollbar-track { background: #021412; }
::-webkit-scrollbar-thumb {
    background: linear-gradient(180deg, #10B981, #06B6D4);
    border-radius: 10px;
}

/* 3. SIDEBAR IN DEEP FOREST TEAL */
[data-testid="stSidebar"],
[data-testid="stSidebar"] > div:first-child {
    background: linear-gradient(180deg, rgba(2, 18, 16, 0.97) 0%, rgba(6, 40, 35, 0.97) 100%) !important;
    border-right: 1px solid rgba(52, 211, 153, 0.22) !important;
    backdrop-filter: blur(14px);
}
[data-testid="stSidebar"] * {
    color: #ECFDF5 !important;
}

/* 4. TYPOGRAPHY */
h1, h2, h3, h4, h5, h6 {
    color: #FFFFFF !important;
    font-weight: 700 !important;
}
p, span, label, .stMarkdown {
    color: #D1FAE5 !important;
}

/* 5. HERO BANNER WITH FLOATING & SHIMMER EFFECTS */
.animated-hero {
    background: linear-gradient(135deg, rgba(16, 185, 129, 0.22) 0%, rgba(6, 95, 70, 0.55) 100%);
    border: 1px solid rgba(52, 211, 153, 0.38);
    padding: 36px 26px;
    border-radius: 26px;
    backdrop-filter: blur(18px);
    -webkit-backdrop-filter: blur(18px);
    text-align: center;
    animation: pulseGlow 4.5s infinite ease-in-out, fadeInUp 0.65s ease-out;
    margin-bottom: 25px;
    position: relative;
    overflow: hidden;
    transition: transform 0.3s ease, border-color 0.3s ease;
}
.animated-hero:hover {
    border-color: rgba(45, 212, 191, 0.75);
}

.shimmer-title {
    font-size: 2.75rem;
    font-weight: 800;
    margin-bottom: 8px;
    background: linear-gradient(90deg, #FFFFFF, #A7F3D0, #2DD4BF, #FBBF24, #FFFFFF);
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

/* 6. TACTILE GLOWING METRIC CARDS (HOVER + TOUCH PRESS) */
.animated-metric {
    background: linear-gradient(135deg, rgba(16, 185, 129, 0.26) 0%, rgba(6, 182, 212, 0.22) 100%);
    border: 1px solid rgba(52, 211, 153, 0.38);
    border-radius: 20px;
    padding: 22px 12px;
    text-align: center;
    backdrop-filter: blur(14px);
    box-shadow: 0 8px 24px rgba(0, 0, 0, 0.35);
    transition: transform 0.28s cubic-bezier(0.175, 0.885, 0.32, 1.275), box-shadow 0.28s ease, border-color 0.28s ease;
    animation: fadeInUp 0.7s ease-out;
    cursor: pointer;
    user-select: none;
}
.animated-metric:hover {
    transform: translateY(-8px) scale(1.03);
    box-shadow: 0 14px 34px rgba(16, 185, 129, 0.45);
    border-color: rgba(45, 212, 191, 0.9);
}
.animated-metric:active {
    transform: translateY(-2px) scale(0.97);
    box-shadow: 0 6px 18px rgba(6, 182, 212, 0.5);
}
.animated-metric h2 {
    color: #FFFFFF !important;
    font-size: 2.2rem;
    margin: 0;
    font-weight: 700;
}
.animated-metric p {
    color: #A7F3D0 !important;
    font-size: 0.9rem;
    margin-top: 4px;
    margin-bottom: 0;
    text-transform: uppercase;
    letter-spacing: 1.2px;
}

/* 7. FROSTED GLASS CARDS WITH TACTILE HOVER */
.glass-card {
    background: rgba(255, 255, 255, 0.055);
    backdrop-filter: blur(18px);
    -webkit-backdrop-filter: blur(18px);
    border: 1px solid rgba(52, 211, 153, 0.24);
    border-radius: 20px;
    padding: 24px;
    box-shadow: 0 10px 34px 0 rgba(1, 16, 14, 0.48);
    margin-bottom: 22px;
    transition: transform 0.3s ease, border-color 0.3s ease, box-shadow 0.3s ease;
    animation: fadeInUp 0.7s ease-out;
}
.glass-card:hover {
    border-color: rgba(45, 212, 191, 0.55);
    box-shadow: 0 14px 38px rgba(16, 185, 129, 0.18);
}

/* 8. SPECIES SHOWCASE CARDS (HOVER + TOUCH PRESS) */
.species-card {
    background: rgba(255, 255, 255, 0.06);
    border: 1px solid rgba(52, 211, 153, 0.25);
    border-radius: 18px;
    padding: 22px 18px;
    text-align: center;
    backdrop-filter: blur(14px);
    transition: transform 0.3s cubic-bezier(0.175, 0.885, 0.32, 1.275), box-shadow 0.3s ease, border-color 0.3s ease;
    cursor: pointer;
    user-select: none;
}
.species-card:hover {
    transform: translateY(-8px) scale(1.02);
    box-shadow: 0 14px 32px rgba(45, 212, 191, 0.35);
    border-color: rgba(251, 191, 36, 0.75);
}
.species-card:active {
    transform: translateY(-2px) scale(0.97);
}
.badge {
    display: inline-block;
    padding: 5px 14px;
    border-radius: 14px;
    font-size: 0.8rem;
    font-weight: 600;
    margin-top: 10px;
}

/* 9. VIBRANT EMERALD-CYAN ACTION BUTTONS WITH LIGHT-SWEEP & TAP EFFECT */
div.stButton > button {
    position: relative;
    overflow: hidden;
    background: linear-gradient(90deg, #059669 0%, #10B981 50%, #06B6D4 100%) !important;
    color: #FFFFFF !important;
    font-weight: 600 !important;
    font-size: 1.05rem !important;
    border: 1px solid rgba(255, 255, 255, 0.22) !important;
    border-radius: 14px !important;
    padding: 12px 28px !important;
    box-shadow: 0 6px 22px rgba(16, 185, 129, 0.4) !important;
    transition: transform 0.22s ease, box-shadow 0.22s ease !important;
}
div.stButton > button::before {
    content: "";
    position: absolute;
    top: 0;
    left: -100%;
    width: 60%;
    height: 100%;
    background: linear-gradient(120deg, transparent, rgba(255, 255, 255, 0.35), transparent);
    transition: left 0.5s ease;
}
div.stButton > button:hover::before {
    left: 130%;
}
div.stButton > button:hover {
    transform: translateY(-3px) !important;
    box-shadow: 0 10px 30px rgba(6, 182, 212, 0.65) !important;
}
div.stButton > button:active {
    transform: scale(0.97) !important;
}

/* 10. STYLED TABS */
.stTabs [data-baseweb="tab-list"] {
    gap: 10px;
    background: rgba(255, 255, 255, 0.05);
    padding: 6px;
    border-radius: 14px;
    border: 1px solid rgba(52, 211, 153, 0.22);
}
.stTabs [data-baseweb="tab"] {
    border-radius: 10px;
    color: #A7F3D0 !important;
    font-weight: 600;
    padding: 8px 18px;
    transition: all 0.25s ease;
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(90deg, #059669, #06B6D4) !important;
    color: #FFFFFF !important;
}

/* TECH PILLS */
.tech-pill {
    display: inline-block;
    background: rgba(16, 185, 129, 0.2);
    border: 1px solid rgba(52, 211, 153, 0.45);
    color: #ECFDF5;
    padding: 6px 14px;
    border-radius: 20px;
    font-size: 0.85rem;
    font-weight: 600;
    margin: 4px;
    transition: transform 0.2s ease, background 0.2s ease;
}
.tech-pill:hover {
    transform: translateY(-2px);
    background: rgba(6, 182, 212, 0.32);
}
</style>

<div class="ambient-orb orb-1"></div>
<div class="ambient-orb orb-2"></div>
""", unsafe_allow_html=True)

# ==========================================
# 3. DATA LOADING & MODEL TRAINING
# ==========================================
COLUMN_ALIASES = {
    "sepal_length": ["sepal_length", "sepallength", "sepallengthcm", "sepal_len", "sl"],
    "sepal_width": ["sepal_width", "sepalwidth", "sepalwidthcm", "sepal_wid", "sw"],
    "petal_length": ["petal_length", "petallength", "petallengthcm", "petal_len", "pl"],
    "petal_width": ["petal_width", "petalwidth", "petalwidthcm", "petal_wid", "pw"],
    "species": ["species", "variety", "class", "target", "label", "iris_species", "flower_species"]
}

def normalize_iris_df(raw_df):
    if raw_df is None or raw_df.empty:
        return None
    df_copy = raw_df.copy()
    cleaned_cols = [
        str(c).strip().lower().replace(".", "_").replace("-", "_").replace(" ", "_").replace("(cm)", "").strip("_")
        for c in df_copy.columns
    ]
    df_copy.columns = cleaned_cols

    rename_map = {}
    for canonical, aliases in COLUMN_ALIASES.items():
        for col in df_copy.columns:
            if col in aliases or col.replace("_", "") in aliases:
                rename_map[col] = canonical
                break
    df_copy = df_copy.rename(columns=rename_map)

    required = ["sepal_length", "sepal_width", "petal_length", "petal_width", "species"]
    if not all(col in df_copy.columns for col in required):
        return None

    df_copy = df_copy[required].copy()
    for col in ["sepal_length", "sepal_width", "petal_length", "petal_width"]:
        df_copy[col] = pd.to_numeric(df_copy[col], errors="coerce")
    df_copy = df_copy.dropna()

    if df_copy.empty:
        return None

    df_copy["species"] = (
        df_copy["species"]
        .astype(str)
        .str.replace("Iris-", "", case=False, regex=False)
        .str.strip()
        .str.capitalize()
    )
    return df_copy

@st.cache_data
def get_dataset():
    candidate_files = [
        "iris_extended.xlsx",
        "IRIS FLOWER CLASSIFICATION.csv.xlsx",
        "IRIS FLOWER CLASSIFICATION.csv",
        "iris.csv",
        "IRIS.csv",
        "dataset.csv"
    ]
    for f in os.listdir("."):
        if f.endswith((".csv", ".xlsx")) and "iris" in f.lower() and f not in candidate_files:
            candidate_files.append(f)

    for fname in candidate_files:
        if os.path.exists(fname):
            try:
                raw_df = pd.read_excel(fname) if fname.endswith(".xlsx") else pd.read_csv(fname, sep=None, engine="python")
                parsed = normalize_iris_df(raw_df)
                if parsed is not None:
                    return parsed, ["sepal_length", "sepal_width", "petal_length", "petal_width"]
            except Exception:
                continue

    raw = load_iris()
    df_fallback = pd.DataFrame(raw.data, columns=["sepal_length", "sepal_width", "petal_length", "petal_width"])
    df_fallback["species"] = pd.Series(raw.target).map({0: "Setosa", 1: "Versicolor", 2: "Virginica"})
    return df_fallback, ["sepal_length", "sepal_width", "petal_length", "petal_width"]

@st.cache_resource
def train_model(df_data):
    X = df_data[["sepal_length", "sepal_width", "petal_length", "petal_width"]]
    y = df_data["species"]
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
species_counts = df["species"].value_counts().to_dict()

SPECIES_COLORS = {
    "Setosa": "#FB7185",      # Coral Rose
    "Versicolor": "#2DD4BF",  # Bioluminescent Teal
    "Virginica": "#FBBF24"    # Warm Amber Gold
}

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
        "desc": "🌸 **Iris Virginica** has the longest petals and sepals among the three species, showcasing rich, deep tones."
    }
}

def apply_emerald_theme(fig):
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#ECFDF5", family="Poppins"),
        title_font=dict(color="#ECFDF5", size=16),
        legend=dict(font=dict(color="#ECFDF5"))
    )
    fig.update_xaxes(gridcolor="rgba(52, 211, 153, 0.14)", zerolinecolor="rgba(52, 211, 153, 0.22)")
    fig.update_yaxes(gridcolor="rgba(52, 211, 153, 0.14)", zerolinecolor="rgba(52, 211, 153, 0.22)")
    return fig

# ==========================================
# 4. SIDEBAR NAVIGATION
# ==========================================
st.sidebar.markdown("## 🌿 **Iris AI Hub**")
st.sidebar.markdown("Intelligent flower classification suite.")
st.sidebar.markdown(
    f"<div style='background: rgba(16, 185, 129, 0.16); border: 1px solid rgba(52, 211, 153, 0.35); "
    f"border-radius: 12px; padding: 10px 14px; margin: 10px 0 18px 0; font-size: 0.85rem;'>"
    f"<b>Dataset:</b> {len(df):,} samples<br>"
    f"<b>Species:</b> {df['species'].nunique()} classes</div>",
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
        <span class="floating-flower">🌿</span>
        <h1 class="shimmer-title">Iris Flower AI Classifier</h1>
        <p style="font-size: 1.12rem; color: #A7F3D0; margin: 0 auto; max-width: 660px;">
            An intelligent machine learning platform for botanical recognition, morphology analysis, and real-time species classification.
        </p>
        <div style="margin-top: 16px;">
            <span style="background: rgba(16, 185, 129, 0.28); padding: 6px 16px; border-radius: 20px; border: 1px solid rgba(52, 211, 153, 0.45); font-size: 0.85rem; color: #ECFDF5; font-weight: 600;">
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
            <ul style="color: #D1FAE5; line-height: 1.85;">
                <li><b>🌿 Sepal Dimensions:</b> Outer protective calyx measurements.</li>
                <li><b>🌺 Petal Dimensions:</b> Core distinguishing floral proportions.</li>
            </ul>
            <hr style="border: 0; border-top: 1px solid rgba(52, 211, 153, 0.25); margin: 18px 0;">
            <p style="margin-bottom: 6px;"><b>🚀 Quick Navigation:</b></p>
            <p style="font-size: 0.95rem; color: #A7F3D0; margin: 0;">
                Navigate to <b>🔮 Prediction</b> to test custom inputs, or <b>📊 Dashboard</b> to explore 3D clusters, radar profiles, and feature correlations!
            </p>
        </div>
        """, unsafe_allow_html=True)

    st.write("")
    st.markdown("<h3 style='text-align: center; margin-bottom: 20px;'>🌿 Meet the Three Iris Species</h3>", unsafe_allow_html=True)
    s1, s2, s3 = st.columns(3)

    with s1:
        st.markdown(f"""
        <div class="species-card">
            <h4 style="color: #FB7185 !important; margin: 0 0 6px 0;">Iris Setosa</h4>
            <p style="font-size: 0.9rem; color: #D1FAE5; margin: 0;">Smallest petals, high sepal width. Cleanest separation boundary.</p>
            <span class="badge" style="background: rgba(251, 113, 133, 0.22); color: #FFE4E6; border: 1px solid #FB7185;">
                {species_counts.get('Setosa', 0):,} Samples
            </span>
        </div>
        """, unsafe_allow_html=True)

    with s2:
        st.markdown(f"""
        <div class="species-card">
            <h4 style="color: #2DD4BF !important; margin: 0 0 6px 0;">Iris Versicolor</h4>
            <p style="font-size: 0.9rem; color: #D1FAE5; margin: 0;">Medium petal and sepal sizes. Balanced floral proportions.</p>
            <span class="badge" style="background: rgba(45, 212, 191, 0.22); color: #CCFBF1; border: 1px solid #2DD4BF;">
                {species_counts.get('Versicolor', 0):,} Samples
            </span>
        </div>
        """, unsafe_allow_html=True)

    with s3:
        st.markdown(f"""
        <div class="species-card">
            <h4 style="color: #FBBF24 !important; margin: 0 0 6px 0;">Iris Virginica</h4>
            <p style="font-size: 0.9rem; color: #D1FAE5; margin: 0;">Longest petals and sepals. Bold, expansive floral structure.</p>
            <span class="badge" style="background: rgba(251, 191, 36, 0.22); color: #FEF3C7; border: 1px solid #FBBF24;">
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
        <p style="color: #A7F3D0; margin: 0;">Adjust the morphological sliders to classify your Iris flower in real time</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### 🌿 Sepal Dimensions")
        sepal_length = st.slider(
            "Sepal Length (cm)",
            min_value=float(round(X["sepal_length"].min() - 0.2, 1)),
            max_value=float(round(X["sepal_length"].max() + 0.2, 1)),
            value=float(round(X["sepal_length"].mean(), 1)),
            step=0.1
        )
        sepal_width = st.slider(
            "Sepal Width (cm)",
            min_value=float(round(X["sepal_width"].min() - 0.2, 1)),
            max_value=float(round(X["sepal_width"].max() + 0.2, 1)),
            value=float(round(X["sepal_width"].mean(), 1)),
            step=0.1
        )

    with col2:
        st.markdown("#### 🌺 Petal Dimensions")
        petal_length = st.slider(
            "Petal Length (cm)",
            min_value=float(round(X["petal_length"].min() - 0.2, 1)),
            max_value=float(round(X["petal_length"].max() + 0.2, 1)),
            value=float(round(X["petal_length"].mean(), 1)),
            step=0.1
        )
        petal_width = st.slider(
            "Petal Width (cm)",
            min_value=float(round(X["petal_width"].min() - 0.1, 1)),
            max_value=float(round(X["petal_width"].max() + 0.2, 1)),
            value=float(round(X["petal_width"].mean(), 1)),
            step=0.1
        )
    st.markdown('</div>', unsafe_allow_html=True)

    predict_btn = st.button("🌿 Classify Flower Now", use_container_width=True)

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
        <div style="background: linear-gradient(90deg, #059669, #0D9488, #06B6D4); padding: 20px; border-radius: 18px; text-align: center; margin: 20px 0; box-shadow: 0 8px 28px rgba(16, 185, 129, 0.5);">
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
                orientation="h",
                color="Species",
                color_discrete_map=SPECIES_COLORS,
                text="Confidence (%)",
                range_x=[0, 1]
            )
            fig_prob.update_layout(
                showlegend=False,
                height=300,
                xaxis_title="Probability",
                yaxis_title=""
            )
            apply_emerald_theme(fig_prob)
            st.plotly_chart(fig_prob, use_container_width=True)

# ==========================================
# PAGE 3: 📊 EYE-CATCHING DASHBOARD (NO ACCURACY)
# ==========================================
elif page == "📊 Dashboard":
    st.markdown("""
    <div class="animated-hero" style="padding: 24px; margin-bottom: 20px;">
        <h1 class="shimmer-title" style="font-size: 2.2rem;">📊 Botanical Analytics & Visual Dashboard</h1>
        <p style="color: #A7F3D0; margin: 0;">Interactive species radar, 3D cluster projections, confusion matrix, and morphology distributions</p>
    </div>
    """, unsafe_allow_html=True)

    k1, k2, k3 = st.columns(3)
    with k1:
        st.markdown(f"""
        <div class="animated-metric">
            <h2>{len(df):,}</h2>
            <p>Total Records</p>
        </div>
        """, unsafe_allow_html=True)
    with k2:
        st.markdown(f"""
        <div class="animated-metric">
            <h2>{len(X_train):,}</h2>
            <p>Training Split (80%)</p>
        </div>
        """, unsafe_allow_html=True)
    with k3:
        st.markdown(f"""
        <div class="animated-metric">
            <h2>{len(X_test):,}</h2>
            <p>Testing Split (20%)</p>
        </div>
        """, unsafe_allow_html=True)

    st.write("")

    ov1, ov2 = st.columns([1, 1.2])
    with ov1:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("#### 🍩 Species Composition")
        count_df = df["species"].value_counts().reset_index()
        count_df.columns = ["Species", "Count"]
        fig_donut = px.pie(
            count_df,
            names="Species",
            values="Count",
            hole=0.58,
            color="Species",
            color_discrete_map=SPECIES_COLORS
        )
        fig_donut.update_traces(textinfo="percent+label", textfont_size=13)
        fig_donut.update_layout(height=360, showlegend=False)
        apply_emerald_theme(fig_donut)
        st.plotly_chart(fig_donut, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with ov2:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("#### 🕸️ Average Morphology Radar")
        means_df = df.groupby("species")[feature_names].mean()
        radar_labels = [f.replace("_", " ").title() for f in feature_names]
        fig_radar = go.Figure()
        for sp in means_df.index:
            vals = means_df.loc[sp].tolist()
            fig_radar.add_trace(go.Scatterpolar(
                r=vals + [vals[0]],
                theta=radar_labels + [radar_labels[0]],
                fill="toself",
                name=sp,
                line=dict(color=SPECIES_COLORS.get(sp, "#2DD4BF"), width=2.5),
                opacity=0.72
            ))
        fig_radar.update_layout(
            polar=dict(
                bgcolor="rgba(0,0,0,0)",
                radialaxis=dict(visible=True, gridcolor="rgba(52, 211, 153, 0.22)", tickfont=dict(color="#A7F3D0")),
                angularaxis=dict(gridcolor="rgba(52, 211, 153, 0.22)", tickfont=dict(color="#ECFDF5", size=12))
            ),
            height=360,
            margin=dict(l=40, r=40, t=25, b=25)
        )
        apply_emerald_theme(fig_radar)
        st.plotly_chart(fig_radar, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["🎯 Confusion Matrix & Importance", "🧬 3D Cluster & Correlations", "📈 Feature Distributions"])

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
                color_continuous_scale="Teal",
                labels=dict(x="Predicted Species", y="Actual Species", color="Count")
            )
            fig_cm.update_layout(height=380)
            apply_emerald_theme(fig_cm)
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
                color_continuous_scale="Tealgrn",
                text_auto=".3f"
            )
            fig_imp.update_layout(height=380, showlegend=False)
            apply_emerald_theme(fig_imp)
            st.plotly_chart(fig_imp, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("#### Per-Class Precision, Recall & F1-Score")
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
                color_continuous_scale="YlGnBu"
            )
            fig_corr.update_layout(height=420)
            apply_emerald_theme(fig_corr)
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
                color_discrete_map=SPECIES_COLORS,
                opacity=0.85
            )
            fig_3d.update_layout(height=420, margin=dict(l=0, r=0, b=0, t=20))
            apply_emerald_theme(fig_3d)
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
            color_discrete_map=SPECIES_COLORS
        )
        fig_box.update_layout(height=430, showlegend=False)
        apply_emerald_theme(fig_box)
        st.plotly_chart(fig_box, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

# ==========================================
# PAGE 4: ℹ️ ABOUT
# ==========================================
elif page == "ℹ️ About":
    st.markdown("""
    <div class="animated-hero" style="padding: 24px; margin-bottom: 20px;">
        <h1 class="shimmer-title" style="font-size: 2.2rem;">ℹ️ About Iris Flower AI</h1>
        <p style="color: #A7F3D0; margin: 0;">Architecture, dataset statistics, and technology stack</p>
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
            <hr style="border: 0; border-top: 1px solid rgba(52, 211, 153, 0.25); margin: 16px 0;">
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
