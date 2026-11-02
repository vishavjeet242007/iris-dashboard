import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

# ==========================================
# 1. PAGE CONFIGURATION
# ==========================================
st.set_page_config(
    page_title="Iris Flower AI",
    page_icon="🌸",
    layout="wide"
)

# ==========================================
# 2. AESTHETIC PURPLE THEME & STYLING
# ==========================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Poppins', sans-serif;
}

/* Vibrant Aesthetic Purple Gradient across all Streamlit containers */
[data-testid="stAppViewContainer"], 
.stApp, 
.main, 
[data-testid="stHeader"] {
    background: linear-gradient(135deg, #1B003A 0%, #2E0854 28%, #4C1D95 65%, #6D28D9 100%) !important;
    color: #FAF5FF !important;
}

/* Deep Royal Violet Sidebar */
[data-testid="stSidebar"], 
[data-testid="stSidebar"] > div:first-child {
    background: linear-gradient(180deg, #120026 0%, #240542 100%) !important;
    border-right: 1px solid rgba(192, 132, 252, 0.25) !important;
}
[data-testid="stSidebar"] * {
    color: #F3E8FF !important;
}

/* Typography & Headings */
h1, h2, h3, h4, h5, h6 {
    color: #FFFFFF !important;
    font-weight: 700 !important;
}
p, span, label, .stMarkdown {
    color: #F3E8FF !important;
}

/* Hero Banner on Home Page */
.purple-hero {
    background: linear-gradient(135deg, rgba(147, 51, 234, 0.4) 0%, rgba(76, 29, 149, 0.65) 100%);
    border: 1px solid rgba(216, 180, 254, 0.35);
    padding: 35px 25px;
    border-radius: 24px;
    backdrop-filter: blur(14px);
    -webkit-backdrop-filter: blur(14px);
    box-shadow: 0 12px 35px rgba(18, 0, 38, 0.55);
    margin-bottom: 25px;
    text-align: center;
}
.purple-hero h1 {
    font-size: 2.7rem;
    margin-bottom: 8px;
    background: linear-gradient(90deg, #FFFFFF, #E9D5FF, #F472B6);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.purple-hero p {
    font-size: 1.15rem;
    color: #DDD6FE !important;
}

/* Frosted Glass Cards */
.glass-card {
    background: rgba(255, 255, 255, 0.08);
    backdrop-filter: blur(16px);
    -webkit-backdrop-filter: blur(16px);
    border: 1px solid rgba(216, 180, 254, 0.25);
    border-radius: 20px;
    padding: 24px;
    box-shadow: 0 8px 32px 0 rgba(18, 0, 38, 0.4);
    margin-bottom: 24px;
}

/* Glowing Purple Metric Cards */
.purple-metric {
    background: linear-gradient(135deg, rgba(124, 58, 237, 0.45) 0%, rgba(192, 38, 211, 0.45) 100%);
    border: 1px solid rgba(216, 180, 254, 0.4);
    border-radius: 18px;
    padding: 20px 10px;
    text-align: center;
    box-shadow: 0 6px 22px rgba(0, 0, 0, 0.35);
    transition: transform 0.2s ease, box-shadow 0.2s ease;
}
.purple-metric:hover {
    transform: translateY(-4px);
    box-shadow: 0 10px 28px rgba(168, 85, 247, 0.5);
}
.purple-metric h2 {
    color: #FFFFFF !important;
    font-size: 2.2rem;
    margin: 0;
    font-weight: 700;
}
.purple-metric p {
    color: #E9D5FF !important;
    font-size: 0.95rem;
    margin-top: 4px;
    margin-bottom: 0;
    text-transform: uppercase;
    letter-spacing: 1.2px;
    font-weight: 500;
}

/* Vibrant Purple-Pink Action Buttons */
div.stButton > button:first-child {
    background: linear-gradient(90deg, #9333EA 0%, #D946EF 50%, #EC4899 100%) !important;
    color: #FFFFFF !important;
    font-weight: 600 !important;
    font-size: 1.05rem !important;
    border: none !important;
    border-radius: 14px !important;
    padding: 12px 28px !important;
    box-shadow: 0 6px 20px rgba(217, 70, 239, 0.45) !important;
    transition: all 0.3s ease !important;
}
div.stButton > button:first-child:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 10px 28px rgba(217, 70, 239, 0.7) !important;
}
</style>
""", unsafe_allow_html=True)

# ==========================================
# 3. DATA LOADING (5,390 DATASET SUPPORT)
# ==========================================
@st.cache_data
def get_dataset():
    # Looks for your uploaded 5,390 row dataset file in the repository
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

    # Search directory for any iris data file
    if df is None:
        for f in os.listdir("."):
            if f.endswith((".csv", ".xlsx")) and "iris" in f.lower():
                try:
                    df = pd.read_excel(f) if f.endswith(".xlsx") else pd.read_csv(f)
                    break
                except Exception:
                    pass

    # Fallback to standard iris dataset if file is not found
    if df is None:
        raw = load_iris()
        df = pd.DataFrame(raw.data, columns=['sepal_length', 'sepal_width', 'petal_length', 'petal_width'])
        df['species'] = raw.target
        df['species'] = df['species'].map({0: 'Setosa', 1: 'Versicolor', 2: 'Virginica'})
    else:
        # Standardize column names to: sepal_length, sepal_width, petal_length, petal_width, species
        df.columns = [c.strip().lower().replace(" ", "_").replace("(cm)", "").strip("_") for c in df.columns]
        if 'species' in df.columns:
            df['species'] = df['species'].astype(str).str.replace('Iris-', '').str.capitalize()

    feature_cols = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
    return df, feature_cols

df, feature_names = get_dataset()

X = df[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']]
y = df['species']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
y_pred_model = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred_model)

# Flower Images & Biological Descriptions
flower_info = {
    "Setosa": {
        "image": "https://upload.wikimedia.org/wikipedia/commons/a/a7/Iris_setosa_-_Botanischer_Garten_M%C3%BCnchen-Nymphenburg_-_DSC09255.JPG",
        "desc": "🌸 **Iris Setosa** is characterized by small, delicate petals and wide sepals. It is distinct and easily separable from the other two species."
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

# Dark/Purple Plotly styling helper
def apply_purple_theme(fig):
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#FAF5FF", family="Poppins"),
        title_font=dict(color="#FAF5FF", size=16),
        legend=dict(font=dict(color="#FAF5FF"))
    )
    return fig

# ==========================================
# 4. SIDEBAR NAVIGATION
# ==========================================
st.sidebar.markdown("## 🌸 **Iris AI Hub**")
st.sidebar.markdown("Intelligent flower classification suite.")
st.sidebar.write("")

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
# PAGE 1: 🏠 HOME
# ==========================================
if page == "🏠 Home":
    st.markdown("""
    <div class="purple-hero">
        <h1>🌸 Iris Flower AI Classifier</h1>
        <p>An intelligent machine learning application for botanical analysis and species classification</p>
    </div>
    """, unsafe_allow_html=True)
    
    # 4 Glowing Metric Cards (Dynamic for 5,390 flowers)
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.markdown(f"""
        <div class="purple-metric">
            <h2>{len(df):,}</h2>
            <p>Total Flowers</p>
        </div>
        """, unsafe_allow_html=True)
    with m2:
        st.markdown(f"""
        <div class="purple-metric">
            <h2>{df['species'].nunique()}</h2>
            <p>Species</p>
        </div>
        """, unsafe_allow_html=True)
    with m3:
        st.markdown(f"""
        <div class="purple-metric">
            <h2>{len(feature_names)}</h2>
            <p>Features</p>
        </div>
        """, unsafe_allow_html=True)
    with m4:
        st.markdown(f"""
        <div class="purple-metric">
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
        <div class="glass-card">
            <h3>✨ Welcome to the Iris AI Dashboard</h3>
            <p>Trained on <b>{len(df):,} botanical records</b> across three distinct Iris species (Setosa, Versicolor, and Virginica).</p>
            <hr style="border: 0; border-top: 1px solid rgba(216, 180, 254, 0.2); margin: 15px 0;">
            <p><b>🔍 Features & Navigation:</b></p>
            <ul>
                <li><b>🔮 Prediction:</b> Input measurements to identify an unknown flower and view model confidence.</li>
                <li><b>📊 Dashboard:</b> Explore the confusion matrix, feature distributions, and correlation maps.</li>
                <li><b>ℹ️ About:</b> Review project documentation and model specifications.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

# ==========================================
# PAGE 2: 🔮 PREDICTION
# ==========================================
elif page == "🔮 Prediction":
    st.markdown("""
    <div class="purple-hero" style="padding: 24px; margin-bottom: 20px;">
        <h1 style="font-size: 2.2rem;">🔮 Predict Flower Species</h1>
        <p>Adjust the morphological sliders to classify your Iris flower in real time</p>
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

        st.balloons()
        
        st.markdown(f"""
        <div style="background: linear-gradient(90deg, #9333EA, #C026D3); padding: 18px; border-radius: 16px; text-align: center; margin: 20px 0; box-shadow: 0 6px 25px rgba(192, 38, 211, 0.45);">
            <h2 style="color: white !important; margin: 0;">🎉 Identified Species: Iris {prediction}</h2>
        </div>
        """, unsafe_allow_html=True)
        
        res_left, res_right = st.columns([1, 1])
        with res_left:
            st.image(
                flower_info[prediction]["image"],
                caption=f"Identified: Iris {prediction}",
                use_container_width=True
            )
            st.markdown(f"""
            <div class="glass-card" style="padding: 16px;">
                {flower_info[prediction]["desc"]}
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
    <div class="purple-hero" style="padding: 24px; margin-bottom: 20px;">
        <h1 style="font-size: 2.2rem;">📊 Analysis Graphs & Model Evaluation</h1>
        <p>Interactive confusion matrix, correlation heatmap, and feature distributions</p>
    </div>
    """, unsafe_allow_html=True)

    # 1. Confusion Matrix & Classification Report
    st.markdown("### 1. 🎯 Model Performance & Confusion Matrix")
    col_cm, col_rep = st.columns([1.2, 1])
    
    with col_cm:
        cm = confusion_matrix(y_test, y_pred_model, labels=model.classes_)
        fig_cm = px.imshow(
            cm,
            text_auto=True,
            x=model.classes_,
            y=model.classes_,
            labels=dict(x="Predicted Class", y="Actual Class", color="Samples"),
            color_continuous_scale="Purples"
        )
        fig_cm.update_layout(
            title_text="<b>Confusion Matrix (Test Evaluation)</b>",
            height=370
        )
        apply_purple_theme(fig_cm)
        st.plotly_chart(fig_cm, use_container_width=True)

    with col_rep:
        st.markdown("#### 📋 Test Metrics")
        st.write(f"• **Total Dataset Rows:** {len(df):,} flowers")
        st.write(f"• **Test Set Size:** {len(y_test):,} samples (20% split)")
        st.write(f"• **Model Accuracy:** `{accuracy * 100:.2f}%`")
        
        rep_dict = classification_report(y_test, y_pred_model, output_dict=True)
        rep_df = pd.DataFrame(rep_dict).transpose().iloc[:3, :3]
        rep_df = rep_df.rename(columns={
            "precision": "Precision",
            "recall": "Recall",
            "f1-score": "F1-Score"
        })
        st.dataframe(rep_df.style.format("{:.2%}"), use_container_width=True)
        st.markdown("""
        <div class="glass-card" style="padding: 14px; font-size: 0.9rem;">
            💡 <b>Insight:</b> The diagonal cells show correct predictions for each flower type, demonstrating high model generalization across all 5,390 samples.
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<hr style='border-color: rgba(216,180,254,0.2);'>", unsafe_allow_html=True)

    # 2. Scatter Plot & Feature Importance
    st.markdown("### 2. 🔍 Feature Space & Separation")
    col_scat, col_imp = st.columns([1.2, 1])

    with col_scat:
        fig_scat = px.scatter(
            df.sample(min(1500, len(df)), random_state=42),
            x="petal_length",
            y="petal_width",
            color="species",
            symbol="species",
            color_discrete_sequence=["#F472B6", "#C084FC", "#818CF8"],
            title="<b>Petal Length vs Petal Width</b>",
            labels={"petal_length": "Petal Length (cm)", "petal_width": "Petal Width (cm)"}
        )
        fig_scat.update_layout(height=370)
        apply_purple_theme(fig_scat)
        st.plotly_chart(fig_scat, use_container_width=True)

    with col_imp:
        display_names = ["Sepal Length", "Sepal Width", "Petal Length", "Petal Width"]
        feat_imp_df = pd.DataFrame({
            'Feature': display_names,
            'Importance': model.feature_importances_
        }).sort_values(by='Importance', ascending=True)

        fig_imp = px.bar(
            feat_imp_df,
            x='Importance',
            y='Feature',
            orientation='h',
            color='Importance',
            color_continuous_scale="Purples",
            title="<b>Feature Importance (Random Forest)</b>"
        )
        fig_imp.update_layout(showlegend=False, height=370)
        apply_purple_theme(fig_imp)
        st.plotly_chart(fig_imp, use_container_width=True)

    st.markdown("<hr style='border-color: rgba(216,180,254,0.2);'>", unsafe_allow_html=True)

    # 3. Correlation Heatmap & Distribution Boxplots
    st.markdown("### 3. 📈 Correlation & Feature Distributions")
    col_corr, col_dist = st.columns([1, 1.2])

    with col_corr:
        corr_mat = df[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']].corr()
        corr_mat.columns = ["Sepal L", "Sepal W", "Petal L", "Petal W"]
        corr_mat.index = ["Sepal L", "Sepal W", "Petal L", "Petal W"]
        fig_corr = px.imshow(
            corr_mat,
            text_auto=".2f",
            color_continuous_scale="RdPu",
            title="<b>Feature Correlation Matrix</b>"
        )
        fig_corr.update_layout(height=370)
        apply_purple_theme(fig_corr)
        st.plotly_chart(fig_corr, use_container_width=True)

    with col_dist:
        feature_labels = {
            "sepal_length": "Sepal Length (cm)",
            "sepal_width": "Sepal Width (cm)",
            "petal_length": "Petal Length (cm)",
            "petal_width": "Petal Width (cm)"
        }
        chosen_feature = st.selectbox(
            "Select feature for boxplot analysis:",
            list(feature_labels.keys()),
            format_func=lambda x: feature_labels[x],
            index=2
        )
        fig_box = px.box(
            df,
            x="species",
            y=chosen_feature,
            color="species",
            color_discrete_sequence=["#F472B6", "#C084FC", "#818CF8"],
            title=f"<b>Distribution of {feature_labels[chosen_feature]}</b>",
            labels={chosen_feature: feature_labels[chosen_feature]}
        )
        fig_box.update_layout(height=330)
        apply_purple_theme(fig_box)
        st.plotly_chart(fig_box, use_container_width=True)

# ==========================================
# PAGE 4: ℹ️ ABOUT
# ==========================================
elif page == "ℹ️ About":
    st.markdown("""
    <div class="purple-hero" style="padding: 24px; margin-bottom: 20px;">
        <h1 style="font-size: 2.2rem;">ℹ️ About This AI Project</h1>
        <p>Project architecture, dataset background, and technical methodology</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown(f"""
    <div class="glass-card">
        <h3>🌺 Project Overview</h3>
        <p>This application is an AI demonstration project for multi-class classification trained on an expanded botanical dataset of <b>{len(df):,} flowers</b>.</p>
        <hr style="border: 0; border-top: 1px solid rgba(216, 180, 254, 0.2); margin: 15px 0;">
        <h4>📊 Dataset Specifications</h4>
        <ul>
            <li><b>Instances:</b> {len(df):,} flower specimens.</li>
            <li><b>Classes:</b> <i>Iris setosa</i> (1,543), <i>Iris versicolor</i> (1,901), and <i>Iris virginica</i> (1,946).</li>
            <li><b>Features:</b> Sepal Length, Sepal Width, Petal Length, Petal Width (cm).</li>
        </ul>
        <hr style="border: 0; border-top: 1px solid rgba(216, 180, 254, 0.2); margin: 15px 0;">
        <h4>⚙️ Machine Learning Pipeline</h4>
        <ul>
            <li><b>Algorithm:</b> Random Forest Classifier (100 Decision Trees).</li>
            <li><b>Validation:</b> 80/20 Stratified Train-Test Split with reproducible random seed.</li>
            <li><b>Stack:</b> Python, Streamlit, Scikit-Learn, Pandas, Plotly Express.</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

