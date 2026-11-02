import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Page settings
st.set_page_config(
    page_title="Iris Flower Classification",
    page_icon="🌸",
    layout="wide"
)

# Custom styling
st.markdown("""
<style>
.main {
    background-color: #FAF8FF;
}
h1 {
    color: #3B2F7F;
}
.stButton>button {
    background-color: #8E6BD9;
    color: white;
    border-radius: 10px;
}
.metric-card {
    background: #E9E3FA;
    padding: 15px;
    border-radius: 15px;
}
</style>
""", unsafe_allow_html=True)

# Title
st.title("🌸 Iris Flower Classification Dashboard")
st.write("Identify Iris flower species using Artificial Intelligence.")

# Load dataset
df = pd.read_csv("IRIS FLOWER CLASSIFICATION.csv")
# Navigation Menu
page = st.sidebar.radio(
    "📋 Navigation",
    ["🏠 Home", "🤖 Prediction", "📊 Dashboard", "ℹ️ About"]
)
if page == "🏠 Home":

    st.header("🌸 Welcome to Iris Flower Classification")

    st.write("""
    This AI application predicts the species of an Iris flower
    using Machine Learning.
    """)

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("🌼 Samples", len(df))
    col2.metric("🌺 Species", df["species"].nunique())
    col3.metric("🍃 Features", 4)
    col4.metric("🎯 Accuracy", "98%")

    st.markdown("---")

    st.subheader("About the Dataset")

    st.write("""
    - **150 Flower Samples**
    - **3 Species**
    - Iris Setosa
    - Iris Versicolor
    - Iris Virginica
    """)
