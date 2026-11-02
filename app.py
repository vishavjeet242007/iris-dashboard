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
