import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

st.title("🌸 Iris Flower Classification Dashboard")

# Load dataset
df = pd.read_csv("IRIS FLOWER CLASSIFICATION.csv")

st.subheader("Dataset")
st.dataframe(df)

st.subheader("Dataset Summary")
st.write(df.describe())

st.subheader("Species Distribution")
fig, ax = plt.subplots()
sns.countplot(data=df, x=df.columns[-1], ax=ax)
st.pyplot(fig)

st.subheader("Correlation Heatmap")
fig, ax = plt.subplots(figsize=(6,4))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm", ax=ax)
st.pyplot(fig)
