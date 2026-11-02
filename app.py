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
    - **5390 Flower Samples**
    - **3 Species**
    - Iris Setosa
    - Iris Versicolor
    - Iris Virginica
    """)
    st.header("🤖 Iris Flower Prediction")

sepal_length = st.slider("Sepal Length", 4.0, 8.0, 5.5)
sepal_width = st.slider("Sepal Width", 2.0, 4.5, 3.0)
petal_length = st.slider("Petal Length", 1.0, 7.0, 4.0)
petal_width = st.slider("Petal Width", 0.1, 2.5, 1.3)

if st.button("Predict"):
    prediction = model.predict([[sepal_length, sepal_width, petal_length, petal_width]])
    st.success(f"Predicted Species: {prediction[0]}")
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

st.header("📊 Model Performance")

st.metric("Accuracy", f"{accuracy_score(y_test, y_pred):.2f}")
st.metric("Precision", f"{precision_score(y_test, y_pred, average='weighted'):.2f}")
st.metric("Recall", f"{recall_score(y_test, y_pred, average='weighted'):.2f}")
st.metric("F1 Score", f"{f1_score(y_test, y_pred, average='weighted'):.2f}")
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

st.header("📌 Confusion Matrix")

cm = confusion_matrix(y_test, y_pred)

fig, ax = plt.subplots(figsize=(5,4))
sns.heatmap(cm, annot=True, cmap="Blues", fmt="d", ax=ax)
ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")
st.pyplot(fig)
from sklearn.metrics import classification_report

st.header("📄 Classification Report")

report = classification_report(y_test, y_pred)
st.text(report)
st.header("🌸 Species Distribution")

fig, ax = plt.subplots()
df['species'].value_counts().plot(kind='bar', color=['pink','purple','green'], ax=ax)
st.pyplot(fig)
st.header("🔥 Correlation Heatmap")

fig, ax = plt.subplots(figsize=(7,5))
sns.heatmap(df.iloc[:,0:4].corr(), annot=True, cmap="coolwarm", ax=ax)
st.pyplot(fig)
st.header("📈 Scatter Plot")

fig, ax = plt.subplots()
sns.scatterplot(
    data=df,
    x="petal_length",
    y="petal_width",
    hue="species",
    ax=ax
)
st.pyplot(fig)
st.header("📊 Histogram")

fig, ax = plt.subplots()
df["sepal_length"].hist(ax=ax)
st.pyplot(fig)
st.header("📦 Box Plot")

fig, ax = plt.subplots()
sns.boxplot(data=df.iloc[:,0:4], ax=ax)
st.pyplot(fig)
st.header("📋 Dataset")

st.dataframe(df)
