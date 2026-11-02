import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import *

st.set_page_config(
    page_title="Iris Flower AI",
    page_icon="🌸",
    layout="wide"
)

st.markdown("""
<style>

body{
background:#faf8ff;
}

.stApp{
background:linear-gradient(135deg,#E9E3FA,#FAF8FF,#F4A6C8);
}

h1,h2,h3{
color:#3B2F7F;
}

.block-container{
padding-top:2rem;
}

.card{
padding:20px;
border-radius:20px;
background:white;
box-shadow:0px 8px 25px rgba(0,0,0,.15);
}

</style>
""",unsafe_allow_html=True)

iris=load_iris()

df=pd.DataFrame(
iris.data,
columns=iris.feature_names
)

df["species"]=iris.target
df["species"]=df["species"].map({
0:"Setosa",
1:"Versicolor",
2:"Virginica"
})

X=df.iloc[:,:4]

y=df["species"]

X_train,X_test,y_train,y_test=train_test_split(
X,
y,
test_size=.2,
random_state=42
)

model=RandomForestClassifier()

model.fit(X_train,y_train)

y_pred=model.predict(X_test)

st.sidebar.title("🌸 Navigation")

page=st.sidebar.radio(
"",
[
"🏠 Home",
"🤖 Prediction",
"📊 Dashboard",
"ℹ About"
]
)

if page=="🏠 Home":

    st.title("🌸 Iris Flower Classification")

    st.subheader("Identify Iris Flowers using Artificial Intelligence")

    col1,col2,col3,col4=st.columns(4)

    col1.metric("Flowers",150)
    col2.metric("Species",3)
    col3.metric("Features",4)
    col4.metric("Accuracy",f"{accuracy_score(y_test,y_pred)*100:.2f}%")

    st.write("")

    st.success("Welcome to the Professional Iris AI Dashboard")

    st.image("https://upload.wikimedia.org/wikipedia/commons/4/41/Iris_versicolor_3.jpg",width=500)
        


