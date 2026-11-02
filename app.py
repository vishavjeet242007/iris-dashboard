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

    col1.metric("Flowers",5390)
    col2.metric("Species",3)
    col3.metric("Features",4)
    col4.metric("Accuracy",f"{accuracy_score(y_test,y_pred)*100:.2f}%")

    st.write("")
    
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

# -----------------------------
# PAGE SETTINGS
# -----------------------------
st.set_page_config(
    page_title="Iris Flower Classification",
    page_icon="🌸",
    layout="wide"
)

# -----------------------------
# CUSTOM CSS
# -----------------------------
st.markdown("""
<style>

.stApp{
background:linear-gradient(135deg,#E9E3FA,#FAF8FF,#F4A6C8);
}

h1,h2,h3{
color:#3B2F7F;
}

[data-testid="stMetric"]{
background:white;
padding:15px;
border-radius:15px;
box-shadow:0 5px 15px rgba(0,0,0,.15);
}

</style>
""",unsafe_allow_html=True)

# -----------------------------
# LOAD DATA
# -----------------------------
iris = load_iris()

df = pd.DataFrame(
    iris.data,
    columns=iris.feature_names
)

df["species"] = iris.target

df["species"] = df["species"].map({
    0:"Setosa",
    1:"Versicolor",
    2:"Virginica"
})

# -----------------------------
# TRAIN MODEL
# -----------------------------
X = df.iloc[:,0:4]
y = df["species"]

X_train,X_test,y_train,y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

model = RandomForestClassifier(random_state=42)

model.fit(X_train,y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test,y_pred)
precision = precision_score(y_test,y_pred,average="weighted")
recall = recall_score(y_test,y_pred,average="weighted")
f1 = f1_score(y_test,y_pred,average="weighted")

# -----------------------------
# SIDEBAR
# -----------------------------
st.sidebar.image(
    "https://upload.wikimedia.org/wikipedia/commons/4/41/Iris_versicolor_3.jpg",
    width=180
)

page = st.sidebar.radio(
    "Navigation",
    [
        "🏠 Home",
        "🤖 Prediction",
        "📊 Dashboard",
        "ℹ About"
    ]
)
# ===============================
# HOME PAGE
# ===============================

if page == "🏠 Home":

    st.title("🌸 Identify Iris Flowers with AI")

    st.markdown("""
    ### Welcome to the **Iris Flower Classification Dashboard**

    This project uses **Machine Learning (Random Forest)** to identify
    Iris flower species from flower measurements.

    ---
    """)

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("🌼 Total Samples", len(df))
    col2.metric("🌺 Species", 3)
    col3.metric("🍃 Features", 4)
    col4.metric("🎯 Accuracy", f"{accuracy*100:.2f}%")

    st.write("")

    st.markdown("## 🌷 Iris Species")

    c1, c2, c3 = st.columns(3)

    with c1:
        st.image(
            "https://upload.wikimedia.org/wikipedia/commons/5/56/Iris_setosa_2.jpg",
            use_container_width=True
        )
        st.success("### Setosa")
        st.write("""
        - Small petals
        - Purple flower
        - Easy to identify
        """)

    with c2:
        st.image(
            "https://upload.wikimedia.org/wikipedia/commons/4/41/Iris_versicolor_3.jpg",
            use_container_width=True
        )
        st.info("### Versicolor")
        st.write("""
        - Medium petals
        - Blue-purple flower
        - Moderate size
        """)

    with c3:
        st.image(
            "https://upload.wikimedia.org/wikipedia/commons/9/9f/Iris_virginica.jpg",
            use_container_width=True
        )
        st.warning("### Virginica")
        st.write("""
        - Large petals
        - Deep purple flower
        - Largest species
        """)

    st.divider()

    st.subheader("📋 Dataset Preview")

    st.dataframe(df.head(10), use_container_width=True)

    st.divider()

    st.subheader("📈 Species Distribution")

    fig = px.p
    
    st.success("Welcome to the Professional Iris AI Dashboard")

    st.image("https://upload.wikimedia.org/wikipedia/commons/4/41/Iris_versicolor_3.jpg",width=500)

# ===============================
# Statistics Cards
# ===============================

st.markdown("## 📊 Dashboard Statistics")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        label="🌸 Total Samples",
        value=len(df)
    )

with col2:
    st.metric(
        label="🌼 Total Species",
        value=df["species"].nunique()
    )

with col3:
    st.metric(
        label="🍃 Total Features",
        value=4
    )

with col4:
    st.metric(
        label="🎯 Model Accuracy",
        value=f"{accuracy*100:.2f}%"
    )


div[data-testid="stMetric"]{
    background: linear-gradient(135deg,#8E6BD9,#F4A6C8);
    color:white;
    border-radius:20px;
    padding:20px;
    text-align:center;
    box-shadow:0px 8px 20px rgba(0,0,0,0.25);
}

div[data-testid="stMetric"]:hover{
    transform:scale(1.05);
    transition:0.3s;
}

# ==========================================
# PREDICTION PAGE
# ==========================================

if page == "🤖 Prediction":

    st.title("🌸 Iris Flower Prediction")

    st.write("Enter the flower measurements (in cm) and click Predict.")

    col1, col2 = st.columns(2)

    with col1:
        sepal_length = st.slider("Sepal Length (cm)", 4.0, 8.0, 5.1, 0.1)
        sepal_width = st.slider("Sepal Width (cm)", 2.0, 4.5, 3.5, 0.1)

    with col2:
        petal_length = st.slider("Petal Length (cm)", 1.0, 7.0, 1.4, 0.1)
        petal_width = st.slider("Petal Width (cm)", 0.1, 2.5, 0.2, 0.1)

    if st.button("🌸 Predict Species"):

        input_data = [[
            sepal_length,
            sepal_width,
            petal_length,
            petal_width
        ]]

        prediction = model.predict(input_data)[0]

        probability = model.predict_proba(input_data)[0]

        st.success(f"### 🌼 Predicted Species : {prediction}")

        st.subheader("📏 Flower Measurements")

        st.write(f"**Sepal Length :** {sepal_length:.1f} cm")
        st.write(f"**Sepal Width :** {sepal_width:.1f} cm")
        st.write(f"**Petal Length :** {petal_length:.1f} cm")
        st.write(f"**Petal Width :** {petal_width:.1f} cm")

        st.subheader("🌸 Flower Image")

        if prediction=="Setosa":

            st.image(
                "https://upload.wikimedia.org/wikipedia/commons/5/56/Iris_setosa_2.jpg",
                width=350,
                caption="Iris Setosa"
            )

        elif prediction=="Versicolor":

            st.image(
                "https://upload.wikimedia.org/wikipedia/commons/4/41/Iris_versicolor_3.jpg",
                width=350,
                caption="Iris Versicolor"
            )

        else:

            st.image(
                "https://upload.wikimedia.org/wikipedia/commons/9/9f/Iris_virginica.jpg",
                width=350,
                caption="Iris Virginica"
            )

        st.subheader("📊 Prediction Confidence")

        confidence = pd.DataFrame({
            "Species": model.classes_,
            "Confidence (%)": probability*100
        })

        st.bar_chart(
            confidence.set_index("Species")
        )

        st.dataframe(confidence)

        st.balloons()
        


