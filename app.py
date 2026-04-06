import streamlit as st
import pandas as pd
from sklearn.tree import DecisionTreeRegressor

# -------------------------------
# 📂 Load Data
# -------------------------------
df = pd.read_csv("cleaned_data.csv")

# -------------------------------
# 🤖 Train Model
# -------------------------------
X = df.drop("Rating", axis=1)
y = df["Rating"]

model = DecisionTreeRegressor()
model.fit(X, y)

# -------------------------------
# 🎨 UI
# -------------------------------
st.set_page_config(page_title="Product Analysis App", layout="wide")

st.title("📊 Product Analysis & Prediction System")

# -------------------------------
# 📌 Sidebar Navigation
# -------------------------------
menu = st.sidebar.selectbox("Menu", [
    "Home",
    "Dataset",
    "EDA",
    "Prediction",
    "Insights"
])

# -------------------------------
# 🏠 HOME
# -------------------------------
if menu == "Home":
    st.header("Overview")
    st.write("This app analyzes product data and predicts ratings based on features.")

# -------------------------------
# 📂 DATASET
# -------------------------------
elif menu == "Dataset":
    st.header("Dataset Preview")
    st.write(df.head())
    st.write("Shape:", df.shape)

# -------------------------------
# 📊 EDA
# -------------------------------
elif menu == "EDA":
    st.header("Exploratory Data Analysis")

    st.subheader("Statistical Summary")
    st.write(df.describe())

    st.subheader("Correlation")
    st.write(df.corr())

    st.subheader("Rating Distribution")
    st.bar_chart(df["Rating"])

# -------------------------------
# 🔮 PREDICTION
# -------------------------------
elif menu == "Prediction":
    st.header("Predict Product Rating")

    price = st.number_input("Price")
    discount = st.number_input("Discount")
    stock = st.selectbox("Stock", [0, 1])

    # Category input
    category = st.selectbox("Category", ["A", "B", "C", "D", "UNKNOWN"])

    # Convert category to one-hot manually
    input_data = {
        "Price": price,
        "Discount": discount,
        "Stock": stock,
        "Category_A": 0,
        "Category_B": 0,
        "Category_C": 0,
        "Category_D": 0,
        "Category_UNKNOWN": 0
    }

    input_data[f"Category_{category}"] = 1

    input_df = pd.DataFrame([input_data])

    if st.button("Predict"):
        result = model.predict(input_df)
        st.success(f"Predicted Rating: {result[0]:.2f}")

# -------------------------------
# 🧠 INSIGHTS
# -------------------------------
elif menu == "Insights":
    st.header("Key Insights")

    corr = df.corr()["Rating"].sort_values(ascending=False)

    st.write("Feature Importance (Correlation):")
    st.write(corr)

    st.write("👉 Price and Discount significantly influence ratings.")