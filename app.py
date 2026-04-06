import streamlit as st
import pandas as pd
from model import train_model

# Load model
model, feature_cols = train_model()

# Load data
df = pd.read_csv("cleaned_data.csv")

# -------------------------------
# PAGE CONFIG
# -------------------------------
st.set_page_config(page_title="Product Dashboard", layout="wide")

st.title("📊 Product Analysis Dashboard")

# -------------------------------
# SIDEBAR
# -------------------------------
menu = st.sidebar.radio("Navigation", [
    "Dashboard",
    "Dataset",
    "EDA",
    "Prediction",
    "Insights"
])

# -------------------------------
# 🏠 DASHBOARD
# -------------------------------
if menu == "Dashboard":
    st.header("Overview")

    col1, col2, col3 = st.columns(3)

    col1.metric("Total Products", len(df))
    col2.metric("Average Rating", round(df["Rating"].mean(), 2))
    col3.metric("Average Price", round(df["Price"].mean(), 2))

    st.subheader("Rating Distribution")
    st.bar_chart(df["Rating"])

# -------------------------------
# 📂 DATASET
# -------------------------------
elif menu == "Dataset":
    st.header("Dataset Preview")
    st.dataframe(df)

# -------------------------------
# 📊 EDA
# -------------------------------
elif menu == "EDA":
    st.header("Exploratory Data Analysis")

    st.subheader("Correlation Heatmap")
    st.dataframe(df.corr())

    st.subheader("Price vs Rating")
    st.scatter_chart(df[["Price", "Rating"]])

    st.subheader("Discount vs Rating")
    st.scatter_chart(df[["Discount", "Rating"]])

# -------------------------------
# 🔮 PREDICTION
# -------------------------------
elif menu == "Prediction":
    st.header("Predict Product Rating")

    col1, col2 = st.columns(2)

    price = col1.number_input("Price")
    discount = col2.number_input("Discount")

    stock = st.selectbox("Stock", [0, 1])
    category = st.selectbox("Category", ["A", "B", "C", "D", "UNKNOWN"])

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

    if st.button("Predict"):
        input_df = pd.DataFrame([input_data])
        input_df = input_df.reindex(columns=feature_cols, fill_value=0)

        result = model.predict(input_df)

        st.success(f"⭐ Predicted Rating: {result[0]:.2f}")

# -------------------------------
# 🧠 INSIGHTS
# -------------------------------
elif menu == "Insights":
    st.header("Key Insights")

    corr = df.corr()["Rating"].sort_values(ascending=False)

    st.write(corr)

    st.info("👉 Discount and Price influence product ratings the most.")
