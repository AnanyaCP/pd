import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error

# -------------------------------
# 📂 Load cleaned data
# -------------------------------
df = pd.read_csv("cleaned_data.csv")

# -------------------------------
# 🎯 Select Target
# -------------------------------
y = df["Rating"]

# -------------------------------
# 🧠 Select Features
# -------------------------------
X = df.drop("Rating", axis=1)

# -------------------------------
# ✂️ Split data
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -------------------------------
# 🤖 Train model
# -------------------------------
model = DecisionTreeRegressor()
model.fit(X_train, y_train)

# -------------------------------
# 🔮 Predict
# -------------------------------
pred = model.predict(X_test)

# -------------------------------
# 📊 Evaluate
# -------------------------------
error = mean_squared_error(y_test, pred)
print("Mean Squared Error:", error)

# -------------------------------
# 🧠 Feature importance
# -------------------------------
importance = pd.Series(model.feature_importances_, index=X.columns)
print("\nFeature Importance:\n", importance.sort_values(ascending=False))