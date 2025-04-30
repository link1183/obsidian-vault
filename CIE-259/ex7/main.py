import json
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

with open("2025-04-30-zollikofen 3m-2024.json", "r") as f:
    data = json.load(f)

records = data["values"]
df = pd.DataFrame(records)

df["datetime"] = pd.to_datetime(df["dateObserved"])
df["hour"] = df["datetime"].dt.hour
df["day"] = df["datetime"].dt.day
df["month"] = df["datetime"].dt.month

X = df[["hour", "day", "month", "relativeHumidity"]]
y = df["temperature"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print(f"R² score: {r2_score(y_test, y_pred):.3f}")
print(f"MSE: {mean_squared_error(y_test, y_pred):.3f}")

example = pd.DataFrame(
    [[20, 15, 9, 70]], columns=["hour", "day", "month", "relativeHumidity"]
)
pred = model.predict(example)[0]
print(f"Predicted example: {pred:.2f} °C")
