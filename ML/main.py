import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt

data = pd.read_excel(r'.\ML\extracted_data 180 point.xlsx')
print("Data loaded successfully, data dimensions:", data.shape)

column_names = data.columns.tolist()
xlabel = column_names[0]
ylabels = column_names[1:4]

X = data.iloc[:, 0].values.reshape(-1, 1)
y = data.iloc[:, 1:4]
print("Features and target values prepared, feature dimensions:", X.shape, "target values dimensions:", y.shape)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=20/180, random_state=42)
print("Dataset split completed, training set feature dimensions:", X_train.shape, "testing set feature dimensions:", X_test.shape)

scaler_X = StandardScaler()
scaler_y = StandardScaler()

X_train_scaled = scaler_X.fit_transform(X_train)
X_test_scaled = scaler_X.transform(X_test)

y_train_scaled = scaler_y.fit_transform(y_train)
y_test_scaled = scaler_y.transform(y_test)
print("Features and target values standardized")

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train_scaled)
print("Model training completed")

y_pred_scaled = model.predict(X_test_scaled)
y_pred = scaler_y.inverse_transform(y_pred_scaled)
print("Prediction completed")

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f'Mean Squared Error (MSE): {mse:.4f}')
print(f'R² Score: {r2:.4f}')

plt.figure(figsize=(12, 6))
for i in range(3):
    plt.subplot(1, 3, i+1)
    plt.scatter(X_test, y_test.iloc[:, i], color='blue', label='Actual')
    plt.scatter(X_test, y_pred[:, i], color='red', label='Predicted')
    plt.xlabel(xlabel)
    plt.ylabel(ylabels[i])
    plt.legend()

plt.tight_layout()
plt.show()