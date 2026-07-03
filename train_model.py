import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

df = pd.read_csv('Food_Delivery_Times.csv')

df.drop(columns=['Order_ID', 'Time_of_Day', 'Vehicle_Type'], inplace=True)

for col in ['Weather', 'Traffic_Level', 'Courier_Experience_yrs']:
    df[col] = df[col].fillna(df[col].mode()[0])

df = pd.get_dummies(df, columns=['Weather', 'Traffic_Level'], drop_first=True, dtype=int)

X = df.drop(columns=['Delivery_Time_min'])
y = df['Delivery_Time_min']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

lr = LinearRegression()
lr.fit(X_train_scaled, y_train)

preds = lr.predict(X_test_scaled)
mae   = mean_absolute_error(y_test, preds)
rmse  = np.sqrt(mean_squared_error(y_test, preds))
r2    = r2_score(y_test, preds)
print(f"MAE={mae:.4f}  RMSE={rmse:.4f}  R²={r2:.4f}")

joblib.dump(lr,                  'model.pkl')
joblib.dump(scaler,              'scaler.pkl')
joblib.dump(X.columns.tolist(),  'feature_names.pkl')
print("Saved: model.pkl  scaler.pkl  feature_names.pkl")
