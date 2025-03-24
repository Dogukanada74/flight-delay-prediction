import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import xgboost as xgb


flight_data = pd.read_csv('C:/Users/dogukan/PycharmProjects/FlightDelayPrediction/Jan_2019_ontime.csv')


flight_data = flight_data.dropna(subset=["ARR_DEL15"]).reset_index(drop=True)


flight_data = flight_data.dropna(axis=1, how='all')

le = LabelEncoder()
categorical_columns = flight_data.select_dtypes(include=['object']).columns
for col in categorical_columns:
    flight_data[col] = le.fit_transform(flight_data[col])


y = flight_data["ARR_DEL15"]
X = flight_data.drop(columns=["DEP_DEL15", "ARR_DEL15"], errors='ignore')


imputer = SimpleImputer(strategy='median')  # 'most_frequent' yerine 'median'
X_imputed = imputer.fit_transform(X)
X_imputed = pd.DataFrame(X_imputed, columns=X.columns)


X_train, X_valid, y_train, y_test = train_test_split(X_imputed, y, test_size=0.2, random_state=42)


scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_valid_scaled = scaler.transform(X_valid)


model = xgb.XGBClassifier(n_estimators=200, max_depth=8, learning_rate=0.05, random_state=42)
model.fit(X_train_scaled, y_train)


importances = model.feature_importances_
important_features = X.columns[np.argsort(importances)[-10:]]

print("Most Important Features:", important_features)


X_train_best = X_train_scaled[:, np.argsort(importances)[-10:]]
X_valid_best = X_valid_scaled[:, np.argsort(importances)[-10:]]

model.fit(X_train_best, y_train)


y_pred = model.predict(X_valid_best)


print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))