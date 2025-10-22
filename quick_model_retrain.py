import numpy as np
import pandas as pd
import joblib
import dill
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load data
df = pd.read_csv("heart.csv")
print("Data loaded successfully")

# Remove duplicates
df = df.drop_duplicates()
print(f"Data shape after removing duplicates: {df.shape}")

# Prepare features and target
X = df.drop("target", axis=1)
y = df["target"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Categorical and numerical columns
categorical_columns = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal']
numerical_columns = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']

# One-Hot Encoding
ohe_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
X_train_encoded = ohe_encoder.fit_transform(X_train[categorical_columns])
X_test_encoded = ohe_encoder.transform(X_test[categorical_columns])

# Save encoder
with open("ohe_encoder.dill", "wb") as f:
    dill.dump(ohe_encoder, f)

# Standardization
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train[numerical_columns])
X_test_scaled = scaler.transform(X_test[numerical_columns])

# Save scaler
with open("standard_encoder.dill", "wb") as f:
    dill.dump(scaler, f)

# Combine features
encoded_columns = ohe_encoder.get_feature_names_out(categorical_columns)
X_train_final = np.hstack((X_train_scaled, X_train_encoded))
X_test_final = np.hstack((X_test_scaled, X_test_encoded))

# Train Random Forest model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_final, y_train)

# Test accuracy
y_pred = model.predict(X_test_final)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model accuracy: {accuracy:.4f}")

# Save model
joblib.dump(model, "overall_best_model.pkl")
print("Model saved successfully!")

# Copy to Django app
import shutil
shutil.copy("overall_best_model.pkl", "heart_disease_project/predictor/")
shutil.copy("ohe_encoder.dill", "heart_disease_project/predictor/")
shutil.copy("standard_encoder.dill", "heart_disease_project/predictor/")
print("Files copied to Django app!")
