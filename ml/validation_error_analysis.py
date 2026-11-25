
import tensorflow as tf
import pandas as pd
import numpy as np

from dataset_loader import load_dataset

# Load validation data
X_val, y_val = load_dataset("data/splits/val.csv")

df = pd.read_csv("data/splits/val.csv")

# Load baseline model
model = tf.keras.models.load_model(
    "ml/models_saved/best_model.keras"
)

# Generate predictions
predictions = model.predict(X_val, verbose=0).flatten()

results = pd.DataFrame({
    "PATIENT_ID": df["PATIENT_ID"].values,
    "Actual_Hb": y_val,
    "Predicted_Hb": predictions
})

results["Error"] = (
    results["Predicted_Hb"] - results["Actual_Hb"]
)

results["Absolute_Error"] = abs(results["Error"])

# Sort by largest prediction error
results = results.sort_values(
    by="Absolute_Error",
    ascending=False
)

print("\nTOP 15 VALIDATION PREDICTION ERRORS")
print("=" * 70)

print(
    results.head(15).round(3).to_string(index=False)
)

results.to_csv(
    "data/validation_error_analysis.csv",
    index=False
)

print("\nResults saved successfully.")
