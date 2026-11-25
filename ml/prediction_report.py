
import pandas as pd
import numpy as np
import tensorflow as tf

from dataset_loader import load_dataset
from sklearn.metrics import mean_absolute_error

# Load test data
X_test, y_test = load_dataset("data/splits/test.csv")

# Load best model
model = tf.keras.models.load_model(
    "ml/models_saved/best_model.keras"
)

# Predict
predictions = model.predict(X_test).flatten()

# Create comparison table
results = pd.read_csv("data/splits/test.csv")

results = results[["PATIENT_ID", "HB_g_dL"]].copy()

results["Predicted_Hb"] = predictions
results["Absolute_Error"] = np.abs(
    results["HB_g_dL"] - predictions
)

results.to_csv(
    "data/prediction_results.csv",
    index=False
)

print("PREDICTION ERROR ANALYSIS")
print("-" * 40)

print("Test patients:", len(results))
print("MAE:", round(mean_absolute_error(
    y_test, predictions
), 4))

print("\nLargest prediction errors:")
print(
    results.nlargest(10, "Absolute_Error")[
        ["PATIENT_ID", "HB_g_dL",
         "Predicted_Hb", "Absolute_Error"]
    ].to_string(index=False)
)

print("\nReport saved to data/prediction_results.csv")
