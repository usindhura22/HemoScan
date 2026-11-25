
import numpy as np
import tensorflow as tf

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

from dataset_loader import load_dataset

# Load untouched test dataset
X_test, y_test = load_dataset("data/splits/test.csv")

# Load best saved model
model = tf.keras.models.load_model(
    "ml/models_saved/best_model.keras"
)

# Generate predictions
predictions = model.predict(X_test).flatten()

# Calculate model metrics
mae = mean_absolute_error(y_test, predictions)
rmse = np.sqrt(mean_squared_error(y_test, predictions))
r2 = r2_score(y_test, predictions)

# Baseline: predict training-set average for every patient
import pandas as pd

train_df = pd.read_csv("data/splits/train.csv")
baseline_value = train_df["HB_g_dL"].mean()

baseline_predictions = np.full(
    len(y_test), baseline_value
)

baseline_mae = mean_absolute_error(
    y_test, baseline_predictions
)

baseline_rmse = np.sqrt(
    mean_squared_error(y_test, baseline_predictions)
)

print("\nHEMOSCAN MODEL EVALUATION")
print("-" * 40)

print("Test patients:", len(y_test))

print("\nEfficientNetB0:")
print("MAE:", round(mae, 4), "g/dL")
print("RMSE:", round(rmse, 4), "g/dL")
print("R2 Score:", round(r2, 4))

print("\nTraining-mean baseline:")
print("MAE:", round(baseline_mae, 4), "g/dL")
print("RMSE:", round(baseline_rmse, 4), "g/dL")

print("\nFirst 10 actual vs predicted values:")
for actual, predicted in zip(y_test[:10], predictions[:10]):
    print(
        f"Actual: {actual:.2f} | Predicted: {predicted:.2f}"
    )
