
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
from sklearn.metrics import mean_absolute_error

from dataset_loader import load_dataset

# Load test data
X_test, y_test = load_dataset("data/splits/test.csv")

# Load trained model
model = tf.keras.models.load_model(
    "ml/models_saved/best_model.keras"
)

predictions = model.predict(X_test).flatten()

# Actual vs predicted graph
plt.figure(figsize=(8, 6))

plt.scatter(
    y_test,
    predictions,
    alpha=0.8
)

minimum = min(y_test.min(), predictions.min())
maximum = max(y_test.max(), predictions.max())

plt.plot(
    [minimum, maximum],
    [minimum, maximum],
    linestyle="--"
)

plt.xlabel("Actual Hemoglobin (g/dL)")
plt.ylabel("Predicted Hemoglobin (g/dL)")
plt.title("HemoScan: Actual vs Predicted Hb")

plt.savefig(
    "data/actual_vs_predicted.png",
    dpi=300
)

plt.show()

print("MAE:", round(mean_absolute_error(y_test, predictions), 4))
print("Graph saved: data/actual_vs_predicted.png")
