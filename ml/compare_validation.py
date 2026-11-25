
import tensorflow as tf

from dataset_loader import load_dataset

# Load validation dataset
X_val, y_val = load_dataset("data/splits/val.csv")

models = {
    "Original Baseline": "ml/models_saved/best_model.keras",
    "Fine-tuned": "ml/models_saved/finetuning/best_finetuned.keras"
}

print("\nVALIDATION MODEL COMPARISON")
print("=" * 50)

for name, path in models.items():

    model = tf.keras.models.load_model(path)

    loss, mae = model.evaluate(
        X_val,
        y_val,
        verbose=0
    )

    print(f"\n{name}")
    print("-" * 30)
    print(f"Validation MSE: {loss:.4f}")
    print(f"Validation RMSE: {loss ** 0.5:.4f}")
    print(f"Validation MAE: {mae:.4f} g/dL")

print("\nCOMPARISON COMPLETED")
