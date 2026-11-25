
import tensorflow as tf
from pathlib import Path

from dataset_loader import load_dataset
from model import build_model

# Load training and validation data
X_train, y_train = load_dataset("data/splits/train.csv")
X_val, y_val = load_dataset("data/splits/val.csv")

print("Training images:", X_train.shape)
print("Validation images:", X_val.shape)

# Build model
model = build_model()

# Create output folder
Path("ml/models_saved").mkdir(
    parents=True, exist_ok=True
)

callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=8,
        restore_best_weights=True
    ),

    tf.keras.callbacks.ModelCheckpoint(
        "ml/models_saved/best_model.keras",
        monitor="val_loss",
        save_best_only=True
    )
]

# Train model
history = model.fit(
    X_train,
    y_train,
    validation_data=(X_val, y_val),
    epochs=30,
    batch_size=8,
    callbacks=callbacks,
    verbose=1
)

# Save final model
model.save("ml/models_saved/final_model.keras")

print("\nTRAINING COMPLETED!")
print("Best model: ml/models_saved/best_model.keras")
print("Final model: ml/models_saved/final_model.keras")
