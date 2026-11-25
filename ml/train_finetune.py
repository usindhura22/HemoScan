
import tensorflow as tf
from pathlib import Path

from dataset_loader import load_dataset

# Load training and validation datasets
X_train, y_train = load_dataset("data/splits/train.csv")
X_val, y_val = load_dataset("data/splits/val.csv")

print("\nTraining:", X_train.shape)
print("Validation:", X_val.shape)

# Load the previously trained baseline model
model = tf.keras.models.load_model(
    "ml/models_saved/best_model.keras"
)

# Access EfficientNetB0
base_model = model.get_layer("efficientnetb0")

# Enable fine-tuning
base_model.trainable = True

# Freeze earlier layers; fine-tune only the last 20
for layer in base_model.layers[:-20]:
    layer.trainable = False

# Keep Batch Normalization layers frozen
for layer in base_model.layers:
    if isinstance(layer, tf.keras.layers.BatchNormalization):
        layer.trainable = False

print("\nTrainable EfficientNet layers:")
print(sum(layer.trainable for layer in base_model.layers))

# Recompile with a lower learning rate
model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.00001
    ),
    loss="mse",
    metrics=["mae"]
)

# Separate experiment directory
output_dir = Path("ml/models_saved/finetuning")
output_dir.mkdir(parents=True, exist_ok=True)

callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True
    ),

    tf.keras.callbacks.ModelCheckpoint(
        str(output_dir / "best_finetuned.keras"),
        monitor="val_loss",
        save_best_only=True
    ),

    tf.keras.callbacks.CSVLogger(
        str(output_dir / "finetuning_history.csv")
    )
]

# Fine-tune the model
history = model.fit(
    X_train,
    y_train,
    validation_data=(X_val, y_val),
    epochs=20,
    batch_size=8,
    callbacks=callbacks,
    verbose=1
)

# Save final fine-tuned model
model.save(output_dir / "final_finetuned.keras")

print("\nFINE-TUNING COMPLETED!")
print("Best model:", output_dir / "best_finetuned.keras")
print("Final model:", output_dir / "final_finetuned.keras")
