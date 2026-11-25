
import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import EfficientNetB0


def build_model():
    # Load pretrained EfficientNetB0
    base_model = EfficientNetB0(
        weights="imagenet",
        include_top=False,
        input_shape=(224, 224, 3)
    )

    # Freeze pretrained layers initially
    base_model.trainable = False

    inputs = layers.Input(shape=(224, 224, 3))

    # Extract image features
    x = base_model(inputs, training=False)

    # Regression head
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(0.3)(x)

    # Predict hemoglobin in g/dL
    outputs = layers.Dense(1, activation="linear")(x)

    model = Model(inputs, outputs)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=0.001
        ),
        loss="mse",
        metrics=["mae"]
    )

    return model


if __name__ == "__main__":
    model = build_model()
    model.summary()

    print("\nEfficientNetB0 model created successfully!")
