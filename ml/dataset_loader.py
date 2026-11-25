
import cv2
import numpy as np
import pandas as pd
from pathlib import Path

IMAGE_SIZE = 224

def load_dataset(csv_path):
    df = pd.read_csv(csv_path)

    images = []
    labels = []

    for _, row in df.iterrows():
        image_path = Path(row["image_path"])

        image = cv2.imread(str(image_path))

        if image is None:
            print("Could not load:", image_path)
            continue

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        image = cv2.resize(
            image,
            (IMAGE_SIZE, IMAGE_SIZE)
        )

        hb_value = float(row["HB_g_dL"])

        if not np.isfinite(hb_value):
            print("Invalid Hb value:", image_path)
            continue

        images.append(image)
        labels.append(hb_value)

    X = np.asarray(images, dtype=np.float32)
    y = np.asarray(labels, dtype=np.float32)

    if len(X) == 0:
        raise ValueError("No valid images were loaded.")

    print("\nDataset validation")
    print("------------------------------")
    print("Images:", len(X))
    print("Labels:", len(y))
    print("Image shape:", X.shape)
    print("Label shape:", y.shape)
    print("Pixel minimum:", X.min())
    print("Pixel maximum:", X.max())
    print("Pixel mean:", round(float(X.mean()), 2))
    print("Hb minimum:", y.min())
    print("Hb maximum:", y.max())
    print("Hb mean:", round(float(y.mean()), 2))

    return X, y


if __name__ == "__main__":
    X_train, y_train = load_dataset(
        "data/splits/train.csv"
    )

    print("\nFirst five Hb values:", y_train[:5])


