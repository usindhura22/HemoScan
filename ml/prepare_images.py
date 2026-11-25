
import cv2
import numpy as np
from pathlib import Path

INPUT_DIR = Path("data/processed")
OUTPUT_DIR = Path("data/model_input")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

processed = 0
skipped = 0

for image_path in INPUT_DIR.glob("*.jpg"):

    image = cv2.imread(str(image_path))

    if image is None:
        skipped += 1
        continue

    # OpenCV loads BGR; convert to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Ensure consistent dimensions
    image = cv2.resize(image, (224, 224))

    # Save as RGB-compatible image
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    cv2.imwrite(
        str(OUTPUT_DIR / image_path.name),
        image
    )

    processed += 1

print("IMAGE PREPARATION COMPLETED")
print("Processed:", processed)
print("Skipped:", skipped)
