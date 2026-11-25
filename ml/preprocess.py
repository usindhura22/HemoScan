
import ast
from pathlib import Path

import cv2
import pandas as pd

DATA_DIR = Path("data")
IMAGE_DIR = DATA_DIR / "photo"
OUTPUT_DIR = DATA_DIR / "processed"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(DATA_DIR / "metadata.csv")

processed = 0
skipped = []

for _, row in df.iterrows():
    patient_id = int(row["PATIENT_ID"])
    image_path = IMAGE_DIR / f"{patient_id}.jpg"

    image = cv2.imread(str(image_path))

    if image is None:
        skipped.append(patient_id)
        continue

    boxes = ast.literal_eval(row["NAIL_BOUNDING_BOXES"])

    # Use the second box: middle finger
    if len(boxes) < 2:
        skipped.append(patient_id)
        continue

    top, left, bottom, right = map(int, boxes[1])

    height, width = image.shape[:2]

    # Validate coordinates
    if not (
        0 <= top < bottom <= height
        and 0 <= left < right <= width
    ):
        print(f"Invalid coordinates for patient {patient_id}: {boxes[1]}")
        skipped.append(patient_id)
        continue

    # Crop using [top, left, bottom, right]
    nail = image[top:bottom, left:right]

    nail = cv2.resize(nail, (224, 224))

    output_path = OUTPUT_DIR / f"{patient_id}.jpg"
    cv2.imwrite(str(output_path), nail)

    processed += 1

print("\nCORRECTED PREPROCESSING COMPLETED")
print("Processed images:", processed)
print("Skipped images:", len(skipped))
print("Skipped patient IDs:", skipped)
print("Output folder:", OUTPUT_DIR)
