
import ast
from pathlib import Path

import pandas as pd
import cv2

DATA_DIR = Path("data")
IMAGE_DIR = DATA_DIR / "photo"

df = pd.read_csv(DATA_DIR / "metadata.csv")

for box_index in range(3):
    valid = 0
    invalid = 0
    widths = []
    heights = []

    for _, row in df.iterrows():
        patient_id = int(row["PATIENT_ID"])
        image_path = IMAGE_DIR / f"{patient_id}.jpg"

        image = cv2.imread(str(image_path))

        if image is None:
            invalid += 1
            continue

        boxes = ast.literal_eval(row["NAIL_BOUNDING_BOXES"])

        if len(boxes) <= box_index:
            invalid += 1
            continue

        top, left, bottom, right = map(
            int, boxes[box_index]
        )

        img_height, img_width = image.shape[:2]

        if not (
            0 <= top < bottom <= img_height
            and 0 <= left < right <= img_width
        ):
            invalid += 1
            continue

        widths.append(right - left)
        heights.append(bottom - top)
        valid += 1

    print(f"\nBOX {box_index + 1}")
    print("-" * 30)
    print("Valid annotations:", valid)
    print("Invalid annotations:", invalid)

    if widths:
        print("Average width:", round(sum(widths) / len(widths), 2))
        print("Average height:", round(sum(heights) / len(heights), 2))
        print("Minimum width:", min(widths))
        print("Maximum width:", max(widths))
        print("Minimum height:", min(heights))
        print("Maximum height:", max(heights))
