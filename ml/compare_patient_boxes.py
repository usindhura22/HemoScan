
import cv2
import ast
import pandas as pd
from pathlib import Path

df = pd.read_csv("data/metadata.csv")

patient_ids = [59, 176, 141, 262, 166]

for patient_id in patient_ids:
    row = df[df["PATIENT_ID"] == patient_id].iloc[0]

    image = cv2.imread(f"data/photo/{patient_id}.jpg")
    boxes = ast.literal_eval(row["NAIL_BOUNDING_BOXES"])

    for i, box in enumerate(boxes):
        top, left, bottom, right = map(int, box)

        crop = image[top:bottom, left:right]

        enlarged = cv2.resize(
            crop,
            (300, 300),
            interpolation=cv2.INTER_CUBIC
        )

        output = (
            f"data/patient_{patient_id}_box_{i+1}.jpg"
        )

        cv2.imwrite(output, enlarged)

        print("Saved:", output)

print("\nVisual comparison images generated.")
