
import cv2
import pandas as pd
import ast

patient_id = 59

image = cv2.imread(f"data/photo/{patient_id}.jpg")

df = pd.read_csv("data/metadata.csv")
row = df[df["PATIENT_ID"] == patient_id].iloc[0]

boxes = ast.literal_eval(row["NAIL_BOUNDING_BOXES"])

# Test clockwise and counterclockwise rotations
rotations = {
    "clockwise": cv2.ROTATE_90_CLOCKWISE,
    "counterclockwise": cv2.ROTATE_90_COUNTERCLOCKWISE
}

for name, rotation in rotations.items():

    rotated = cv2.rotate(image, rotation)

    preview = rotated.copy()

    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = map(int, box)

        cv2.rectangle(
            preview,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            3
        )

        cv2.putText(
            preview,
            f"Box {i + 1}",
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 255),
            2
        )

    cv2.imwrite(
        f"data/rotation_{name}.jpg",
        preview
    )

    print(name, "dimensions:", rotated.shape)

print("Rotation previews created.")
