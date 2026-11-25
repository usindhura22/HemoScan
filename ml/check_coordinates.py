
import cv2
import ast
import pandas as pd

patient_id = 59

image = cv2.imread(f"data/photo/{patient_id}.jpg")

df = pd.read_csv("data/metadata.csv")
row = df[df["PATIENT_ID"] == patient_id].iloc[0]

boxes = ast.literal_eval(row["NAIL_BOUNDING_BOXES"])

print("Original image dimensions:", image.shape)

# Interpretation A: [x1, y1, x2, y2]
preview_a = image.copy()

# Interpretation B: [y1, x1, y2, x2]
preview_b = image.copy()

for i, box in enumerate(boxes):
    a, b, c, d = map(int, box)

    cv2.rectangle(
        preview_a,
        (a, b),
        (c, d),
        (0, 255, 0),
        2
    )

    cv2.rectangle(
        preview_b,
        (b, a),
        (d, c),
        (0, 0, 255),
        2
    )

cv2.imwrite("data/coordinates_xy.jpg", preview_a)
cv2.imwrite("data/coordinates_yx.jpg", preview_b)

print("Coordinate comparison images saved.")
