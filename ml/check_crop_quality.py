
import cv2
import ast
import pandas as pd

df = pd.read_csv("data/metadata.csv")

results = []

for _, row in df.iterrows():

    patient_id = row["PATIENT_ID"]

    image_path = f"data/photo/{patient_id}.jpg"
    image = cv2.imread(image_path)

    if image is None:
        print(f"Missing image: {patient_id}")
        continue

    boxes = ast.literal_eval(row["NAIL_BOUNDING_BOXES"])

    # Second bounding box = middle finger
    top, left, bottom, right = map(int, boxes[1])

    crop = image[top:bottom, left:right]

    if crop.size == 0:
        continue

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

    # Calculate sharpness
    sharpness = cv2.Laplacian(
        gray,
        cv2.CV_64F
    ).var()

    results.append({
        "PATIENT_ID": patient_id,
        "Width": right - left,
        "Height": bottom - top,
        "Sharpness": sharpness
    })

results_df = pd.DataFrame(results)

print("\nCROP SHARPNESS SUMMARY")
print("=" * 55)

print(
    results_df["Sharpness"].describe().round(2)
)

print("\n20 LOWEST SHARPNESS SCORES")

print(
    results_df.nsmallest(20, "Sharpness").round(2).to_string(index=False)
)

print("\n20 HIGHEST SHARPNESS SCORES")

print(
    results_df.nlargest(20, "Sharpness").round(2).to_string(index=False)
)

results_df.to_csv(
    "data/crop_quality_results.csv",
    index=False
)

print("\nResults saved to data/crop_quality_results.csv")
