
import ast
import pandas as pd

df = pd.read_csv("data/metadata.csv")

results = []

for _, row in df.iterrows():
    boxes = ast.literal_eval(row["NAIL_BOUNDING_BOXES"])

    # Second box = middle finger
    top, left, bottom, right = map(int, boxes[1])

    height = bottom - top
    width = right - left

    results.append({
        "PATIENT_ID": row["PATIENT_ID"],
        "Width": width,
        "Height": height,
        "Area": width * height,
        "Aspect_Ratio": round(width / height, 2)
    })

sizes = pd.DataFrame(results)

print("\nMIDDLE-FINGER CROP SIZE SUMMARY")
print("=" * 55)
print(sizes[["Width", "Height", "Area", "Aspect_Ratio"]].describe().round(2))

print("\n10 SMALLEST CROPS")
print(sizes.nsmallest(10, "Area").to_string(index=False))

print("\n10 LARGEST CROPS")
print(sizes.nlargest(10, "Area").to_string(index=False))

print("\n10 MOST EXTREME ASPECT RATIOS")
print(
    sizes.assign(
        Ratio_Difference=(sizes["Aspect_Ratio"] - 1).abs()
    ).nlargest(10, "Ratio_Difference").to_string(index=False)
)

sizes.to_csv("data/middle_finger_crop_audit.csv", index=False)

print("\nAudit saved to data/middle_finger_crop_audit.csv")
