
import cv2
import ast
import pandas as pd
import os

# Patients with the largest validation errors
patient_ids = [293, 59, 2, 289]

df = pd.read_csv("data/metadata.csv")

for patient_id in patient_ids:

    print("\n" + "=" * 60)
    print(f"INSPECTING PATIENT {patient_id}")
    print("=" * 60)

    image_path = f"data/photo/{patient_id}.jpg"
    image = cv2.imread(image_path)

    if image is None:
        print(f"Image not found: {image_path}")
        continue

    row_data = df[df["PATIENT_ID"] == patient_id]

    if row_data.empty:
        print(f"Metadata not found for patient {patient_id}")
        continue

    row = row_data.iloc[0]
    boxes = ast.literal_eval(row["NAIL_BOUNDING_BOXES"])

    height, width = image.shape[:2]

    print("Image dimensions:", width, height)
    print("Actual Hb:", row["HB_LEVEL_GperL"] / 10, "g/dL")
    print("Bounding boxes:", boxes)

    # Create a separate output folder for each patient
    output_dir = f"data/inspection/patient_{patient_id}"
    os.makedirs(output_dir, exist_ok=True)

    for i, box in enumerate(boxes):

        top, left, bottom, right = map(int, box)

        print(f"\nBox {i + 1}: {box}")

        # Draw bounding box on original image
        preview = image.copy()

        cv2.rectangle(
            preview,
            (left, top),
            (right, bottom),
            (0, 255, 0),
            3
        )

        cv2.putText(
            preview,
            f"Box {i + 1}",
            (left, max(25, top - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2
        )

        cv2.imwrite(
            f"{output_dir}/box_{i + 1}_preview.jpg",
            preview
        )

        # Extract exact crop
        nail_crop = image[top:bottom, left:right]

        if nail_crop.size == 0:
            print("Invalid crop!")
            continue

        cv2.imwrite(
            f"{output_dir}/nail_crop_{i + 1}.jpg",
            nail_crop
        )

        # Enlarge crop for visual inspection
        enlarged = cv2.resize(
            nail_crop,
            (400, 400),
            interpolation=cv2.INTER_CUBIC
        )

        cv2.imwrite(
            f"{output_dir}/nail_crop_{i + 1}_enlarged.jpg",
            enlarged
        )

        # Include surrounding context
        margin = 20

        y1 = max(0, top - margin)
        y2 = min(height, bottom + margin)
        x1 = max(0, left - margin)
        x2 = min(width, right + margin)

        context_crop = image[y1:y2, x1:x2]

        cv2.imwrite(
            f"{output_dir}/nail_context_{i + 1}.jpg",
            context_crop
        )

    print(f"\nImages saved in: {output_dir}")

print("\nPATIENT INSPECTION COMPLETED!")
