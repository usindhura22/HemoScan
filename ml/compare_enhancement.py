
import cv2
import os

patient_ids = [293, 59, 2, 289]

output_dir = "data/enhancement_comparison"
os.makedirs(output_dir, exist_ok=True)

for patient_id in patient_ids:

    original_path = (
        f"data/inspection/patient_{patient_id}/"
        "nail_crop_2.jpg"
    )

    image = cv2.imread(original_path)

    if image is None:
        print(f"Image missing for patient {patient_id}")
        continue

    # 1. Save original
    cv2.imwrite(
        f"{output_dir}/{patient_id}_original.jpg",
        image
    )

    # 2. CLAHE enhancement
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

    l_channel, a_channel, b_channel = cv2.split(lab)

    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(4, 4)
    )

    enhanced_l = clahe.apply(l_channel)

    enhanced_lab = cv2.merge(
        [enhanced_l, a_channel, b_channel]
    )

    clahe_image = cv2.cvtColor(
        enhanced_lab,
        cv2.COLOR_LAB2BGR
    )

    cv2.imwrite(
        f"{output_dir}/{patient_id}_clahe.jpg",
        clahe_image
    )

    # 3. Mild sharpening
    blurred = cv2.GaussianBlur(
        image,
        (0, 0),
        sigmaX=1.0
    )

    sharpened = cv2.addWeighted(
        image,
        1.5,
        blurred,
        -0.5,
        0
    )

    cv2.imwrite(
        f"{output_dir}/{patient_id}_sharpened.jpg",
        sharpened
    )

    print(f"Completed patient {patient_id}")

print("\nENHANCEMENT COMPARISON COMPLETED")
print("Output folder:", output_dir)
