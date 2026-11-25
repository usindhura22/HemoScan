
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

DATA_DIR = Path("data")
IMAGE_DIR = DATA_DIR / "model_input"

# Load dataset metadata
df = pd.read_csv(DATA_DIR / "metadata.csv")

# Convert hemoglobin from g/L to g/dL
df["HB_g_dL"] = df["HB_LEVEL_GperL"] / 10.0

# Match each patient to their processed image
df["image_path"] = df["PATIENT_ID"].apply(
    lambda pid: str(IMAGE_DIR / f"{int(pid)}.jpg")
)

# Check all image files exist
missing = [
    path for path in df["image_path"]
    if not Path(path).is_file()
]

if missing:
    raise FileNotFoundError(
        f"{len(missing)} images are missing: {missing[:5]}"
    )

if df["PATIENT_ID"].duplicated().any():
    raise ValueError("Duplicate patient IDs detected")

# Split patients, not individual nail crops
train_df, remaining_df = train_test_split(
    df,
    test_size=0.30,
    random_state=42,
    shuffle=True
)

val_df, test_df = train_test_split(
    remaining_df,
    test_size=0.50,
    random_state=42,
    shuffle=True
)

# Save split metadata
SPLIT_DIR = DATA_DIR / "splits"
SPLIT_DIR.mkdir(parents=True, exist_ok=True)

train_df.to_csv(SPLIT_DIR / "train.csv", index=False)
val_df.to_csv(SPLIT_DIR / "val.csv", index=False)
test_df.to_csv(SPLIT_DIR / "test.csv", index=False)

print("DATASET SPLITTING COMPLETED")
print("-" * 40)
print("Training patients:", len(train_df))
print("Validation patients:", len(val_df))
print("Testing patients:", len(test_df))
print("Total patients:",
      len(train_df) + len(val_df) + len(test_df))

print("\nAverage Hb (g/dL):")
print("Train:", round(train_df["HB_g_dL"].mean(), 2))
print("Validation:", round(val_df["HB_g_dL"].mean(), 2))
print("Test:", round(test_df["HB_g_dL"].mean(), 2))

print("\nSaved to:", SPLIT_DIR)
