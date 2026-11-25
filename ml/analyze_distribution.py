
import pandas as pd

files = {
    "Training": "data/splits/train.csv",
    "Validation": "data/splits/val.csv",
    "Testing": "data/splits/test.csv"
}

print("HEMOGLOBIN DISTRIBUTION ANALYSIS")
print("-" * 50)

for name, path in files.items():

    df = pd.read_csv(path)
    hb = df["HB_g_dL"]

    print(f"\n{name} Dataset")
    print("-" * 30)

    print("Total patients:", len(df))
    print("Minimum Hb:", round(hb.min(), 2))
    print("Maximum Hb:", round(hb.max(), 2))
    print("Average Hb:", round(hb.mean(), 2))

    print("Below 8 g/dL:", (hb < 8).sum())
    print("8 to below 11:", ((hb >= 8) & (hb < 11)).sum())
    print("11 to below 12.5:", ((hb >= 11) & (hb < 12.5)).sum())
    print("12.5 and above:", (hb >= 12.5).sum())
