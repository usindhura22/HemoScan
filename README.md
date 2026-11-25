# HemoScan – Experimental Hemoglobin Estimation Using Fingernail Images

HemoScan is an academic prototype built with Streamlit and TensorFlow. It accepts a **cropped fingernail image**, preprocesses it to 224 × 224 pixels, and uses a saved EfficientNetB0-based regression model to produce an experimental hemoglobin estimate.

> **Important medical notice:** This application is for education and research only. Its predictions are not clinically validated and must not be used to diagnose anemia, guide treatment, or replace a laboratory hemoglobin test. Consult a qualified healthcare professional for medical assessment.

## Project contents

```text
hemoscan/
├── app.py                         # Streamlit web application
├── requirements.txt               # Python dependencies
├── README.md                       # Project and setup guide
├── .gitignore
├── .python-version                 # Python runtime for deployment
├── ml/
│   ├── model.py                    # EfficientNetB0 model definition
│   ├── dataset_loader.py           # Loads image/label CSV files
│   ├── preprocess.py               # Crops and resizes nail images
│   ├── train.py                    # Model training script
│   ├── evaluate.py                  # Model evaluation utilities
│   └── models_saved/
│       ├── best_model.keras         # Model loaded by app.py
│       └── final_model.keras
└── data/
    ├── metadata.csv
    ├── photo/                      # Original dataset images
    ├── model_input/                 # Model input images
    ├── processed/                   # Processed nail images
    ├── splits/                      # train.csv, val.csv, test.csv
    └── prediction_results.csv       # Actual vs predicted values for dashboard
```

## Requirements

- Python 3.11 (recommended; model was saved with Keras 2.15)
- pip
- Internet connection for installing packages

## Run locally on Windows

1. Extract the ZIP file.
2. Open the extracted `hemoscan` folder in VS Code.
3. Open **Terminal → New Terminal**. Make sure the terminal is in the folder containing `app.py`.
4. Create and activate a virtual environment:

   ```powershell
   py -3.11 -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```

   If PowerShell blocks activation, run this once in that terminal and activate again:

   ```powershell
   Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
   .\.venv\Scripts\Activate.ps1
   ```

5. Install dependencies:

   ```powershell
   python -m pip install --upgrade pip
   pip install -r requirements.txt
   ```

6. Confirm that this model file exists: `ml/models_saved/best_model.keras`.
7. Start the app:

   ```powershell
   streamlit run app.py
   ```

8. Streamlit will show a local URL, usually `http://localhost:8501`. Open it in your browser.
9. Upload a **cropped fingernail image** in JPG, JPEG, or PNG format and select **Predict Hemoglobin Level**.

## Train the model again (optional)

The ZIP already contains the trained model used by the app, so training is not required just to run the website.

From the project root, after installing requirements:

```powershell
python ml/train.py
```

Training may take a long time and may require substantial RAM/compute. The training script uses the CSV files under `data/splits/` and image paths stored in those CSV files. If you move the project, keep its folder structure unchanged. The script may download ImageNet weights the first time it builds EfficientNetB0, so an internet connection is needed.

## Deploy free using Streamlit Community Cloud

1. Create/sign in to GitHub.
2. Create a new **public** repository, for example `hemoscan` (only make it public after checking dataset permission, privacy, and licensing).
3. Extract this ZIP and upload the project files and folders to the repository. Upload the **contents** of the extracted project, not the ZIP file itself. Keep `app.py`, `requirements.txt`, `ml/`, and `data/` at the repository root as shown above.
4. Check that `ml/models_saved/best_model.keras` is committed. The app cannot make predictions without it.
5. Open https://share.streamlit.io/ and sign in with GitHub.
6. Choose **Create app**, select your repository and branch (usually `main`), and set **Main file path** to `app.py`.
7. In Advanced settings, select Python 3.11 if the option is shown, then deploy.
8. Wait for dependency installation and app startup. Open the generated `*.streamlit.app` URL.

If deployment fails, open **Manage app → Logs** and read the first error. Common causes are a missing `requirements.txt`, an incorrect main file path, a missing model file, or an incompatible Python/package version.

## Important deployment notes

- `app.py` expects the model at `ml/models_saved/best_model.keras` and the results CSV at `data/prediction_results.csv`.
- The app expects an already-cropped nail image; it does not detect/crop nails from a full-hand photo.
- Prediction history is written to `data/prediction_history.csv`. Files written to a hosted app's local filesystem may not be permanent across restarts/redeployments. Use a database or cloud storage if persistent history is required.
- Do not upload API keys, passwords, personal information, or private patient data to GitHub. Review dataset terms before publishing the dataset or model.
- The model output is displayed as g/dL. It is an experimental estimate only, not a medical result.

## Troubleshooting

**`ModuleNotFoundError`**: Activate `.venv` and run `pip install -r requirements.txt` again.

**Model file not found**: Check that `ml/models_saved/best_model.keras` exists and that capitalization/spelling match exactly.

**TensorFlow install error**: Use Python 3.11 in a fresh virtual environment, then reinstall requirements.

**App starts but prediction fails**: Use a valid JPG/JPEG/PNG cropped fingernail image and check the deployment logs for the underlying error.
