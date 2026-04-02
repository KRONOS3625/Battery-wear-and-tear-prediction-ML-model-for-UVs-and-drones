# Battery Degradation Predictor

This project builds a battery wear-and-tear predictor for EV and drone style lithium batteries using real NASA aging datasets already present in the workspace. It trains machine learning models to estimate:

- `SoH` (state of health)
- `wear percent`
- `RUL` (remaining useful life in cycles)

The dashboard takes these live inputs:

- Internal resistance
- Capacity
- Cycle number
- Temperature

It then generates a prediction summary plus a degradation forecast chart and feature contribution chart.

## Data used

The training pipeline uses the public NASA Li-ion Battery Aging Dataset files inside:

- `C:\ML proj\5. Battery Data Set`

Reference links for the public dataset family:

- NASA batteries catalog: [data.nasa.gov](https://data.nasa.gov/dataset/?organization=nasa&tags=batteries&tags=phm)
- Randomized usage metadata: [catalog.data.gov](https://catalog.data.gov/dataset/randomized-battery-usage-4-40c-right-skewed-random-walk-a3e9a)

The local archives contain real `.mat` records with discharge capacity, cycle history, temperature traces, and impedance-derived resistance (`Re + Rct`).

## Project files

- `train_model.py`: extracts NASA battery records, creates a tabular dataset, trains the models, and saves plots
- `app.py`: lightweight local web server and prediction API
- `web/`: polished frontend UI
- `models/`: trained model bundle and metadata
- `data/processed/`: generated feature dataset
- `assets/`: training diagnostics and feature importance plots

## How to run in VS Code

1. Open `C:\ML proj` in VS Code.
2. Run the training pipeline:

```powershell
python train_model.py
```

3. Start the local app:

```powershell
python app.py
```

4. Open [http://127.0.0.1:8000](http://127.0.0.1:8000) in your browser.

## Notes

- The model is trained on real NASA aging data, not synthetic values.
- The current system Python has a broken `pandas` install, so this project intentionally avoids `pandas` and uses `numpy`, `scipy`, `scikit-learn`, and `matplotlib` directly.
- Your abstract PDF is aligned with the implementation: SoH, degradation, thermal behavior, internal resistance variation, and RUL are all reflected in the pipeline and UI.
