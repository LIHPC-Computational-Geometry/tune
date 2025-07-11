import pandas as pd

df = pd.read_csv("Trimesh_SB3_old_random-9_dataset.csv")

SEUIL_OUTLIER = -20

df['corrected_value'] = df['Value'].copy()

window_size = 5
half_window = window_size // 2

for i in range(len(df)):
    val = df.at[i, 'Value']
    if val < SEUIL_OUTLIER:
        start = max(i - half_window, 0)
        end = min(i + half_window + 1, len(df))
        window = df['Value'].iloc[start:end]

        window = window[window >= SEUIL_OUTLIER]
        if not window.empty:
            median = window.median()
            df.at[i, 'corrected_value'] = median

if 'Value' in df.columns:
    alpha =0.05
    df["Smoothed"]=df["Value"].ewm(alpha=alpha, adjust=False).mean()

    df.to_csv("Trimesh_SB3_old_random-9_dataset.csv",index=False)