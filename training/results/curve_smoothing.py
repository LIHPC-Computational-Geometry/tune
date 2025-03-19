import pandas as pd

df = pd.read_csv("PPO_3.csv")

if 'Value' in df.columns:
    alpha =0.1
    df["Smoothed"]=df["Value"].ewm(alpha=alpha, adjust=False).mean()

    df.to_csv("smoothed.csv",index=False)