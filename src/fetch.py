"""Download weekly national influenza-like illness (ILI) data from the CMU Delphi Epidata API (CDC FluView)."""
import requests, pandas as pd
r = requests.get("https://api.delphi.cmu.edu/epidata/fluview/", params={"regions": "nat", "epiweeks": "201001-202552"}, timeout=120)
r.raise_for_status()
df = pd.DataFrame(r.json()["epidata"])[["epiweek", "wili", "ili", "num_ili", "num_patients", "num_providers"]]
df.to_csv("data/fluview_national.csv", index=False)
print(len(df), df.epiweek.min(), df.epiweek.max())
