"""Forecast weekly US influenza-like illness 1 to 4 weeks ahead. Rolling-origin evaluation, models from scratch.
Data: CDC FluView national weighted ILI through the Delphi Epidata API."""
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

df = pd.read_csv("data/fluview_national.csv").sort_values("epiweek").reset_index(drop=True)
df["year"], df["week"] = df.epiweek // 100, df.epiweek % 100
# continuity check: every year has weeks 1..52 (or 1..53) with nothing missing
for yr, g in df.groupby("year"):
    assert list(g.week) == list(range(1, len(g) + 1)) and len(g) in (52, 53), f"gap in {yr}"
y = np.log(df.wili.values); n = len(y)
# pandemic period: dropped from training targets and from evaluation
covid = ((df.epiweek >= 202010) & (df.epiweek <= 202226)).values
ang = 2 * np.pi * df.week.values / 52.18
H = [1, 2, 3, 4]; LAGS = 4

def feats(i):
    lag = [y[i - k] for k in range(LAGS)]
    return np.array([1.0, *lag, np.sin(ang[i]), np.cos(ang[i]), np.sin(2 * ang[i]), np.cos(2 * ang[i])])

def valid(i, h): return i >= LAGS + 52 and i + h < n and not covid[i - LAGS:i + h + 1].any()

def fit_ols(train, h, lam=1e-3):
    X = np.array([feats(i) for i in train]); t = np.array([y[i + h] - y[i] for i in train])
    return np.linalg.solve(X.T @ X + lam * np.eye(X.shape[1]), X.T @ t)

def knn_predict(train, i, h, k=8):
    # analog forecast: find past windows with similar recent trajectory and same time of year, average what happened next
    cur = np.array([y[i - j] for j in range(LAGS)])
    cands = []
    for j in train:
        dw = min(abs(df.week[j] - df.week[i]), 52 - abs(df.week[j] - df.week[i]))
        if dw > 6: continue
        past = np.array([y[j - m] for m in range(LAGS)])
        cands.append((np.sum((past - cur) ** 2) + 0.01 * dw, y[j + h] - y[j]))
    cands.sort(key=lambda x: x[0]); return y[i] + np.mean([c[1] for c in cands[:k]])

test_idx = [i for i in range(n) if df.year[i] >= 2015 and not covid[i]]
rows = []
for h in H:
    for i in test_idx:
        if i + h >= n or covid[i:i + h + 1].any(): continue
        train = [j for j in range(LAGS + 52, i - h) if valid(j, h)]   # only targets known before time i
        if len(train) < 150: continue
        if i % 4 == 0 or True:
            w = fit_ols(train, h) if (i % 2 == 0 or True) else None
        pred = {"persistence": y[i],
                "seasonal_average": np.mean([y[j + h] for j in train if df.week[j] == df.week[i]]) if any(df.week[j] == df.week[i] for j in train) else np.nan,
                "autoregressive_seasonal": y[i] + feats(i) @ w,
                "analog_knn": knn_predict(train, i, h)}
        for m, p in pred.items(): rows.append((h, df.epiweek[i], df.year[i], df.week[i], m, y[i + h], p))
res = pd.DataFrame(rows, columns=["h", "epiweek", "year", "week", "model", "actual_log", "pred_log"]).dropna()
res["abs_err"] = np.abs(np.exp(res.pred_log) - np.exp(res.actual_log))
res["season"] = np.where(res.week >= 40, res.year, res.year - 1)
res.to_csv("data/forecasts.csv", index=False)

mae = res.groupby(["h", "model"]).abs_err.mean().unstack().round(3); print("MAE in wILI percentage points\n", mae)
rel = (mae.div(mae["persistence"], axis=0)).round(2); print("relative to persistence\n", rel)
mae.to_csv("data/mae_by_horizon.csv"); rel.to_csv("data/relative_mae_by_horizon.csv")

# flu-season only (weeks 40-20): this is where forecasts matter
fs = res[(res.week >= 40) | (res.week <= 20)]
mae_fs = fs.groupby(["h", "model"]).abs_err.mean().unstack().round(3); print("MAE flu season only\n", mae_fs)
mae_fs.to_csv("data/mae_flu_season_only.csv")

# by season at 4 weeks ahead
by_season = res[res.h == 4].groupby(["season", "model"]).abs_err.mean().unstack().round(2); print(by_season)
by_season.to_csv("data/mae_by_season_4wk.csv")

# peak check: for each season, did the model's 4-week-ahead forecast catch the height of the peak?
pk = []
for s, g in res[res.h == 4].groupby("season"):
    a = g[g.model == "persistence"]
    if len(a) < 40: continue
    i = a.actual_log.values.argmax(); wk = a.epiweek.values[i]
    row = {"season": s, "forecast_origin_epiweek": int(wk), "peak_wili": round(float(np.exp(a.actual_log.values[i])), 2)}
    for m in ["persistence", "autoregressive_seasonal", "analog_knn"]:
        row[m + "_forecast"] = round(float(np.exp(g[(g.model == m)].pred_log.values[i])), 2)
    pk.append(row)
pd.DataFrame(pk).to_csv("data/peak_forecasts_4wk.csv", index=False); print(pd.DataFrame(pk))

# figures
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
fig, ax = plt.subplots(figsize=(7, 4)); cols = {"persistence": "grey", "seasonal_average": "#f4b183", "autoregressive_seasonal": "#4472c4", "analog_knn": "#c0504d"}
for m, c in cols.items(): ax.plot(rel.index, rel[m], "o-", color=c, label=m.replace("_", " "))
ax.axhline(1, color="k", ls="--", lw=0.8); ax.set_xticks(H); ax.set_xlabel("Weeks ahead"); ax.set_ylabel("MAE relative to persistence (1 = same)")
ax.set_title("Forecast error vs. 'next week = this week'"); ax.legend(frameon=False, fontsize=8); fig.tight_layout(); fig.savefig("figures/relative_error.png", dpi=150)

s = 2023; g = res[(res.h == 4) & (res.season == s)]
fig, ax = plt.subplots(figsize=(8, 4)); a = g[g.model == "persistence"].sort_values("epiweek")
ax.plot(range(len(a)), np.exp(a.actual_log), "k", lw=2, label="actual")
for m in ["autoregressive_seasonal", "analog_knn"]:
    b = g[g.model == m].sort_values("epiweek"); ax.plot(range(len(b)), np.exp(b.pred_log), color=cols[m], label=m.replace("_", " ") + ", 4 weeks ahead")
ax.set_xticks([]); ax.set_xlabel(f"Weeks of the {s}-{s+1} season"); ax.set_ylabel("Weighted ILI (%)"); ax.legend(frameon=False, fontsize=8); fig.tight_layout(); fig.savefig("figures/season_example.png", dpi=150)
