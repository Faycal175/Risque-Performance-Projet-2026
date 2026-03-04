from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# ===============================
# 1) Mapping portefeuille ancien
# ===============================

mapping_fonds = {
    "LU0072462186": {"Nom": "BGF European Value A2"},
    "LU1893597309": {"Nom": "BSF European Unconstrained"},
    "LU0154236417": {"Nom": "BGF US Flexible Equity"},
    "LU1883854199": {"Nom": "Amundi US Eq Fundm Gr"},
    "FR0010983924": {"Nom": "EdR Japan C"},
    "LU1244893696": {"Nom": "EdRF Big Data A EUR"},
    "LU1919842267": {"Nom": "Oddo Artificial Intellig"},
    "LU1279613365": {"Nom": "BGF Asian Dragon A2"},
    "FR0010868901": {"Nom": "Ellipsis European Conv"},
    "LU1161527038": {"Nom": "EdRF Bond Allocation Acc"},
    "LU1191877379": {"Nom": "BGF European High Yield"},
    "FR0011288513": {"Nom": "Sycomore Sélection Crédit"},
    "LU1882449801": {"Nom": "Amundi Em Mkts Bd"},
    "LU1897613763": {"Nom": "EdRF Emerging Sovereign"},
}

isins = list(mapping_fonds.keys())


# ===============================
# 2) Chargement Data.xlsb
# ===============================

project_root = Path(__file__).resolve().parent
file_path = project_root / "Projet" / "Data.xlsb"

df = pd.read_excel(file_path, sheet_name="Prices", engine="pyxlsb")
df.columns = [c.lower().strip() for c in df.columns]

# Adaptation colonnes
df = df.rename(columns={
    "date": "date",
    "isin": "ticker",
    "ticker": "ticker",
    "close": "close",
    "price": "close"
})

# Nettoyage
df["date"] = pd.to_datetime(df["date"], errors="coerce")
df["ticker"] = df["ticker"].astype(str).str.strip()
df["close"] = (
    df["close"]
    .astype(str)
    .str.replace(",", ".", regex=False)
)
df["close"] = pd.to_numeric(df["close"], errors="coerce")

df = df.dropna(subset=["date", "ticker", "close"])

# ===============================
# 3) Pivot wide
# ===============================

prices = (
    df[df["ticker"].isin(isins)]
    .pivot(index="date", columns="ticker", values="close")
    .sort_index()
    .ffill()
)

prices = prices.dropna()


# ===============================
# 4) Calcul performance
# ===============================

perf = prices.iloc[-1] / prices.iloc[0] - 1
perf = perf.sort_values(ascending=False)

top8 = perf.head(8).index

print("\nTop 8 performeurs :")
for isin in top8:
    print(mapping_fonds[isin]["Nom"], ":", round(perf[isin]*100, 2), "%")


# ===============================
# 5) Normalisation base 100
# ===============================

prices_norm = prices / prices.iloc[0] * 100


# ===============================
# 6) Plot
# ===============================

plt.figure(figsize=(14, 7))

# Fonds non top8 en gris
for col in prices_norm.columns:
    if col not in top8:
        plt.plot(
            prices_norm.index,
            prices_norm[col],
            color="lightgray",
            linewidth=1,
            alpha=0.5
        )

# Top 8 en surbrillance avec NOM
for col in top8:
    plt.plot(
        prices_norm.index,
        prices_norm[col],
        linewidth=2.8,
        label=mapping_fonds[col]["Nom"]
    )

plt.title("Top 8 Performeurs - Portefeuille Ancien (Base 100)")
plt.xlabel("Date")
plt.ylabel("Base 100")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


# ===============================
# 7) Bar chart Top 8
# ===============================

plt.figure(figsize=(10, 5))
plt.bar(
    [mapping_fonds[i]["Nom"] for i in top8],
    perf[top8] * 100
)

plt.xticks(rotation=45, ha="right")
plt.ylabel("Performance (%)")
plt.title("Top 8 - Performance Totale")
plt.tight_layout()
plt.show()
