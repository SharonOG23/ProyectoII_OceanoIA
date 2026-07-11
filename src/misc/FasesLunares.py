import pandas as pd
from pathlib import Path

df = pd.read_csv("../../data/raw/RNN/moon_phases_UTC_2020-2050.csv")

phase_name = {
    0: "Luna nueva",
    1: "Luna creciente",
    2: "Cuarto creciente",
    3: "Luna gibosa creciente",
    4: "Luna llena",
    5: "Luna gibosa menguante",
    6: "Cuarto menguante",
    7: "Luna menguante"
}

phase_emoji = {
    0: "🌑",
    1: "🌒",
    2: "🌓",
    3: "🌔",
    4: "🌕",
    5: "🌖",
    6: "🌗",
    7: "🌘"
}

df["PhaseName"] = df["Category"].map(phase_name)
df["PhaseEmoji"] = df["Category"].map(phase_emoji)

df.drop(columns=["Area"], inplace=True)
df.drop(columns=["Phase"], inplace=True)

# df.to_csv(
#     "fasesLunares_enriched.csv",
#     index=False,
#     encoding="utf-8-sig"
# )


BASE_DIR = Path(__file__).parent.parent.parent

output_file = BASE_DIR / "data" / "processed" / "fasesLunares_enriched.csv"
output_file.parent.mkdir(parents=True, exist_ok=True)

df.to_csv(output_file, index=False, encoding="utf-8-sig")

print(df.head(10))
print("Fases lunares procesadas exitosamente.")