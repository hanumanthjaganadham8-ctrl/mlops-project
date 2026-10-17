"""
Generate a synthetic crop dataset with the SAME schema as the public Kaggle
"Crop Recommendation Dataset" (Atharva Ingle):

    N, P, K, temperature, humidity, ph, rainfall, label

To use real data, just drop the Kaggle CSV at data/crop_data.csv and skip this file.
"""
from pathlib import Path

import numpy as np
import pandas as pd

RNG = np.random.default_rng(42)

# (mean, std) per feature for each crop - rough agronomic ranges
CROP_PROFILES = {
    #            N          P          K          temp       humidity   ph         rainfall
    "rice":      [(80, 12), (48, 10), (40, 6),   (24, 2),   (82, 5),   (6.4, 0.4), (230, 35)],
    "maize":     [(78, 12), (48, 10), (20, 5),   (22, 3),   (65, 6),   (6.2, 0.5), (85, 20)],
    "chickpea":  [(40, 8),  (68, 8),  (80, 6),   (19, 3),   (17, 4),   (7.3, 0.5), (80, 15)],
    "kidneybeans":[(20, 6), (68, 8),  (20, 5),   (20, 3),   (22, 4),   (5.7, 0.4), (105, 20)],
    "banana":    [(100, 10),(82, 8),  (50, 6),   (27, 1.5), (80, 4),   (6.0, 0.4), (105, 15)],
    "mango":     [(20, 6),  (28, 6),  (30, 5),   (31, 2),   (50, 5),   (5.8, 0.5), (95, 15)],
    "grapes":    [(23, 6),  (133, 8), (200, 8),  (24, 4),   (82, 4),   (6.0, 0.4), (70, 10)],
    "watermelon":[(100, 10),(18, 5),  (50, 5),   (25, 2),   (85, 4),   (6.5, 0.4), (50, 8)],
    "cotton":    [(118, 10),(46, 8),  (20, 4),   (24, 2),   (80, 5),   (6.9, 0.5), (80, 12)],
    "coffee":    [(100, 10),(28, 6),  (30, 5),   (25, 3),   (58, 5),   (6.8, 0.4), (158, 25)],
    "lentil":    [(20, 6),  (68, 8),  (20, 5),   (25, 4),   (65, 6),   (6.9, 0.4), (46, 8)],
    "pomegranate":[(20, 6), (18, 5),  (40, 5),   (22, 3),   (90, 4),   (6.4, 0.5), (108, 15)],
}
FEATURES = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]


def generate(samples_per_crop: int = 200) -> pd.DataFrame:
    frames = []
    for crop, profile in CROP_PROFILES.items():
        cols = {f: RNG.normal(m, s, samples_per_crop) for f, (m, s) in zip(FEATURES, profile)}
        df = pd.DataFrame(cols)
        df["label"] = crop
        frames.append(df)
    data = pd.concat(frames, ignore_index=True)
    # keep values physically valid
    data[["N", "P", "K"]] = data[["N", "P", "K"]].clip(0, 300)
    data["humidity"] = data["humidity"].clip(5, 100)
    data["ph"] = data["ph"].clip(3.5, 9.5)
    data["rainfall"] = data["rainfall"].clip(10, 300)
    data["temperature"] = data["temperature"].clip(8, 45)
    return data.sample(frac=1, random_state=42).reset_index(drop=True).round(2)


if __name__ == "__main__":
    out = Path("data")
    out.mkdir(exist_ok=True)
    df = generate()
    df.to_csv(out / "crop_data.csv", index=False)
    print(f"Saved {len(df)} rows, {df['label'].nunique()} crops -> data/crop_data.csv")
