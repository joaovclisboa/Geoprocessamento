"""Enriquece a tabela com índices espectrais calculados a partir das bandas Sentinel-2."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

TABLE_INPUT = Path("arroz_concat_to_20251125.parquet")

SENTINEL_BANDS = ["B02", "B03", "B04", "B08", "B11", "B12"]
PLANET_BANDS = ["PL_B1", "PL_B2", "PL_B3", "PL_B4"]

def safe_ratio(num: pd.Series, den: pd.Series) -> pd.Series:
	"""Normaliza divisões evitando infinito e NaN."""
	out = num.astype("float32") / den.astype("float32")
	return out.replace([np.inf, -np.inf], np.nan)


def compute_sentinel_indices(df: pd.DataFrame) -> pd.DataFrame:
    missing = [col for col in SENTINEL_BANDS if col not in df.columns]
    if missing:
        raise ValueError(f"Colunas Sentinel ausentes: {missing}")

    data = df.copy()
    blue = data["B02"].astype("float32")
    green = data["B03"].astype("float32")
    red = data["B04"].astype("float32")
    nir = data["B08"].astype("float32")
    swir1 = data["B11"].astype("float32")
    swir2 = data["B12"].astype("float32")

    idx = {}
    idx["S2_NDVI"] = safe_ratio(nir - red, nir + red)
    idx["S2_GNDVI"] = safe_ratio(nir - green, nir + green)
    idx["S2_NDWI"] = safe_ratio(green - nir, green + nir)
    idx["S2_MNDWI"] = safe_ratio(green - swir1, green + swir1)
    idx["S2_AWEI_sh"] = (blue + 2.5 * green - 1.5 * (nir + swir1) - 0.25 * swir2).astype("float32")
    idx["S2_AWEI_nsh"] = (4 * (green - swir1) - (0.25 * nir + 2.75 * swir2)).astype("float32")
    idx["S2_EVI"] = safe_ratio(2.5 * (nir - red), nir + 6 * red - 7.5 * blue + 1)

    for name, series in idx.items():
        data[name] = series.astype("float32")

    return data

def compute_planet_indices(df: pd.DataFrame) -> pd.DataFrame:
    missing = [col for col in PLANET_BANDS if col not in df.columns]
    if missing:
        raise ValueError(f"Colunas Planet ausentes: {missing}")

    data = df.copy()
    blue = data["PL_B1"].astype("float32")
    green = data["PL_B2"].astype("float32")
    red = data["PL_B3"].astype("float32")
    nir = data["PL_B4"].astype("float32")

    idx = {}
    idx["P_NDVI"] = safe_ratio(nir - red, nir + red)
    idx["P_GNDVI"] = safe_ratio(nir - green, nir + green)
    idx["P_NDWI"] = safe_ratio(green - nir, green + nir)

    for name, series in idx.items():
        data[name] = series.astype("float32")

    return data



def main() -> None:
    if not TABLE_INPUT.exists():
        raise FileNotFoundError(f"Parquet não encontrado: {TABLE_INPUT}")

    table = pd.read_parquet(TABLE_INPUT)
    print(f"Tabela carregada: {TABLE_INPUT} ({len(table):,} linhas)")

    enriched = compute_sentinel_indices(table)
    added_cols = sorted(set(enriched.columns) - set(table.columns))
    print(f"Índices adicionados: {added_cols}")

    output_path = TABLE_INPUT.with_name(TABLE_INPUT.stem + "_sentinel_idx.parquet")
    enriched.to_parquet(output_path, index=False)
    print(f"Arquivo salvo em: {output_path}")


if __name__ == "__main__":
    main()
