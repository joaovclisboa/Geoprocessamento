#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Enriquecimento vetorizado com NDVI, SOLO e ELEVAÇÃO,
agora com busca inteligente de arquivos (sem subpastas).

Novidades:
- parâmetros de linha de comando para personalizar diretórios/arquivos;
- opção de anexar novas amostras a partir de um shapefile (ex.: Tocantins);
- relatório detalhado dos rasters encontrados por tile.

Funções:
- procura NDVI/SOLO: <tile>.tif com variações
- procura ELEV: <tile>*mad_10m.tif e <tile>*slope_10m.tif
"""

import argparse
import logging
from pathlib import Path
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.transform import rowcol
from shapely.geometry import Point

# =============================================================================
# CONFIG
# =============================================================================

TABLE_INPUT  = Path("s2_amostras_arroz_20240604_atualizado_novasbandas.parquet")
TABLE_OUTPUT = Path("s2_amostras_arroz_20240604_com_ndvi_elev_solo_fast.parquet")

BASE_DIR = Path(r"\\Nebula\homes\Ester\UNet_arroz\dataset")
SENTINEL_DIR = BASE_DIR / "sentinel" / "inteiros"
PLANET_DIR = BASE_DIR / "planet" / "inteiros"
NDVI_DIR = BASE_DIR / "variacao_ndvi" / "inteiros   "
SOLO_DIR = BASE_DIR / "solos" / "inteiros"
ELEV_DIR = BASE_DIR / "elevacao" / "inteiros"

MAX_IN_MEMORY_RASTER_BYTES = 2 * 1024**3  # 2 GiB

COLS_SENTINEL = ("B02","B03","B04","B05","B06","B07","B08","B11","B12","B8A")
COLS_PLANET   = ("PL_B1","PL_B2","PL_B3","PL_B4")
COLS_NDVI = ("NDVI_RANGE_ANUAL", "NDVI_STD_ANUAL")
COL_SOLO  = "SOLO"
COL_ELEV_MAD   = "ELEV_MAD_10M"
COL_ELEV_SLOPE = "ELEV_SLOPE_10M"
EXTRA_FLAG_COLUMN = "__extra_sample__"


# =============================================================================
# HELPERS
# =============================================================================

def candidate_tile_variants(tile_id: str, season_variants: list[str] | tuple[str, ...] | None = None):
    """
    Gera variações de nome para tentar localizar arquivos.
    Ex.: pol_58 → pol_58, pol58, aoi_58, aoi58, 58
    """
    tile_id = str(tile_id).strip().lower()

    if not tile_id:
        return []

    cands = set()

    # original
    cands.add(tile_id)

    # sem underline
    if "_" in tile_id:
        cands.add(tile_id.replace("_", ""))

    # prefixos pol_XX → aoi_XX
    if tile_id.startswith("pol_"):
        suffix = tile_id.split("_", 1)[1]
        cands.add(f"aoi_{suffix}")
        cands.add(f"aoi{suffix}")
        if suffix.isdigit():
            cands.add(f"aoi_{int(suffix):02d}")
            cands.add(f"aoi{int(suffix):02d}")
            cands.add(suffix)
            cands.add(str(int(suffix)))

    # aoi → pol
    if tile_id.startswith("aoi_"):
        suffix = tile_id.split("_", 1)[1]
        cands.add(f"pol_{suffix}")
        cands.add(f"pol{suffix}")
        if suffix.isdigit():
            cands.add(f"pol_{int(suffix):02d}")
            cands.add(f"pol{int(suffix):02d}")
            cands.add(suffix)
            cands.add(str(int(suffix)))

    seasons = [str(s).strip().lower() for s in (season_variants or []) if str(s).strip()]
    if seasons:
        extra = set()
        for variant in cands:
            for season in seasons:
                extra.add(f"{variant}_{season}")
                extra.add(f"{variant}{season}")
                extra.add(f"{season}_{variant}")
        cands |= extra

    # dedup
    return list(cands)


def find_raster_simple(root: Path, tile_id: str, season_variants: list[str] | tuple[str, ...] | None = None) -> Path | None:
    """
    Procura NDVI/SOLO (<tile>.tif) sem subpastas.
    """
    variants = candidate_tile_variants(tile_id, season_variants)
    for variant in variants:
        p = root / f"{variant}.tif"
        if p.exists():
            return p
    for variant in variants:
        try:
            matches = sorted(root.rglob(f"{variant}.tif"))
        except Exception:
            matches = []
        if matches:
            return matches[0]
    return None


def find_raster_flexible(root: Path, tile_id: str, season_variants: list[str] | tuple[str, ...] | None = None) -> Path | None:
    """Procura raster permitindo subpastas (usado para Sentinel/Planet)."""

    variants = candidate_tile_variants(tile_id, season_variants)

    for variant in variants:
        direct = root / f"{variant}.tif"
        if direct.exists():
            return direct

        subdir = root / variant
        if subdir.is_dir():
            exact = subdir / f"{variant}.tif"
            if exact.exists():
                return exact
            matches = sorted(subdir.glob(f"{variant}_*.tif"))
            if matches:
                return matches[0]

        matches = sorted(root.glob(f"{variant}_*.tif"))
        if matches:
            return matches[0]

    for variant in variants:
        try:
            matches = sorted(root.rglob(f"{variant}*.tif"))
        except Exception:
            matches = []
        if matches:
            return matches[0]

    return None


def find_elevation(root: Path, tile_id: str, kind: str, season_variants: list[str] | tuple[str, ...] | None = None):
    """
    kind = "mad"  →  *_mad_10m.tif
    kind = "slope" → *_slope_10m.tif
    """
    suffix = "mad_10m" if kind == "mad" else "slope_10m"

    variants = candidate_tile_variants(tile_id, season_variants)
    for variant in variants:
        matches = list(root.glob(f"{variant}*{suffix}.tif"))
        if matches:
            return matches[0]
    for variant in variants:
        try:
            matches = sorted(root.rglob(f"{variant}*{suffix}.tif"))
        except Exception:
            matches = []
        if matches:
            return matches[0]
    return None


# =============================================================================
# VETORIZADO
# =============================================================================

def _ensure_subset_columns(df, cols):
    cols = list(cols)
    for c in cols:
        if c not in df.columns:
            df[c] = np.nan
    return df


def _apply_subset(gdf_full, subset, cols):
    cols = list(cols)
    for c in cols:
        if c not in gdf_full.columns:
            gdf_full[c] = np.nan
    gdf_full.loc[subset.index, cols] = subset[cols]
    return gdf_full


def _should_read_all(src, nbands):
    bytes_needed = nbands * src.width * src.height * np.dtype("float32").itemsize
    return bytes_needed <= MAX_IN_MEMORY_RASTER_BYTES


def _read_all(src):
    arr = src.read(out_dtype="float32")
    if src.nodata is not None:
        arr = np.where(arr == src.nodata, np.nan, arr)
    try:
        masks = src.read_masks()
        arr = np.where(masks == 0, np.nan, arr)
    except:
        pass
    return arr


def _safe_output_path(inp: Path, out: Path, allow_overwrite: bool) -> Path:
    try:
        same = inp.resolve() == out.resolve()
    except FileNotFoundError:
        same = inp.absolute() == out.absolute()

    if same and not allow_overwrite:
        suffix = out.suffix or ".parquet"
        candidate = out.with_name(f"{out.stem}_enriquecido{suffix}")
        logging.warning(
            "Saída coincide com a entrada. Gravando em %s (use --allow-overwrite para forçar mesmo arquivo)",
            candidate,
        )
        return candidate
    return out


def _random_point_within_geometry(geom, rng, max_tries: int = 500):
    """Gera um ponto aleatório dentro de um polígono (fallback: representative_point)."""

    if geom.is_empty:
        return geom

    geom_type = geom.geom_type

    if geom_type == "Point":
        return geom

    if geom_type == "MultiPoint":
        pts = list(geom.geoms)
        return pts[rng.integers(0, len(pts))]

    if geom_type == "MultiPolygon":
        polys = [g for g in geom.geoms if not g.is_empty]
        if not polys:
            return geom.representative_point()
        areas = np.array([p.area for p in polys], dtype=float)
        probs = areas / areas.sum() if areas.sum() > 0 else None
        chosen = polys[rng.choice(len(polys), p=probs)]
        return _random_point_within_geometry(chosen, rng, max_tries)

    if geom_type != "Polygon":
        return geom.representative_point()

    minx, miny, maxx, maxy = geom.bounds
    for _ in range(max_tries):
        p = Point(rng.uniform(minx, maxx), rng.uniform(miny, maxy))
        if geom.contains(p):
            return p
    return geom.representative_point()


def _geometry_to_points(geom, mode: str, samples_per_feature: int, rng):
    mode = (mode or "auto").lower()
    samples = max(1, samples_per_feature)

    if mode == "keep":
        if geom.geom_type == "Point":
            return [geom]
        if geom.geom_type == "MultiPoint":
            return list(geom.geoms)
        return [geom.representative_point()]

    if mode == "centroid":
        return [geom.centroid]

    if mode == "random":
        return [_random_point_within_geometry(geom, rng) for _ in range(samples)]

    # auto
    if geom.geom_type in {"Point", "MultiPoint"}:
        return [geom] if geom.geom_type == "Point" else list(geom.geoms)
    return [geom.centroid]


def _prepare_extra_geometry(extra_gdf, mode: str, samples_per_feature: int, seed: int | None):
    rng = np.random.default_rng(seed)
    rows = []

    for _, row in extra_gdf.iterrows():
        geom = row.geometry
        if geom is None or geom.is_empty:
            continue

        for point in _geometry_to_points(geom, mode, samples_per_feature, rng):
            new_row = row.copy()
            new_row.geometry = point
            rows.append(new_row)

    if not rows:
        raise ValueError("Nenhuma geometria válida encontrada no shapefile informado.")

    return gpd.GeoDataFrame(rows, geometry="geometry", crs=extra_gdf.crs)


def append_extra_samples(
    base_gdf: gpd.GeoDataFrame,
    shapefile_path: Path,
    tile_field: str = "tile_id",
    season_field: str = "season",
    geometry_mode: str = "auto",
    samples_per_feature: int = 1,
    seed: int | None = 42,
    force_tile_value: str | None = None,
    force_season_values: list[str] | None = None,
):
    """Anexa novas amostras de um shapefile antes do enriquecimento."""

    extra = gpd.read_file(shapefile_path)

    if tile_field not in extra.columns and not force_tile_value:
        raise ValueError(f"Coluna '{tile_field}' não encontrada em {shapefile_path}.")

    if tile_field in extra.columns:
        extra = extra.rename(columns={tile_field: "tile_id"})

    if force_tile_value:
        extra["tile_id"] = force_tile_value

    extra = extra[extra["tile_id"].notna()].copy()
    extra["tile_id"] = extra["tile_id"].astype(str).str.strip()

    if season_field in extra.columns:
        extra = extra.rename(columns={season_field: "season"})

    extra = _prepare_extra_geometry(extra, geometry_mode, samples_per_feature, seed)

    if force_season_values:
        expanded = []
        cleaned = [s for s in (force_season_values or []) if s]
        if not cleaned:
            cleaned = [None]
        for season in cleaned:
            tmp = extra.copy()
            tmp["season"] = season
            expanded.append(tmp)
        extra = pd.concat(expanded, ignore_index=True)

    target_crs = base_gdf.crs or extra.crs
    if target_crs is None:
        raise ValueError("Não foi possível determinar o CRS alvo para reprojeção.")

    base_gdf = base_gdf.to_crs(target_crs)
    extra = extra.to_crs(target_crs)

    for col in base_gdf.columns:
        if col not in extra.columns:
            extra[col] = np.nan

    if "crop_stage" in extra.columns:
        extra.loc[:, "crop_stage"] = 1

    extra = extra[base_gdf.columns]

    if EXTRA_FLAG_COLUMN not in base_gdf.columns:
        base_gdf = base_gdf.copy()
        base_gdf[EXTRA_FLAG_COLUMN] = False

    extra[EXTRA_FLAG_COLUMN] = True

    combined = pd.concat([base_gdf, extra], ignore_index=True)
    return gpd.GeoDataFrame(combined, geometry="geometry", crs=target_crs)


def amostrar(subset, raster_path, columns):
    subset = _ensure_subset_columns(subset.copy(), columns)

    if not raster_path or not raster_path.exists():
        for c in columns:
            subset[c] = np.nan
        return subset

    with rasterio.open(raster_path) as src:
        # reproject subset if CRS differs
        if subset.crs != src.crs:
            subset = subset.to_crs(src.crs)

        xs = subset.geometry.x.to_numpy()
        ys = subset.geometry.y.to_numpy()

        rows, cols = rowcol(src.transform, xs, ys)
        rows = rows.astype(int)
        cols = cols.astype(int)

        H, W = src.height, src.width
        inside = (rows>=0)&(rows<H)&(cols>=0)&(cols<W)

        nb = len(columns)
        result = np.full((len(subset), nb), np.nan, dtype=np.float32)

        if _should_read_all(src, src.count):
            try:
                arr = _read_all(src)
                use = min(arr.shape[0], nb)
                for b in range(use):
                    result[inside, b] = arr[b, rows[inside], cols[inside]]
            except:
                pass

        if np.isnan(result).all():
            coords = list(zip(xs[inside], ys[inside]))
            raw = np.asarray(list(src.sample(coords)), dtype=np.float32)
            if src.nodata is not None:
                raw = np.where(raw == src.nodata, np.nan, raw)
            use = min(raw.shape[1], nb)
            result[inside, :use] = raw[:, :use]

        for i, col in enumerate(columns):
            subset[col] = result[:, i]

    return subset


# =============================================================================
# PROCESSAMENTO
# =============================================================================

def processar_novas_bandas(
    gdf,
    ndvi_dir: Path,
    solo_dir: Path,
    elev_dir: Path,
    sentinel_dir: Path | None = None,
    planet_dir: Path | None = None,
    sentinel_file: Path | None = None,
    planet_file: Path | None = None,
    ndvi_file: Path | None = None,
    solo_file: Path | None = None,
    elev_mad_file: Path | None = None,
    elev_slope_file: Path | None = None,
    tile_filter: list[str] | None = None,
    show_paths: bool = False,
):

    for c in (*COLS_SENTINEL, *COLS_PLANET, *COLS_NDVI, COL_SOLO, COL_ELEV_MAD, COL_ELEV_SLOPE):
        if c not in gdf.columns:
            gdf[c] = np.nan

    tiles = gdf["tile_id"].dropna().astype(str).unique()

    if tile_filter:
        wanted = {t.lower() for t in tile_filter}
        tiles = [t for t in tiles if t.lower() in wanted]

    print(f"🔹 {len(tiles)} tiles distintos processados.")

    for tile in tiles:
        print(f"\n🧩 Tile: {tile}")

        idx = gdf[gdf["tile_id"].astype(str)==tile].index
        subset = gdf.loc[idx].copy()
        season_variants = subset.get("season")
        if season_variants is not None:
            seasons = season_variants.dropna().astype(str).str.lower().unique().tolist()
        else:
            seasons = []

        # SENTINEL
        if sentinel_file or sentinel_dir:
            s2 = sentinel_file if sentinel_file else find_raster_flexible(sentinel_dir, tile, seasons)
            if show_paths:
                sentinel_source = sentinel_file if sentinel_file else sentinel_dir
                print(f"   S2   : {s2 if s2 else f'não encontrado em {sentinel_source}'}")
            subset = amostrar(subset, s2, COLS_SENTINEL)
            gdf = _apply_subset(gdf, subset, COLS_SENTINEL)

        # PLANET
        if planet_file or planet_dir:
            pl = planet_file if planet_file else find_raster_flexible(planet_dir, tile, seasons)
            if show_paths:
                planet_source = planet_file if planet_file else planet_dir
                print(f"   PL   : {pl if pl else f'não encontrado em {planet_source}'}")
            subset = amostrar(subset, pl, COLS_PLANET)
            gdf = _apply_subset(gdf, subset, COLS_PLANET)

        # NDVI
        if ndvi_file or ndvi_dir:
            ndvi = ndvi_file if ndvi_file else find_raster_simple(ndvi_dir, tile, seasons)
            if show_paths:
                ndvi_source = ndvi_file if ndvi_file else ndvi_dir
                print(f"   NDVI : {ndvi if ndvi else f'não encontrado em {ndvi_source}'}")
            subset = amostrar(subset, ndvi, COLS_NDVI)
            gdf = _apply_subset(gdf, subset, COLS_NDVI)

        # SOLO
        if solo_file or solo_dir:
            solo = solo_file if solo_file else find_raster_simple(solo_dir, tile, seasons)
            if show_paths:
                solo_source = solo_file if solo_file else solo_dir
                print(f"   SOLO : {solo if solo else f'não encontrado em {solo_source}'}")
            subset = amostrar(subset, solo, [COL_SOLO])
            gdf = _apply_subset(gdf, subset, [COL_SOLO])

        # ELEVAÇÃO
        if elev_mad_file or elev_dir:
            elev_mad = elev_mad_file if elev_mad_file else find_elevation(elev_dir, tile, "mad", seasons)
            if show_paths:
                mad_source = elev_mad_file if elev_mad_file else elev_dir
                print(f"   MAD  : {elev_mad if elev_mad else f'não encontrado em {mad_source}'}")
            subset = amostrar(subset, elev_mad, [COL_ELEV_MAD])
            gdf = _apply_subset(gdf, subset, [COL_ELEV_MAD])

        if elev_slope_file or elev_dir:
            elev_slope = elev_slope_file if elev_slope_file else find_elevation(elev_dir, tile, "slope", seasons)
            if show_paths:
                slope_source = elev_slope_file if elev_slope_file else elev_dir
                print(f"   SLOPE: {elev_slope if elev_slope else f'não encontrado em {slope_source}'}")
            subset = amostrar(subset, elev_slope, [COL_ELEV_SLOPE])
            gdf = _apply_subset(gdf, subset, [COL_ELEV_SLOPE])

    return gdf


# =============================================================================
# MAIN
# =============================================================================

def build_parser():
    parser = argparse.ArgumentParser(
        description="Enriquecimento vetorial com NDVI/SOLO/ELEVAÇÃO e ingestão opcional de novos shapefiles."
    )

    parser.add_argument("--table-input", type=Path, default=TABLE_INPUT,
                        help="Tabela base (Parquet) já existente.")
    parser.add_argument("--table-output", type=Path, default=TABLE_OUTPUT,
                        help="Arquivo Parquet de saída com colunas enriquecidas.")

    parser.add_argument("--base-dir", type=Path,
                        help="Diretório raiz contendo subpastas variacao_ndvi/solos/elevacao.")
    parser.add_argument("--ndvi-dir", type=Path, help="Pasta que contém rasters de NDVI.")
    parser.add_argument("--solo-dir", type=Path, help="Pasta que contém rasters de solos.")
    parser.add_argument("--elev-dir", type=Path, help="Pasta que contém rasters de elevação.")
    parser.add_argument("--sentinel-dir", type=Path, help="Pasta com rasters multibanda do Sentinel.")
    parser.add_argument("--planet-dir", type=Path, help="Pasta com rasters multibanda do Planet.")
    parser.add_argument("--sentinel-file", type=Path,
                        help="Arquivo raster do Sentinel a ser usado diretamente (ignora busca por tile).")
    parser.add_argument("--planet-file", type=Path,
                        help="Arquivo raster do Planet a ser usado diretamente (ignora busca por tile).")
    parser.add_argument("--ndvi-file", type=Path,
                        help="Arquivo NDVI específico para usar em todos os pontos.")
    parser.add_argument("--solo-file", type=Path,
                        help="Arquivo de solo específico para usar em todos os pontos.")
    parser.add_argument("--elev-mad-file", type=Path,
                        help="Arquivo de elevação MAD 10m específico para usar em todos os pontos.")
    parser.add_argument("--elev-slope-file", type=Path,
                        help="Arquivo de declividade 10m específico para usar em todos os pontos.")

    parser.add_argument("--only-tiles", nargs="+",
                        help="Processar apenas estes tiles (ex.: --only-tiles tocantins pol1).")
    parser.add_argument("--season-filter", nargs="+",
                        help="Mantém somente estas seasons na tabela final (ex.: --season-filter abril_maio set).")
    parser.add_argument("--show-paths", action="store_true",
                        help="Imprime os caminhos encontrados para cada raster por tile.")
    parser.add_argument("--allow-overwrite", action="store_true",
                        help="Permite sobrescrever o arquivo de saída se igual ao de entrada.")
    parser.add_argument("--append-output", action="store_true",
                        help="Se o arquivo de saída existir, acrescenta apenas as novas amostras ao final dele.")
    parser.add_argument("--output-only-extras", action="store_true",
                        help="Grava somente as amostras vindas do shapefile extra (ignora linhas originais da tabela base).")
    parser.add_argument("--drop-nodata", action="store_true",
                        help="Remove linhas cujos rasters requeridos ficaram com valores NaN/nodata após a amostragem.")
    parser.add_argument("--require-sentinel", action="store_true",
                        help="Ao usar --drop-nodata, exige que todas as bandas Sentinel estejam preenchidas.")
    parser.add_argument("--require-planet", action="store_true",
                        help="Ao usar --drop-nodata, exige que todas as bandas Planet estejam preenchidas.")
    parser.add_argument("--require-elevation", action="store_true",
                        help="Ao usar --drop-nodata, exige que as duas bandas de elevação estejam preenchidas.")
    parser.add_argument("--require-ndvi", action="store_true",
                        help="Ao usar --drop-nodata, exige que as colunas de NDVI estejam preenchidas.")
    parser.add_argument("--require-solo", action="store_true",
                        help="Ao usar --drop-nodata, exige que a coluna de solo esteja preenchida.")

    # Shapefile extra
    parser.add_argument("--extra-shapefile", type=Path,
                        help="Shapefile com novas geometrias a anexar antes do enriquecimento.")
    parser.add_argument("--extra-tile-field", default="tile_id",
                        help="Nome da coluna no shapefile que indica o tile.")
    parser.add_argument("--force-tile-value",
                        help="Define um tile fixo para todas as features do shapefile (ignora coluna).")
    parser.add_argument("--extra-geometry-mode", choices=["auto", "centroid", "random", "keep"],
                        default="auto", help="Como converter polígonos/linhas em pontos de amostragem.")
    parser.add_argument("--extra-samples-per-feature", type=int, default=1,
                        help="Número de pontos aleatórios por feição (apenas no modo random).")
    parser.add_argument("--random-seed", type=int, default=42,
                        help="Seed para geração de pontos aleatórios no shapefile.")
    parser.add_argument("--extra-season-field", default="season",
                        help="Coluna do shapefile com o valor de season. (default: %(default)s)")
    parser.add_argument("--force-season-values", nargs="+",
                        help="Lista de seasons a atribuir às novas amostras (duplica os pontos para cada valor se necessário).")

    return parser


def main(args=None):

    parser = build_parser()
    args = parser.parse_args(args=args)

    logging.basicConfig(level=logging.INFO)
    print(f"\n### ENRIQUECIMENTO — NDVI / SOLO / ELEV ###\n")

    table_input: Path = Path(args.table_input)
    table_output: Path = _safe_output_path(table_input, Path(args.table_output), args.allow_overwrite)

    base_override = args.base_dir is not None
    base_dir = Path(args.base_dir) if base_override else BASE_DIR

    def _dir(arg_value, default_from_base: Path, subpath: str):
        if arg_value:
            return Path(arg_value)
        if base_override:
            return base_dir / subpath
        return default_from_base

    ndvi_dir = _dir(args.ndvi_dir, NDVI_DIR, Path("variacao_ndvi") / "inteiros")
    solo_dir = _dir(args.solo_dir, SOLO_DIR, Path("solos") / "inteiros")
    elev_dir = _dir(args.elev_dir, ELEV_DIR, Path("elevacao") / "inteiros")
    sentinel_dir = _dir(args.sentinel_dir, SENTINEL_DIR, Path("sentinel") / "inteiros")
    planet_dir = _dir(args.planet_dir, PLANET_DIR, Path("planet") / "inteiros")
    sentinel_file = Path(args.sentinel_file) if args.sentinel_file else None
    planet_file = Path(args.planet_file) if args.planet_file else None
    ndvi_file = Path(args.ndvi_file) if args.ndvi_file else None
    solo_file = Path(args.solo_file) if args.solo_file else None
    elev_mad_file = Path(args.elev_mad_file) if args.elev_mad_file else None
    elev_slope_file = Path(args.elev_slope_file) if args.elev_slope_file else None

    gdf = gpd.read_parquet(table_input)
    print(f"✅ {len(gdf):,} pontos carregados da tabela base.")

    if EXTRA_FLAG_COLUMN not in gdf.columns:
        gdf[EXTRA_FLAG_COLUMN] = False

    if "geometry" not in gdf.columns:
        raise ValueError("Tabela não possui a coluna geometry.")

    if "crop_stage" in gdf.columns:
        before_counts = gdf["crop_stage"].value_counts(dropna=False)
        gdf["crop_stage"] = gdf["crop_stage"].fillna(0)
        gdf.loc[gdf["crop_stage"].isin([0, 1, 2]), "crop_stage"] = 0
        gdf.loc[gdf["crop_stage"] >= 3, "crop_stage"] = 1
        after_counts = gdf["crop_stage"].value_counts(dropna=False)
        print("🔁 crop_stage binarizado (0/1). Distribução antes/depois:\n", before_counts, "\n", after_counts)

    if args.extra_shapefile:
        before = len(gdf)
        gdf = append_extra_samples(
            gdf,
            shapefile_path=Path(args.extra_shapefile),
            tile_field=args.extra_tile_field,
            season_field=args.extra_season_field,
            geometry_mode=args.extra_geometry_mode,
            samples_per_feature=args.extra_samples_per_feature,
            seed=args.random_seed,
            force_tile_value=args.force_tile_value,
            force_season_values=args.force_season_values,
        )
        added = len(gdf) - before
        print(f"➕ {added:,} novas amostras adicionadas a partir de {args.extra_shapefile}.")

    if args.season_filter:
        if "season" not in gdf.columns:
            raise ValueError("Season filter informado, mas a coluna 'season' não existe na tabela.")
        allowed = {s.lower() for s in args.season_filter}
        before = len(gdf)
        mask = gdf["season"].astype(str).str.lower().isin(allowed)
        gdf = gdf[mask].copy()
        removed = before - len(gdf)
        print(f"🔎 Season filter aplicado ({', '.join(args.season_filter)}). Linhas removidas: {removed:,}.")

    if args.output_only_extras and args.append_output:
        raise ValueError("--output-only-extras não pode ser combinado com --append-output. Use um arquivo de saída separado.")

    gdf = processar_novas_bandas(
        gdf,
        ndvi_dir=ndvi_dir,
        solo_dir=solo_dir,
        elev_dir=elev_dir,
        sentinel_dir=sentinel_dir,
        planet_dir=planet_dir,
        sentinel_file=sentinel_file,
        planet_file=planet_file,
        ndvi_file=ndvi_file,
        solo_file=solo_file,
        elev_mad_file=elev_mad_file,
        elev_slope_file=elev_slope_file,
        tile_filter=args.only_tiles,
        show_paths=args.show_paths,
    )

    if args.drop_nodata:
        required_columns: list[str] = []
        if (sentinel_file or sentinel_dir) and args.require_sentinel:
            required_columns.extend(COLS_SENTINEL)
        if (planet_file or planet_dir) and args.require_planet:
            required_columns.extend(COLS_PLANET)
        if (ndvi_file or ndvi_dir) and args.require_ndvi:
            required_columns.extend(COLS_NDVI)
        if (solo_file or solo_dir) and args.require_solo:
            required_columns.append(COL_SOLO)
        if (elev_mad_file or elev_dir) and args.require_elevation:
            required_columns.append(COL_ELEV_MAD)
        if (elev_slope_file or elev_dir) and args.require_elevation:
            required_columns.append(COL_ELEV_SLOPE)

        required_columns = [c for c in required_columns if c in gdf.columns]
        if required_columns:
            before_drop = len(gdf)
            mask_valid = ~gdf[required_columns].isna().any(axis=1)
            gdf = gdf[mask_valid].copy()
            removed = before_drop - len(gdf)
            print(f"🧹 drop-nodata: {removed:,} linhas removidas por conterem valores NaN em {len(required_columns)} colunas requeridas.")
            if gdf.empty:
                raise ValueError("drop-nodata removeu todas as linhas. Ajuste os requisitos (--require-*) ou verifique os rasters.")
        else:
            logging.warning("--drop-nodata solicitado, mas nenhuma das colunas requeridas foi encontrada para validação.")

    if args.output_only_extras:
        if EXTRA_FLAG_COLUMN not in gdf.columns:
            raise ValueError("--output-only-extras exige que um shapefile tenha sido anexado nesta execução.")
        extra_mask = gdf[EXTRA_FLAG_COLUMN].fillna(False)
        gdf = gdf.loc[extra_mask].copy()
        if gdf.empty:
            raise ValueError("Nenhuma amostra extra encontrada para exportar.")
        gdf = gdf.drop(columns=[EXTRA_FLAG_COLUMN], errors="ignore")
        gdf.to_parquet(table_output)
        print(f"\n✅ Tabela gerada apenas com {len(gdf):,} novas amostras: {table_output}\n")
        return

    if args.append_output:
        if EXTRA_FLAG_COLUMN not in gdf.columns:
            raise ValueError("--append-output exige a coluna de controle de amostras extras. Reinicie após adicionar shapefile.")

        extra_mask = gdf[EXTRA_FLAG_COLUMN].fillna(False)
        extras_only = gdf.loc[extra_mask].drop(columns=[EXTRA_FLAG_COLUMN], errors="ignore")
        extras_only = extras_only.reset_index(drop=True)

        if extras_only.empty:
            logging.warning("Nenhuma nova amostra sobrou para acrescentar ao arquivo destino. Arquivo existente permanece inalterado.")
            if not table_output.exists():
                extras_only.to_parquet(table_output)
                print(f"\n✅ Arquivo criado, porém sem novas amostras: {table_output}\n")
            return

        if table_output.exists():
            existing = gpd.read_parquet(table_output)
            combined = pd.concat([existing, extras_only], ignore_index=True)
        else:
            combined = extras_only

        combined.to_parquet(table_output)
        print(f"\n✅ Acrescentadas {len(extras_only):,} novas amostras em: {table_output}\n")
    else:
        gdf = gdf.drop(columns=[EXTRA_FLAG_COLUMN], errors="ignore")
        gdf.to_parquet(table_output)
        print(f"\n✅ Salvo em: {table_output}\n")


if __name__ == "__main__":
    main()
