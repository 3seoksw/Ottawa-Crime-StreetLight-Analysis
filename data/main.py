import os
import argparse
import pandas as pd
import geopandas as gpd
import numpy as np
from shapely.geometry import box
from crime_data.preprocess import (
    load_crime_data,
    parse_crime_dates,
    add_crime_group,
    filter_areas,
    assign_crimes_to_grid,
    add_crime_features,
    get_period_freq,
    print_crime_ratios,
    plot_crime_hexbin,
)
from street_light_data.preprocess import (
    load_street_light_data,
    plot_street_lights,
    assign_street_lights_to_grid,
)
from pathlib import Path


def build_grid_cells(gdf: gpd.GeoDataFrame, cell_size: int = 50):
    min_x, min_y, max_x, max_y = gdf.total_bounds
    x_range = np.arange(min_x, max_x + cell_size, cell_size)
    y_range = np.arange(min_y, max_y + cell_size, cell_size)

    cells, cell_ids = [], []
    for i, x in enumerate(x_range):
        for j, y in enumerate(y_range):
            x, y = float(x), float(y)
            grid = box(x, y, x + cell_size, y + cell_size)
            cells.append(grid)
            cell_ids.append(i * len(y_range) + j)

    grids = gpd.GeoDataFrame({"cell_id": cell_ids, "geometry": cells}, crs=gdf.crs)
    grids["centroid_x"] = grids.geometry.centroid.x
    grids["centroid_y"] = grids.geometry.centroid.y
    assert (
        (grids.groupby("cell_id")[["centroid_x", "centroid_y"]].nunique() == 1)
        .all()
        .all()
    )
    print(f"Grid cells created: {len(grids):,}, {grids.crs}")
    return grids


def preprocess_crime_data(record_frequency: str = "monthly", cell_size: int = 400):
    print("============ CRIME DATA ============ ")
    par_dir = "data"
    crime_data_name = "crime_data/Criminal_Offences_Open_Data_-621494644292511792.csv"
    crime_data_name = f"{par_dir}/{crime_data_name}"

    crime_gdf = load_crime_data(crime_data_name)
    crime_gdf = parse_crime_dates(crime_gdf)
    crime_gdf = add_crime_group(crime_gdf)

    prev_len = len(crime_gdf)
    crime_gdf = filter_areas(crime_gdf)
    post_len = len(crime_gdf)
    print(f"{prev_len - post_len:,} dropped after filtering areas")
    # plot_crime_hexbin(crime_gdf, True)

    if crime_gdf.crs is None or crime_gdf.crs.to_epsg() != 2951:
        crime_gdf = crime_gdf.to_crs(epsg=2951)
    grids = build_grid_cells(crime_gdf, cell_size)

    panel = assign_crimes_to_grid(crime_gdf, grids, record_frequency=record_frequency)
    panel = add_crime_features(panel)
    print_crime_ratios(panel)
    return panel, grids


def preprocess_street_light_data():
    print("============ STREET LIGHT DATA ============ ")
    par_dir = "data"
    street_light_name = "street_light_data/Street_Lights.csv"
    street = f"{par_dir}/{street_light_name}"

    light_gdf = load_street_light_data(street)
    # plot_street_lights(light_gdf)

    return light_gdf


def validate_panel(panel: pd.DataFrame):
    assert panel[["cell_id", "time_period", "crime_group"]].isna().sum().sum() == 0
    assert panel.duplicated(["cell_id", "time_period", "crime_group"]).sum() == 0
    assert (panel["crime_count"] >= 0).all()
    assert (panel["prev_crime_count"] >= 0).all()
    assert (panel["avg_crime_count"] >= 0).all()
    assert (panel["cumulative_crime_count"] >= 0).all()

    assert set(panel["crime_group"].unique()) == {0, 1}
    assert panel["crime_count"].dtype == int
    assert (panel["cumulative_crime_count"] >= panel["prev_crime_count"]).all()
    print(panel.columns)
    print("Validation passed")


def encode_time_period(
    df: pd.DataFrame,
    record_frequency: str = "monthly",
) -> pd.DataFrame:
    df = df.copy()
    period_freq = get_period_freq(record_frequency)
    dt = df["time_period"].dt.to_timestamp()

    df["year"] = dt.dt.year

    if period_freq == "M":
        cycle_position = dt.dt.month
        cycle_length = 12
    else:
        cycle_position = dt.dt.quarter
        cycle_length = 4

    df["time_sin"] = np.sin(2 * np.pi * cycle_position / cycle_length)
    df["time_cos"] = np.cos(2 * np.pi * cycle_position / cycle_length)

    min_period = df["time_period"].min()
    df["time_index"] = (df["time_period"] - min_period).apply(lambda x: x.n)
    return df


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build the Ottawa crime-light panel dataset."
    )
    parser.add_argument(
        "--record-frequency",
        choices=["monthly", "quarterly"],
        default="quarterly",
        help="Temporal aggregation level for crime records.",
    )
    parser.add_argument("--cell_size", type=int, default=400)
    return parser.parse_args()


def main():
    args = parse_args()

    panel, grids = preprocess_crime_data(
        record_frequency=args.record_frequency, cell_size=args.cell_size
    )
    light = preprocess_street_light_data()
    print(f"============ AGGREGATION ({args.record_frequency}) ============ ")
    lights_cell = assign_street_lights_to_grid(light, grids)

    data_panel = panel.merge(lights_cell, on="cell_id", how="left")

    light_feats = [
        "light_count",
        "total_wattage",
        "total_intensity",
        "avg_wattage",
    ]
    data_panel[light_feats] = data_panel[light_feats].fillna(0)
    data_panel["avg_install_month"] = data_panel["avg_install_month"].fillna(-1)

    # Final Aggregation (coordinates included)
    grid_info = ["cell_id", "geometry", "centroid_x", "centroid_y"]
    data_panel = data_panel.merge(
        grids[grid_info],
        on="cell_id",
        how="left",
    )
    data_panel = gpd.GeoDataFrame(data_panel, geometry="geometry", crs=grids.crs)
    data_panel = encode_time_period(
        data_panel,
        record_frequency=args.record_frequency,
    )

    print(data_panel)
    non_zero = data_panel["crime_count"] != 0
    zero = data_panel["crime_count"] == 0
    print(f"Final number of records: {len(data_panel):,}")
    print(f"Non-Zero records: {len(data_panel[non_zero]):,}")
    print(f"Zero records: {len(data_panel[zero]):,}")

    validate_panel(data_panel)

    # Save
    print(f"Saving data_panel_{args.cell_size}.csv ...")
    if not Path("data/preprocessed").exists():
        os.mkdir("data/preprocessed")

    data_panel.to_csv(f"data/preprocessed/data_panel_{args.cell_size}.csv", index=False)


if __name__ == "__main__":
    main()
