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
    print_crime_ratios,
    plot_crime_hexbin,
)
from street_light_data.preprocess import (
    load_street_light_data,
    plot_street_lights,
    assign_street_lights_to_grid,
)


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


def preprocess_crime_data():
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
    grids = build_grid_cells(crime_gdf, 400)

    panel = assign_crimes_to_grid(crime_gdf, grids)
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
    assert panel[["cell_id", "year_month", "crime_group"]].isna().sum().sum() == 0
    assert panel.duplicated(["cell_id", "year_month", "crime_group"]).sum() == 0
    assert (panel["crime_count"] >= 0).all()
    assert (panel["prev_crime_count"] >= 0).all()
    assert (panel["avg_crime_count"] >= 0).all()
    assert (panel["cumulative_crime_count"] >= 0).all()

    assert set(panel["crime_group"].unique()) == {0, 1}
    assert panel["crime_count"].dtype == int
    assert (panel["cumulative_crime_count"] >= panel["prev_crime_count"]).all()
    print("Validation passed")


def encode_year_month(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    dt = df["year_month"].dt.to_timestamp()

    df["year"] = dt.dt.year
    df["month_num"] = dt.dt.month

    df["month_sin"] = np.sin(2 * np.pi * dt.dt.month / 12)
    df["month_cos"] = np.cos(2 * np.pi * dt.dt.month / 12)

    min_period = df["year_month"].min()
    df["time_index"] = (df["year_month"] - min_period).apply(lambda x: x.n)

    df = df.drop(columns=["month_num"])
    # df = df.drop(columns=["year_month"])
    return df


def main():
    panel, grids = preprocess_crime_data()
    light = preprocess_street_light_data()
    print("============ AGGREGATION ============ ")
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
    data_panel = encode_year_month(data_panel)

    print(data_panel)
    non_zero = data_panel["crime_count"] != 0
    zero = data_panel["crime_count"] == 0
    print(f"Final number of records: {len(data_panel):,}")
    print(f"Non-Zero records: {len(data_panel[non_zero]):,}")
    print(f"Zero records: {len(data_panel[zero]):,}")

    validate_panel(data_panel)

    # Save
    # print("Saving data_panel.parquet ...")
    # data_panel.to_parquet("data/data_panel.parquet", index=False)
    print("Saving data_panel.csv ...")
    data_panel.to_csv("data/data_panel.csv", index=False)
    # data_panel.to_file("data/data_panel.gpkg", layer="data_panel", driver="GPKG")


if __name__ == "__main__":
    main()
