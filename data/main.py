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
            cell_ids.append(i * (len(y_range) - 1) + j)

    grids = gpd.GeoDataFrame({"cell_id": cell_ids, "geometry": cells}, crs=gdf.crs)
    grids["centroid_x"] = grids.geometry.centroid.x
    grids["centroid_y"] = grids.geometry.centroid.y
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


def main():
    panel, grids = preprocess_crime_data()
    light = preprocess_street_light_data()
    print("============ AGGREGATION ============ ")
    lights_cell = assign_street_lights_to_grid(light, grids)

    data_panel = panel.merge(lights_cell, on="cell_id", how="left")
    light_feats = ["light_count", "total_wattage", "total_intensity", "avg_wattage"]
    data_panel[light_feats] = data_panel[light_feats].fillna(0)

    # Final Aggregation (coordinates included)
    grid_info = ["cell_id", "geometry", "centroid_x", "centroid_y"]
    data_panel = data_panel.merge(
        grids[grid_info],
        on="cell_id",
        how="left",
    )
    data_panel = gpd.GeoDataFrame(data_panel, geometry="geometry", crs=grids.crs)

    # Save
    data_panel.to_parquet("data/data_panel.parquet", index=False)
    data_panel.to_file("data/data_panel.gpkg", layer="data_panel", driver="GPKG")


if __name__ == "__main__":
    main()
