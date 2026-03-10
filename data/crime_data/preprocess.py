import pandas as pd
import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt
import contextily as ctx


def load_crime_data(
    filename: str = "data/crime_data/Criminal_Offences_Open_Data_-621494644292511792.csv",
) -> gpd.GeoDataFrame:
    df = pd.read_csv(filename)
    prev_len = len(df)

    # Drop records with missing coordinates
    df["x"] = pd.to_numeric(df["x"], errors="coerce")
    df["y"] = pd.to_numeric(df["y"], errors="coerce")
    df = df.dropna(subset=["x", "y"])

    # Drop old records, maintain records from 2018 to 2024
    df["Occurred Date"] = pd.to_datetime(df["Occurred Date"], errors="coerce")
    df["year"] = df["Occurred Date"].dt.year
    df = pd.DataFrame(df[df["year"].between(2018, 2024)])

    # Source coordinates are projected Ottawa-area meters (EPSG:2951).
    gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.x, df.y), crs="EPSG:2951")
    post_len = len(gdf)
    n_dropped = prev_len - post_len
    print(f"Records dropped: {n_dropped:,} due to missing coordinates")
    return gdf


def get_is_nighttime(gdf: gpd.GeoDataFrame) -> pd.Series:
    """
    Returns `GeoDataFrame` with nighttime indicator;
        1: nighttime
        0: not a nighttime
        -1: NaN "Occurred Hour"
    """
    sunset_hours_by_month = {
        1: 17,
        2: 17,
        3: 18,
        4: 20,
        5: 20,
        6: 21,
        7: 21,
        8: 20,
        9: 20,
        10: 19,
        11: 17,
        12: 16,
    }
    hours = gdf["Occurred Hour"] // 100
    sunset_hours = pd.Series(gdf["month"]).map(sunset_hours_by_month)

    is_after_sunset = hours >= sunset_hours
    is_before_dawn = hours < 6
    is_nighttime = (is_after_sunset | is_before_dawn).astype(int)
    return is_nighttime.where(hours.notna(), other=-1)


def parse_crime_dates(gdf: gpd.GeoDataFrame):
    gdf = gdf.copy()

    gdf["Occurred Date"] = pd.to_datetime(gdf["Occurred Date"], errors="coerce")
    # gdf["year"] = gdf["Year"].astype(int)
    gdf["month"] = gdf["Occurred Date"].dt.month
    gdf["day"] = gdf["Occurred Date"].dt.day
    gdf["year_month"] = gdf["Occurred Date"].dt.to_period("M")

    # Consider sunset hours per month to determine whether it's night
    gdf["is_nighttime"] = get_is_nighttime(gdf)

    # cols = ["is_nighttime", "Occurred Hour", "Reported Hour"]
    return gdf


def add_crime_group(gdf: gpd.GeoDataFrame):
    gdf["crime_group"] = np.where(
        gdf["is_nighttime"] == 1,
        "night",
        "non_night",
    )
    return gdf


def filter_areas(gdf: gpd.GeoDataFrame):
    min_x, min_y, max_x, max_y = (
        355000,
        5020000,  # southwest
        372000,
        5034000,  # northeast
    )

    mask = (
        (gdf.geometry.x >= min_x)
        & (gdf.geometry.x <= max_x)
        & (gdf.geometry.y >= min_y)
        & (gdf.geometry.y <= max_y)
    )

    return gpd.GeoDataFrame(gdf[mask].copy())


def assign_crimes_to_grid(crime_gdf: gpd.GeoDataFrame, grids: gpd.GeoDataFrame):
    grid_gdf = gpd.GeoDataFrame(
        grids[["cell_id", "geometry"]], geometry="geometry", crs=grids.crs
    )
    joined = gpd.sjoin(
        crime_gdf,
        grid_gdf,
        predicate="within",
        how="inner",
    )

    aggregated = joined.groupby(
        ["cell_id", "year_month", "crime_group"], observed=True
    ).size()
    aggregated = aggregated.to_frame("crime_count").reset_index()

    cells = grids["cell_id"].unique()
    months = pd.period_range(
        crime_gdf["year_month"].min(),
        crime_gdf["year_month"].max(),
        freq="M",
    )
    crime_groups = ["night", "non_night"]
    neighbourhood = crime_gdf["Neighbourhood Name"].unique()

    # Full skeleton
    full_index = pd.MultiIndex.from_product(
        [cells, months, crime_groups, neighbourhood],
        names=["cell_id", "year_month", "crime_group", "neighbourhood"],
    )
    panel = pd.DataFrame(index=full_index).reset_index()
    panel = panel.merge(
        aggregated,
        on=["cell_id", "year_month", "crime_group"],
        how="left",
    )
    panel["crime_count"] = panel["crime_count"].fillna(0).astype(int)
    panel = panel.sort_values(["cell_id", "crime_group", "year_month"])

    return panel


def add_crime_features(df: pd.DataFrame):
    # Accumulated crime count
    df["total_crime_count"] = (
        df.groupby("cell_id")["crime_count"].cumsum().shift(1).fillna(0)
    )

    # Average crime count
    df["avg_crime_count"] = (
        df.groupby("cell_id")["crime_count"]
        .expanding()
        .mean()
        .shift(1)
        .reset_index(level=0, drop=True)
        .fillna(0)
    )

    # Previous crime count
    df["prev_crime_count"] = df.groupby("cell_id")["crime_count"].shift(1).fillna(0)
    return df


def plot_crime_hexbin(crime_gdf, is_filtered=False):
    crime_plot = crime_gdf.to_crs(epsg=3857)

    x = crime_plot.geometry.x
    y = crime_plot.geometry.y

    fig, ax = plt.subplots(figsize=(10, 10))

    hb = ax.hexbin(
        x,
        y,
        gridsize=60,
        cmap="inferno",
        mincnt=1,
        bins="log",
    )

    ctx.add_basemap(ax, source=ctx.providers.OpenStreetMap.Mapnik, crs=crime_plot.crs)

    ax.set_aspect("equal")
    ax.set_title("Crime Density (Hexbin)")
    ax.axis("off")

    cb = fig.colorbar(hb, ax=ax)
    cb.set_label("Crime Count")

    plt.tight_layout()
    if is_filtered:
        plt.savefig("test.png", dpi=300)
    else:
        plt.savefig("og.png", dpi=300)
    plt.show()


def print_crime_ratios(panel: pd.DataFrame):
    non_zeros = panel[panel["crime_count"] > 0]
    non_zero_count = len(non_zeros)
    zeros = panel[panel["crime_count"] == 0]
    zero_count = len(zeros)
    print(f"Non-zero: {non_zero_count:,}")
    print(f"Zero: {zero_count:,}")
    print(
        f"Non-zero ratio: {non_zero_count / (non_zero_count + zero_count) * 100:.2f}%"
    )
