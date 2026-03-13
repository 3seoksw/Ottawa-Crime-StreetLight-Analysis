import pandas as pd
import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt
import contextily as ctx

period_config = {
    "monthly": "M",
    "quarterly": "Q",
}


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


def get_period_freq(record_frequency: str) -> str:
    try:
        return period_config[record_frequency]
    except KeyError as exc:
        raise ValueError(
            f"Unsupported record frequency: {record_frequency}. "
            f"Expected one of {sorted(period_config)}."
        ) from exc


def add_crime_group(gdf: gpd.GeoDataFrame):
    gdf["crime_group"] = np.where(
        gdf["is_nighttime"] == 1,
        1,  # "night",
        0,  # "non_night",
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


def assign_crimes_to_grid(
    crime_gdf: gpd.GeoDataFrame,
    grids: gpd.GeoDataFrame,
    record_frequency: str = "monthly",
):
    period_freq = get_period_freq(record_frequency)
    grid_gdf = gpd.GeoDataFrame(
        grids[["cell_id", "geometry"]], geometry="geometry", crs=grids.crs
    )
    joined = gpd.sjoin(
        crime_gdf,
        grid_gdf,
        predicate="within",
        how="inner",
    )
    joined["time_period"] = joined["Occurred Date"].dt.to_period(period_freq)

    aggregated = joined.groupby(
        ["cell_id", "time_period", "crime_group"], observed=True
    ).size()
    aggregated = aggregated.to_frame("crime_count").reset_index()
    aggregated["crime_group"] = aggregated["crime_group"].astype(int)
    crime_groups = [0, 1]

    cells = grids["cell_id"].unique()
    periods = pd.period_range(
        crime_gdf["Occurred Date"].min().to_period(period_freq),
        crime_gdf["Occurred Date"].max().to_period(period_freq),
        freq=period_freq,
    )

    full_index = pd.MultiIndex.from_product(
        [cells, periods, crime_groups],
        names=["cell_id", "time_period", "crime_group"],
    )

    panel = pd.DataFrame(index=full_index).reset_index()
    panel = panel.merge(
        aggregated, on=["cell_id", "time_period", "crime_group"], how="left"
    )
    panel["crime_count"] = panel["crime_count"].fillna(0).astype(int)

    # panel = panel.merge(grids[["cell_id"]], on="cell_id", how="left")

    return panel


def add_crime_features(df: pd.DataFrame):
    df = df.sort_values(["cell_id", "crime_group", "time_period"]).copy()
    g = df.groupby(["cell_id", "crime_group"])["crime_count"]

    shifted = g.shift(1).fillna(0)
    df["cumulative_crime_count"] = shifted.groupby(
        [df["cell_id"], df["crime_group"]]
    ).cumsum()

    avg_crime_count = g.expanding().mean()
    df["avg_crime_count"] = (
        avg_crime_count.groupby(level=[0, 1])
        .shift(1)
        .reset_index(level=[0, 1], drop=True)
        .fillna(0)
    )
    df["prev_crime_count"] = shifted
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
