import pandas as pd
import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import contextily as ctx


def remove_outliers(df: pd.DataFrame, col: str):
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    return df[(df[col] >= Q1 - 1.5 * IQR) & (df[col] <= Q3 + 1.5 * IQR)]


def load_street_light_data(filename: str = "data/street_light_data/Street_Lights.csv"):
    df = pd.read_csv(filename)
    prev_len = len(df)
    df["X"] = pd.to_numeric(df["X"], errors="coerce")
    df["Y"] = pd.to_numeric(df["Y"], errors="coerce")
    df = df.dropna(subset=["X", "Y"])

    df = pd.DataFrame(remove_outliers(df, "X"))
    df = pd.DataFrame(remove_outliers(df, "Y"))

    # Install dates
    df["INSTALL_LIGHT"] = pd.to_datetime(df["INSTALL_LIGHT"], errors="coerce")
    invalid_date = (df["INSTALL_LIGHT"].dt.year < 1900) | (
        df["INSTALL_LIGHT"].dt.year >= 2025
    )
    df.loc[invalid_date, "INSTALL_LIGHT"] = pd.NaT
    df["install_year_month"] = df["INSTALL_LIGHT"].dt.to_period("M")

    max_install = df["install_year_month"].max()
    df["install_year_month"] = (max_install - df["install_year_month"]).apply(
        lambda x: x.n if pd.notna(x) else -1
    )

    # Intensity Light information
    df["intensity"] = df["WATTAGE"] * df["LIGHTS_NUM"]

    gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.X, df.Y), crs="EPSG:3857")
    post_len = len(gdf)
    n_dropped = prev_len - post_len
    print(f"Records dropped: {n_dropped:,} due to missing coordinates")

    return gdf


def assign_street_lights_to_grid(gdf: gpd.GeoDataFrame, grids: gpd.GeoDataFrame):
    if gdf.crs is None or grids.crs is None:
        raise ValueError("Both street light and grid GeoDataFrames must have a CRS.")
    if gdf.crs != grids.crs:
        gdf = gdf.to_crs(grids.crs)

    grid_gdf = gpd.GeoDataFrame(
        grids[["cell_id", "geometry"]], geometry="geometry", crs=grids.crs
    )
    joined = gpd.sjoin(gdf, grid_gdf, predicate="within", how="inner")

    aggregated = (
        joined.groupby("cell_id")
        .agg(
            light_count=("LIGHTS_NUM", "sum"),
            total_wattage=("WATTAGE", "sum"),
            total_intensity=("intensity", "sum"),
            avg_wattage=("WATTAGE", "mean"),
            avg_install_month=(
                "install_year_month",
                lambda s: s[s != -1].mean() if (s != -1).any() else -1,
            ),
        )
        .reset_index()
    )
    return aggregated


def plot_street_lights(gdf: gpd.GeoDataFrame):
    gdf = gdf.to_crs(epsg=2951)
    x = gdf.geometry.x
    y = gdf.geometry.y
    _, ax = plt.subplots(figsize=(14, 12))
    hb = ax.hexbin(
        x,
        y,
        C=gdf["intensity"],
        reduce_C_function=np.sum,
        gridsize=150,
        cmap="YlOrRd",
        norm=mcolors.LogNorm(),
        alpha=0.75,
        linewidths=0,
    )
    ctx.add_basemap(ax, crs="EPSG:3857", source=ctx.providers.OpenStreetMap.Mapnik)
    ax.set_aspect("equal")

    cbar = plt.colorbar(hb, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("Intensity", color="white", fontsize=11)
    cbar.ax.yaxis.set_tick_params(color="white")
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color="white")

    ax.set_axis_off()
    plt.title(
        "Street Light Intensity Map",
        color="white",
        fontsize=16,
        fontweight="bold",
        pad=12,
    )
    plt.tight_layout()
    plt.show()
