from pathlib import Path
import shutil

import numpy as np
import rasterio
from matplotlib import pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.patches import Patch
from pyproj import Geod
from rasterio.enums import Resampling
from rasterio.transform import array_bounds
from rasterio.warp import calculate_default_transform, reproject


DEM_PATH = Path(
    r"D:\The_worker\Non_work\PhD_related\sansar_phD\Assignment\Work\Download_DEM"
    r"\rasters_COP30\output_hh.tif"
)
OUT_DIR = Path(
    r"D:\The_worker\Non_work\PhD_related\sansar_phD\Assignment\Work\Results\FRom_dem\SLope"
    r"\try01"
)

TARGET_CRS = "EPSG:32644"  # WGS 84 / UTM zone 44N, suitable for Joshimath, Uttarakhand.
SLOPE_TIF = OUT_DIR / "joshimath_slope_degrees_utm44n.tif"
MAP_PNG = OUT_DIR / "joshimath_slope_map_utm44n.png"
SCRIPT_COPY = OUT_DIR / "create_joshimath_slope_map.py"


def read_and_reproject_dem():
    with rasterio.open(DEM_PATH) as src:
        dst_transform, dst_width, dst_height = calculate_default_transform(
            src.crs,
            TARGET_CRS,
            src.width,
            src.height,
            *src.bounds,
            resolution=30,
        )

        dem = np.full((dst_height, dst_width), np.nan, dtype="float32")
        reproject(
            source=rasterio.band(src, 1),
            destination=dem,
            src_transform=src.transform,
            src_crs=src.crs,
            src_nodata=src.nodata,
            dst_transform=dst_transform,
            dst_crs=TARGET_CRS,
            dst_nodata=np.nan,
            resampling=Resampling.bilinear,
        )

    return dem, dst_transform


def compute_slope_degrees(dem, transform):
    x_size = abs(transform.a)
    y_size = abs(transform.e)

    valid = np.isfinite(dem)
    filled = dem.copy()
    if np.any(valid):
        filled[~valid] = np.nanmedian(filled[valid])

    dz_dy, dz_dx = np.gradient(filled, y_size, x_size)
    slope = np.degrees(np.arctan(np.hypot(dz_dx, dz_dy))).astype("float32")
    slope[~valid] = np.nan
    return slope


def compute_hillshade(dem, transform, azimuth=315, altitude=45):
    x_size = abs(transform.a)
    y_size = abs(transform.e)

    valid = np.isfinite(dem)
    filled = dem.copy()
    if np.any(valid):
        filled[~valid] = np.nanmedian(filled[valid])

    dy, dx = np.gradient(filled, y_size, x_size)
    slope = np.pi / 2.0 - np.arctan(np.hypot(dx, dy))
    aspect = np.arctan2(-dx, dy)

    azimuth_rad = np.radians(360.0 - azimuth + 90.0)
    altitude_rad = np.radians(altitude)
    shaded = (
        np.sin(altitude_rad) * np.sin(slope)
        + np.cos(altitude_rad) * np.cos(slope) * np.cos(azimuth_rad - aspect)
    )
    hillshade = 255.0 * (shaded + 1.0) / 2.0
    hillshade[~valid] = np.nan
    return hillshade


def write_slope_geotiff(slope, transform):
    profile = {
        "driver": "GTiff",
        "height": slope.shape[0],
        "width": slope.shape[1],
        "count": 1,
        "dtype": "float32",
        "crs": TARGET_CRS,
        "transform": transform,
        "nodata": -9999.0,
        "compress": "deflate",
        "predictor": 2,
    }

    out = np.where(np.isfinite(slope), slope, profile["nodata"]).astype("float32")
    with rasterio.open(SLOPE_TIF, "w", **profile) as dst:
        dst.write(out, 1)
        dst.set_band_description(1, "Slope in degrees")


def reproject_array_to_wgs84(array, transform, resampling=Resampling.bilinear):
    height, width = array.shape
    left, bottom, right, top = array_bounds(height, width, transform)
    dst_transform, dst_width, dst_height = calculate_default_transform(
        TARGET_CRS, "EPSG:4326", width, height, left, bottom, right, top
    )

    dst = np.full((dst_height, dst_width), np.nan, dtype="float32")
    reproject(
        source=array.astype("float32"),
        destination=dst,
        src_transform=transform,
        src_crs=TARGET_CRS,
        src_nodata=np.nan,
        dst_transform=dst_transform,
        dst_crs="EPSG:4326",
        dst_nodata=np.nan,
        resampling=resampling,
    )
    return dst, dst_transform


def add_north_arrow(ax):
    ax.annotate(
        "N",
        xy=(0.95, 0.89),
        xytext=(0.95, 0.76),
        xycoords="axes fraction",
        ha="center",
        va="center",
        fontsize=16,
        fontweight="bold",
        arrowprops=dict(facecolor="black", edgecolor="black", width=4, headwidth=14),
    )


def nice_scale_length(width_km):
    candidates = np.array([1, 2, 5, 10, 20, 50], dtype=float)
    return candidates[np.argmin(np.abs(candidates - width_km / 5.0))]


def add_scale_bar(ax, extent):
    lon_min, lon_max, lat_min, lat_max = extent
    width_deg = lon_max - lon_min
    height_deg = lat_max - lat_min
    geod = Geod(ellps="WGS84")
    _, _, width_m = geod.inv(lon_min, lat_min, lon_max, lat_min)
    length_km = nice_scale_length(width_m / 1000.0)
    x0 = lon_min + 0.07 * width_deg
    y0 = lat_min + 0.07 * height_deg
    lon1, lat1, _ = geod.fwd(x0, y0, 90, length_km * 1000.0)

    ax.plot([x0, lon1], [y0, lat1], color="black", lw=4, solid_capstyle="butt", zorder=8)
    tick_h = 0.006 * height_deg
    ax.plot([x0, x0], [y0 - tick_h, y0 + tick_h], color="black", lw=2, zorder=8)
    ax.plot([lon1, lon1], [lat1 - tick_h, lat1 + tick_h], color="black", lw=2, zorder=8)
    ax.text(
        (x0 + lon1) / 2,
        y0 + 0.018 * height_deg,
        f"{length_km:g} km",
        ha="center",
        va="bottom",
        fontsize=10,
        fontweight="bold",
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.75, pad=2),
    )


def add_joshimath_label(ax, extent):
    x, y = 79.566, 30.555
    xmin, xmax, ymin, ymax = extent
    if xmin <= x <= xmax and ymin <= y <= ymax:
        ax.scatter([x], [y], s=42, c="black", edgecolors="white", linewidths=1.0, zorder=9)
        ax.text(
            x + 0.01,
            y + 0.01,
            "Joshimath",
            fontsize=10,
            fontweight="bold",
            color="black",
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.78, pad=2),
            zorder=6,
        )


def add_decimal_graticule(ax, extent):
    lon_min, lon_max, lat_min, lat_max = extent
    lon_ticks = np.round(np.arange(np.ceil(lon_min * 20) / 20, lon_max, 0.05), 3)
    lat_ticks = np.round(np.arange(np.ceil(lat_min * 20) / 20, lat_max, 0.05), 3)
    ax.set_xticks(lon_ticks)
    ax.set_yticks(lat_ticks)
    ax.set_xticklabels([f"({x:.3f})" for x in lon_ticks], fontsize=9)
    ax.set_yticklabels([f"({y:.3f})" for y in lat_ticks], fontsize=9)
    ax.grid(color="white", linewidth=0.55, alpha=0.62)
    ax.tick_params(top=True, right=True, labeltop=False, labelright=False, length=4)


def make_map(slope, dem, transform):
    hillshade = compute_hillshade(dem, transform)
    slope_map, map_transform = reproject_array_to_wgs84(slope, transform, Resampling.nearest)
    hillshade_map, _ = reproject_array_to_wgs84(hillshade, transform, Resampling.bilinear)
    lon_min, lat_min, lon_max, lat_max = array_bounds(
        slope_map.shape[0], slope_map.shape[1], map_transform
    )
    extent = (lon_min, lon_max, lat_min, lat_max)

    class_bounds = [0, 5, 15, 25, 35, 45, 90]
    colors = ["#1a9850", "#91cf60", "#fee08b", "#fc8d59", "#d73027", "#762a83"]
    labels = ["0-5 deg", "5-15 deg", "15-25 deg", "25-35 deg", "35-45 deg", ">45 deg"]
    cmap = ListedColormap(colors)
    norm = BoundaryNorm(class_bounds, cmap.N)

    fig = plt.figure(figsize=(11.6, 7.8), dpi=300)
    ax = fig.add_axes([0.065, 0.055, 0.91, 0.875])

    ax.imshow(hillshade_map, cmap="gray", extent=extent, origin="upper", alpha=0.42)
    ax.imshow(
        slope_map,
        cmap=cmap,
        norm=norm,
        extent=extent,
        origin="upper",
        alpha=0.82,
        interpolation="nearest",
    )

    add_joshimath_label(ax, extent)
    add_north_arrow(ax)
    add_scale_bar(ax, extent)

    ax.set_title(
        "Slope Map of Joshimath Region, Uttarakhand",
        fontsize=15,
        fontweight="bold",
        pad=6,
    )
    ax.set_xlabel("")
    ax.set_ylabel("")
    add_decimal_graticule(ax, extent)

    patches = [Patch(facecolor=c, edgecolor="0.25", label=l) for c, l in zip(colors, labels)]
    legend = ax.legend(
        handles=patches,
        title="Slope classes",
        loc="lower right",
        frameon=True,
        fontsize=10,
        title_fontsize=11,
        borderpad=0.9,
        labelspacing=0.75,
        bbox_to_anchor=(0.985, 0.09),
    )
    legend.get_frame().set_facecolor("white")
    legend.get_frame().set_edgecolor("0.35")
    legend.get_frame().set_alpha(0.84)

    fig.savefig(MAP_PNG, bbox_inches="tight", pad_inches=0.02, facecolor="white")
    plt.close(fig)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    dem, transform = read_and_reproject_dem()
    slope = compute_slope_degrees(dem, transform)
    write_slope_geotiff(slope, transform)
    make_map(slope, dem, transform)
    shutil.copy2(Path(__file__), SCRIPT_COPY)

    print(f"Wrote slope GeoTIFF: {SLOPE_TIF}")
    print(f"Wrote slope map PNG: {MAP_PNG}")
    print(f"Saved script copy: {SCRIPT_COPY}")


if __name__ == "__main__":
    main()
