from pathlib import Path
import shutil

import numpy as np
import rasterio
from matplotlib import pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.patches import Patch
from pyproj import Geod, Transformer
from rasterio.enums import Resampling
from rasterio.transform import array_bounds
from rasterio.warp import calculate_default_transform, reproject


NDVI_PATH = Path(
    r"D:\The_worker\Non_work\PhD_related\sansar_phD\Assignment\Work\Senteine02\NDVI"
    r"\2024-10-31-00_00_2025-04-29-23_59_Sentinel-2_L2A_NDVI.tiff"
)
OUT_DIR = Path(
    r"D:\The_worker\Non_work\PhD_related\sansar_phD\Assignment\Work\Results\FROM_s2"
)

NDVI_TIF = OUT_DIR / "joshimath_ndvi.tif"
MAP_PNG = OUT_DIR / "joshimath_ndvi_map.png"
MAP_JPG = OUT_DIR / "joshimath_ndvi_map.jpg"
MAP_PDF = OUT_DIR / "joshimath_ndvi_map.pdf"
SCRIPT_COPY = OUT_DIR / "create_joshimath_ndvi_map.py"


def read_ndvi():
    with rasterio.open(NDVI_PATH) as src:
        ndvi = src.read(1).astype("float32")
        nodata = src.nodata
        if nodata is not None:
            ndvi = np.where(ndvi == nodata, np.nan, ndvi)
        ndvi = np.where(np.isfinite(ndvi), ndvi, np.nan)

        # Some providers store NDVI as scaled integers; normalize when needed.
        finite = ndvi[np.isfinite(ndvi)]
        if finite.size and (np.nanmax(finite) > 2.0 or np.nanmin(finite) < -2.0):
            scale = 10000.0 if np.nanmax(np.abs(finite)) > 100 else 100.0
            ndvi = ndvi / scale
        ndvi = np.clip(ndvi, -1.0, 1.0)
        return ndvi, src.transform, src.crs, src.profile


def write_ndvi_geotiff(ndvi, transform, crs, src_profile):
    profile = src_profile.copy()
    profile.update(
        driver="GTiff",
        dtype="float32",
        count=1,
        nodata=-9999.0,
        compress="deflate",
        predictor=2,
        transform=transform,
        crs=crs,
    )
    out = np.where(np.isfinite(ndvi), ndvi, -9999.0).astype("float32")
    with rasterio.open(NDVI_TIF, "w", **profile) as dst:
        dst.write(out, 1)
        dst.set_band_description(1, "Normalized Difference Vegetation Index")


def reproject_array_to_wgs84(array, transform, src_crs, resampling=Resampling.bilinear):
    if str(src_crs).upper() in ("EPSG:4326", "OGC:CRS84"):
        return array, transform
    height, width = array.shape
    left, bottom, right, top = array_bounds(height, width, transform)
    dst_transform, dst_width, dst_height = calculate_default_transform(
        src_crs, "EPSG:4326", width, height, left, bottom, right, top
    )
    dst = np.full((dst_height, dst_width), np.nan, dtype="float32")
    reproject(
        source=array.astype("float32"),
        destination=dst,
        src_transform=transform,
        src_crs=src_crs,
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
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.78, pad=2),
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
            zorder=9,
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


def make_map(ndvi, transform, crs):
    ndvi_map, map_transform = reproject_array_to_wgs84(ndvi, transform, crs, Resampling.bilinear)
    lon_min, lat_min, lon_max, lat_max = array_bounds(
        ndvi_map.shape[0], ndvi_map.shape[1], map_transform
    )
    extent = (lon_min, lon_max, lat_min, lat_max)

    class_bounds = [-1.0, 0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    colors = ["#8c510a", "#d9a441", "#f6e8a4", "#b8e186", "#4dac26", "#006837"]
    labels = [
        "Water/Bare (<0.0)",
        "Very low (0.0-0.2)",
        "Low (0.2-0.4)",
        "Moderate (0.4-0.6)",
        "High (0.6-0.8)",
        "Very high (>0.8)",
    ]
    cmap = ListedColormap(colors)
    norm = BoundaryNorm(class_bounds, cmap.N)

    fig = plt.figure(figsize=(11.6, 7.8), dpi=300)
    ax = fig.add_axes([0.065, 0.055, 0.91, 0.875])
    ax.imshow(
        ndvi_map,
        cmap=cmap,
        norm=norm,
        extent=extent,
        origin="upper",
        interpolation="nearest",
    )

    add_joshimath_label(ax, extent)
    add_north_arrow(ax)
    add_scale_bar(ax, extent)

    ax.set_title(
        "NDVI Map of Joshimath Region, Uttarakhand",
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
        title="NDVI classes",
        loc="lower right",
        frameon=True,
        fontsize=9,
        title_fontsize=11,
        borderpad=0.9,
        labelspacing=0.60,
        bbox_to_anchor=(0.985, 0.09),
    )
    legend.get_frame().set_facecolor("white")
    legend.get_frame().set_edgecolor("0.35")
    legend.get_frame().set_alpha(0.84)

    for path, kwargs in [
        (MAP_PNG, {}),
        (MAP_JPG, {"pil_kwargs": {"quality": 95}}),
        (MAP_PDF, {}),
    ]:
        fig.savefig(path, bbox_inches="tight", pad_inches=0.02, facecolor="white", **kwargs)
    plt.close(fig)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ndvi, transform, crs, profile = read_ndvi()
    write_ndvi_geotiff(ndvi, transform, crs, profile)
    make_map(ndvi, transform, crs)
    shutil.copy2(Path(__file__), SCRIPT_COPY)

    finite = ndvi[np.isfinite(ndvi)]
    print(f"Input NDVI: {NDVI_PATH}")
    print(f"CRS: {crs}")
    print(f"NDVI min/max/mean: {float(np.nanmin(finite)):.3f}, {float(np.nanmax(finite)):.3f}, {float(np.nanmean(finite)):.3f}")
    print(f"Wrote NDVI GeoTIFF: {NDVI_TIF}")
    print(f"Wrote NDVI map PNG: {MAP_PNG}")
    print(f"Wrote NDVI map JPG: {MAP_JPG}")
    print(f"Wrote NDVI map PDF: {MAP_PDF}")
    print(f"Saved script copy: {SCRIPT_COPY}")


if __name__ == "__main__":
    main()
