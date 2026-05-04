from collections import deque
from pathlib import Path
import heapq
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
    r"D:\The_worker\Non_work\PhD_related\sansar_phD\Assignment\Work\Results\FRom_dem\TWI"
)

TARGET_CRS = "EPSG:32644"
TWI_TIF = OUT_DIR / "joshimath_twi_utm44n.tif"
MAP_PNG = OUT_DIR / "joshimath_twi_map.png"
MAP_JPG = OUT_DIR / "joshimath_twi_map.jpg"
MAP_PDF = OUT_DIR / "joshimath_twi_map.pdf"
SCRIPT_COPY = OUT_DIR / "create_joshimath_twi_map.py"

NEIGHBORS = [
    (-1, -1, np.sqrt(2.0)),
    (-1, 0, 1.0),
    (-1, 1, np.sqrt(2.0)),
    (0, -1, 1.0),
    (0, 1, 1.0),
    (1, -1, np.sqrt(2.0)),
    (1, 0, 1.0),
    (1, 1, np.sqrt(2.0)),
]


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


def fill_nodata(array):
    valid = np.isfinite(array)
    filled = array.copy()
    if np.any(valid):
        filled[~valid] = np.nanmedian(filled[valid])
    return filled, valid


def priority_flood_fill(dem):
    rows, cols = dem.shape
    valid = np.isfinite(dem)
    filled = dem.copy()
    visited = np.zeros(dem.shape, dtype=bool)
    heap = []

    def push(r, c):
        visited[r, c] = True
        heapq.heappush(heap, (float(filled[r, c]), r, c))

    for r in range(rows):
        for c in (0, cols - 1):
            if valid[r, c] and not visited[r, c]:
                push(r, c)
    for c in range(cols):
        for r in (0, rows - 1):
            if valid[r, c] and not visited[r, c]:
                push(r, c)

    while heap:
        elev, r, c = heapq.heappop(heap)
        for dr, dc, _ in NEIGHBORS:
            nr, nc = r + dr, c + dc
            if nr < 0 or nr >= rows or nc < 0 or nc >= cols:
                continue
            if visited[nr, nc] or not valid[nr, nc]:
                continue
            visited[nr, nc] = True
            if filled[nr, nc] < elev:
                filled[nr, nc] = elev
            heapq.heappush(heap, (float(filled[nr, nc]), nr, nc))

    filled[~valid] = np.nan
    return filled


def compute_slope_radians(dem, transform):
    x_size = abs(transform.a)
    y_size = abs(transform.e)
    filled, valid = fill_nodata(dem)
    dz_dy, dz_dx = np.gradient(filled, y_size, x_size)
    slope = np.arctan(np.hypot(dz_dx, dz_dy)).astype("float32")
    slope[~valid] = np.nan
    return slope


def compute_d8_receivers(filled_dem, transform):
    rows, cols = filled_dem.shape
    valid = np.isfinite(filled_dem)
    cell = abs(transform.a)
    receiver = np.full(rows * cols, -1, dtype=np.int64)
    indegree = np.zeros(rows * cols, dtype=np.int32)

    for r in range(rows):
        for c in range(cols):
            if not valid[r, c]:
                continue
            best_drop = 0.0
            best_idx = -1
            z = filled_dem[r, c]
            for dr, dc, dist_mult in NEIGHBORS:
                nr, nc = r + dr, c + dc
                if nr < 0 or nr >= rows or nc < 0 or nc >= cols or not valid[nr, nc]:
                    continue
                drop = (z - filled_dem[nr, nc]) / (cell * dist_mult)
                if drop > best_drop:
                    best_drop = drop
                    best_idx = nr * cols + nc
            idx = r * cols + c
            receiver[idx] = best_idx
            if best_idx >= 0:
                indegree[best_idx] += 1

    return receiver, indegree, valid


def compute_flow_accumulation(receiver, indegree, valid):
    rows, cols = valid.shape
    flat_valid = valid.ravel()
    acc = np.zeros(rows * cols, dtype="float64")
    acc[flat_valid] = 1.0

    queue = deque(np.flatnonzero(flat_valid & (indegree == 0)))
    while queue:
        idx = queue.popleft()
        rec = receiver[idx]
        if rec >= 0:
            acc[rec] += acc[idx]
            indegree[rec] -= 1
            if indegree[rec] == 0:
                queue.append(rec)

    # Flat unresolved cells can remain in cycles after depression filling; keep their own cell area.
    return acc.reshape(rows, cols)


def compute_twi(dem, transform):
    filled_dem = priority_flood_fill(dem)
    slope = compute_slope_radians(filled_dem, transform)
    receiver, indegree, valid = compute_d8_receivers(filled_dem, transform)
    accumulation = compute_flow_accumulation(receiver, indegree, valid)

    cell = abs(transform.a)
    specific_catchment_area = accumulation * cell
    tan_slope = np.tan(slope)
    tan_slope = np.where(tan_slope < 0.001, 0.001, tan_slope)
    twi = np.log(specific_catchment_area / tan_slope).astype("float32")
    twi[~valid] = np.nan
    return twi


def compute_hillshade(dem, transform, azimuth=315, altitude=45):
    x_size = abs(transform.a)
    y_size = abs(transform.e)
    filled, valid = fill_nodata(dem)
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
    return hillshade.astype("float32")


def write_twi_geotiff(twi, transform):
    profile = {
        "driver": "GTiff",
        "height": twi.shape[0],
        "width": twi.shape[1],
        "count": 1,
        "dtype": "float32",
        "crs": TARGET_CRS,
        "transform": transform,
        "nodata": -9999.0,
        "compress": "deflate",
        "predictor": 2,
    }
    out = np.where(np.isfinite(twi), twi, profile["nodata"]).astype("float32")
    with rasterio.open(TWI_TIF, "w", **profile) as dst:
        dst.write(out, 1)
        dst.set_band_description(1, "Topographic Wetness Index")


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


def classify_twi(twi_map):
    valid = twi_map[np.isfinite(twi_map)]
    edges = np.percentile(valid, [2, 20, 40, 60, 80, 95, 99])
    edges = np.unique(np.round(edges, 2))
    if len(edges) < 7:
        edges = np.linspace(float(np.nanmin(valid)), float(np.nanmax(valid)), 7)
    labels = [
        f"{edges[i]:.1f}-{edges[i + 1]:.1f}" for i in range(len(edges) - 1)
    ]
    return edges, labels


def make_map(twi, dem, transform):
    hillshade = compute_hillshade(dem, transform)
    twi_map, map_transform = reproject_array_to_wgs84(twi, transform, Resampling.bilinear)
    hillshade_map, _ = reproject_array_to_wgs84(hillshade, transform, Resampling.bilinear)
    lon_min, lat_min, lon_max, lat_max = array_bounds(
        twi_map.shape[0], twi_map.shape[1], map_transform
    )
    extent = (lon_min, lon_max, lat_min, lat_max)

    class_bounds, labels = classify_twi(twi_map)
    colors = ["#8c510a", "#d8b365", "#f6e8c3", "#c7eae5", "#5ab4ac", "#01665e"]
    cmap = ListedColormap(colors[: len(labels)])
    norm = BoundaryNorm(class_bounds, cmap.N)

    fig = plt.figure(figsize=(11.6, 7.8), dpi=300)
    ax = fig.add_axes([0.065, 0.055, 0.91, 0.875])
    ax.imshow(hillshade_map, cmap="gray", extent=extent, origin="upper", alpha=0.38)
    ax.imshow(
        twi_map,
        cmap=cmap,
        norm=norm,
        extent=extent,
        origin="upper",
        alpha=0.86,
        interpolation="nearest",
    )

    add_joshimath_label(ax, extent)
    add_north_arrow(ax)
    add_scale_bar(ax, extent)

    ax.set_title(
        "Topographic Wetness Index Map of Joshimath Region, Uttarakhand",
        fontsize=14,
        fontweight="bold",
        pad=6,
    )
    ax.set_xlabel("")
    ax.set_ylabel("")
    add_decimal_graticule(ax, extent)

    patches = [
        Patch(facecolor=colors[i], edgecolor="0.25", label=labels[i])
        for i in range(len(labels))
    ]
    legend = ax.legend(
        handles=patches,
        title="TWI classes",
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
    dem, transform = read_and_reproject_dem()
    twi = compute_twi(dem, transform)
    write_twi_geotiff(twi, transform)
    make_map(twi, dem, transform)
    shutil.copy2(Path(__file__), SCRIPT_COPY)

    print(f"Wrote TWI GeoTIFF: {TWI_TIF}")
    print(f"Wrote TWI map PNG: {MAP_PNG}")
    print(f"Wrote TWI map JPG: {MAP_JPG}")
    print(f"Wrote TWI map PDF: {MAP_PDF}")
    print(f"Saved script copy: {SCRIPT_COPY}")


if __name__ == "__main__":
    main()
