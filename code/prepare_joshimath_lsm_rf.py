from pathlib import Path
import json
import shutil

import joblib
import numpy as np
import pandas as pd
import rasterio
import shapefile
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from pyproj import Transformer
from rasterio.features import rasterize
from rasterio.enums import Resampling
from rasterio.transform import array_bounds, xy
from rasterio.warp import reproject
from scipy.ndimage import binary_dilation
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    auc,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedGroupKFold


WORK = Path(r"D:\The_worker\Non_work\PhD_related\sansar_phD\Assignment\Work")
OUT_DIR = WORK / "ML_work" / "Results"
STACK_DIR = OUT_DIR / "01_aligned_stack"
DATA_DIR = OUT_DIR / "02_training_testing_data"
MODEL_DIR = OUT_DIR / "03_models_metrics"
MAP_DIR = OUT_DIR / "04_susceptibility_maps"

AOI_SHP = (
    WORK
    / "Downloaded_invent"
    / "JOshimath"
    / "1_20260501161152176"
    / "degree_sheet_landslide_20260501161135235"
    / "degree_sheet_landslide_20260501161135235.shp"
)
LANDSLIDE_POINTS_SHP = (
    WORK
    / "Downloaded_invent"
    / "JOshimath"
    / "1_20260501161152176"
    / "landslide_point_20260501161135235"
    / "landslide_point_20260501161135235.shp"
)

RASTERS = {
    "elevation": WORK / "Download_DEM" / "rasters_COP30" / "output_hh.tif",
    "slope": WORK / "Results" / "FRom_dem" / "SLope" / "try01" / "joshimath_slope_degrees_utm44n.tif",
    "aspect": WORK / "Results" / "FRom_dem" / "Aspect" / "joshimath_aspect_degrees_utm44n.tif",
    "twi": WORK / "Results" / "FRom_dem" / "TWI" / "joshimath_twi_utm44n.tif",
    "ndvi": WORK / "Results" / "FROM_s2" / "joshimath_ndvi.tif",
    "insar_velocity": WORK / "InSAR" / "Joshimath" / "jOSHIMATH_velocity_ps.tif",
}

REFERENCE = RASTERS["slope"]
BASE_FEATURES = ["elevation", "slope", "aspect", "twi", "ndvi"]
INSAR_FEATURES = BASE_FEATURES + ["insar_velocity"]
RANDOM_SEED = 42
NEGATIVE_RATIO = 1
NEGATIVE_EXCLUSION_RADIUS_M = 150
SPATIAL_BLOCK_SIZE_M = 1000


def ensure_dirs():
    for path in [OUT_DIR, STACK_DIR, DATA_DIR, MODEL_DIR, MAP_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def read_shapes(path):
    reader = shapefile.Reader(str(path))
    return [shape.__geo_interface__ for shape in reader.shapes()]


def shapefile_crs(path):
    prj = Path(path).with_suffix(".prj")
    if not prj.exists():
        return "EPSG:4326"
    text = prj.read_text(errors="ignore")
    if "WGS_1984" in text or "WGS 84" in text:
        return "EPSG:4326"
    return "EPSG:4326"


def align_raster(name, path, ref_profile, ref_transform, ref_crs, ref_shape):
    out_path = STACK_DIR / f"{name}_aligned.tif"
    with rasterio.open(path) as src:
        data = np.full(ref_shape, np.nan, dtype="float32")
        src_data = src.read(1).astype("float32")
        if src.nodata is not None:
            src_data = np.where(src_data == src.nodata, np.nan, src_data)
        reproject(
            source=src_data,
            destination=data,
            src_transform=src.transform,
            src_crs=src.crs,
            src_nodata=np.nan,
            dst_transform=ref_transform,
            dst_crs=ref_crs,
            dst_nodata=np.nan,
            resampling=Resampling.bilinear,
        )

    profile = ref_profile.copy()
    profile.update(dtype="float32", count=1, nodata=-9999.0, compress="deflate", predictor=2)
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(np.where(np.isfinite(data), data, -9999.0).astype("float32"), 1)
        dst.set_band_description(1, name)
    return data


def build_aligned_stack():
    with rasterio.open(REFERENCE) as ref:
        ref_profile = ref.profile.copy()
        ref_transform = ref.transform
        ref_crs = ref.crs
        ref_shape = (ref.height, ref.width)

    stack = {}
    for name, path in RASTERS.items():
        stack[name] = align_raster(name, path, ref_profile, ref_transform, ref_crs, ref_shape)
    return stack, ref_profile, ref_transform, ref_crs, ref_shape


def rasterize_aoi(ref_shape, ref_transform, ref_crs):
    aoi_shapes = read_shapes(AOI_SHP)
    # Inventory shapefiles are WGS84; rasterio can rasterize only in target CRS, so transform polygon rings.
    src_crs = shapefile_crs(AOI_SHP)
    transformer = Transformer.from_crs(src_crs, ref_crs, always_xy=True)
    transformed = []
    for geom in aoi_shapes:
        rings = []
        for ring in geom["coordinates"]:
            rings.append([transformer.transform(x, y) for x, y in ring])
        transformed.append({"type": "Polygon", "coordinates": rings})
    mask = rasterize(
        [(geom, 1) for geom in transformed],
        out_shape=ref_shape,
        transform=ref_transform,
        fill=0,
        dtype="uint8",
    ).astype(bool)
    return mask


def landslide_pixel_mask(ref_shape, ref_transform, ref_crs):
    reader = shapefile.Reader(str(LANDSLIDE_POINTS_SHP))
    src_crs = shapefile_crs(LANDSLIDE_POINTS_SHP)
    transformer = Transformer.from_crs(src_crs, ref_crs, always_xy=True)
    mask = np.zeros(ref_shape, dtype=bool)
    rows, cols = ref_shape
    for shp in reader.shapes():
        x, y = shp.points[0]
        tx, ty = transformer.transform(x, y)
        col, row = ~ref_transform * (tx, ty)
        row, col = int(np.floor(row)), int(np.floor(col))
        if 0 <= row < rows and 0 <= col < cols:
            mask[row, col] = True
    return mask


def write_mask(mask, ref_profile, path, description):
    profile = ref_profile.copy()
    profile.update(dtype="uint8", count=1, nodata=0, compress="deflate")
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(mask.astype("uint8"), 1)
        dst.set_band_description(1, description)


def spatial_blocks(rows, cols, transform):
    rr, cc = np.indices((rows, cols))
    xs = transform.c + (cc + 0.5) * transform.a + (rr + 0.5) * transform.b
    ys = transform.f + (cc + 0.5) * transform.d + (rr + 0.5) * transform.e
    bx = np.floor(xs / SPATIAL_BLOCK_SIZE_M).astype("int64")
    by = np.floor(ys / SPATIAL_BLOCK_SIZE_M).astype("int64")
    return by * 10_000_000 + bx


def build_samples(stack, aoi_mask, landslide_mask, ref_transform):
    all_features = INSAR_FEATURES
    valid_base = aoi_mask.copy()
    valid_insar = aoi_mask.copy()
    for name in BASE_FEATURES:
        valid_base &= np.isfinite(stack[name])
    for name in INSAR_FEATURES:
        valid_insar &= np.isfinite(stack[name])

    # Fair comparison: use exactly the same samples where InSAR and base layers are both available.
    common_valid = valid_insar
    pos_mask = landslide_mask & common_valid
    radius_px = max(1, int(round(NEGATIVE_EXCLUSION_RADIUS_M / abs(ref_transform.a))))
    structure = np.ones((radius_px * 2 + 1, radius_px * 2 + 1), dtype=bool)
    exclusion = binary_dilation(landslide_mask, structure=structure)
    neg_pool = common_valid & ~exclusion

    pos_rows, pos_cols = np.where(pos_mask)
    neg_rows, neg_cols = np.where(neg_pool)
    if len(pos_rows) < 10:
        raise RuntimeError(
            f"Only {len(pos_rows)} landslide points overlap valid InSAR/common raster coverage. "
            "Need more overlap or a larger inventory area."
        )

    rng = np.random.default_rng(RANDOM_SEED)
    n_neg = min(len(neg_rows), len(pos_rows) * NEGATIVE_RATIO)
    neg_idx = rng.choice(len(neg_rows), size=n_neg, replace=False)

    sample_rows = np.concatenate([pos_rows, neg_rows[neg_idx]])
    sample_cols = np.concatenate([pos_cols, neg_cols[neg_idx]])
    y = np.concatenate([np.ones(len(pos_rows), dtype=int), np.zeros(n_neg, dtype=int)])

    blocks = spatial_blocks(*common_valid.shape, ref_transform)
    xs, ys = xy(ref_transform, sample_rows, sample_cols, offset="center")
    data = {
        "x": xs,
        "y": ys,
        "row": sample_rows,
        "col": sample_cols,
        "class": y,
        "spatial_block": blocks[sample_rows, sample_cols],
    }
    for name in all_features:
        data[name] = stack[name][sample_rows, sample_cols]
    df = pd.DataFrame(data)
    return df, valid_base, valid_insar, common_valid


def choose_splits(df):
    n_blocks = df["spatial_block"].nunique()
    n_splits = int(min(5, n_blocks))
    if n_splits < 2:
        raise RuntimeError("Not enough spatial blocks for spatial cross-validation.")
    return n_splits


def train_eval_experiment(df, features, name):
    X = df[features].to_numpy(dtype="float32")
    y = df["class"].to_numpy(dtype=int)
    groups = df["spatial_block"].to_numpy()
    n_splits = choose_splits(df)

    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_SEED)
    oof_prob = np.zeros(len(df), dtype="float32")
    oof_pred = np.zeros(len(df), dtype=int)
    fold_rows = []

    for fold, (train_idx, test_idx) in enumerate(cv.split(X, y, groups), start=1):
        model = RandomForestClassifier(
            n_estimators=500,
            max_features="sqrt",
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=RANDOM_SEED + fold,
            n_jobs=-1,
        )
        model.fit(X[train_idx], y[train_idx])
        prob = model.predict_proba(X[test_idx])[:, 1]
        pred = (prob >= 0.5).astype(int)
        oof_prob[test_idx] = prob
        oof_pred[test_idx] = pred
        fold_rows.append(
            {
                "experiment": name,
                "fold": fold,
                "n_train": len(train_idx),
                "n_test": len(test_idx),
                "test_positive": int(y[test_idx].sum()),
                "test_negative": int((y[test_idx] == 0).sum()),
                "auroc": roc_auc_score(y[test_idx], prob) if len(np.unique(y[test_idx])) == 2 else np.nan,
                "f1": f1_score(y[test_idx], pred, zero_division=0),
                "precision": precision_score(y[test_idx], pred, zero_division=0),
                "recall": recall_score(y[test_idx], pred, zero_division=0),
            }
        )

    final_model = RandomForestClassifier(
        n_estimators=500,
        max_features="sqrt",
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=RANDOM_SEED,
        n_jobs=-1,
    )
    final_model.fit(X, y)

    metrics = {
        "experiment": name,
        "features": features,
        "samples": int(len(df)),
        "positive_samples": int(y.sum()),
        "negative_samples": int((y == 0).sum()),
        "spatial_blocks": int(df["spatial_block"].nunique()),
        "cv_folds": n_splits,
        "oof_auroc": float(roc_auc_score(y, oof_prob)),
        "oof_f1": float(f1_score(y, oof_pred, zero_division=0)),
        "oof_precision": float(precision_score(y, oof_pred, zero_division=0)),
        "oof_recall": float(recall_score(y, oof_pred, zero_division=0)),
        "confusion_matrix": confusion_matrix(y, oof_pred).tolist(),
        "feature_importance": dict(zip(features, final_model.feature_importances_.astype(float))),
    }
    joblib.dump(final_model, MODEL_DIR / f"random_forest_{name}.joblib")
    pd.DataFrame(fold_rows).to_csv(MODEL_DIR / f"fold_metrics_{name}.csv", index=False)
    return final_model, metrics, oof_prob, oof_pred


def predict_raster(stack, features, valid_mask, model, ref_profile, out_name):
    rows, cols = valid_mask.shape
    pred = np.full((rows, cols), np.nan, dtype="float32")
    sample = np.column_stack([stack[f][valid_mask] for f in features]).astype("float32")
    pred[valid_mask] = model.predict_proba(sample)[:, 1].astype("float32")

    profile = ref_profile.copy()
    profile.update(dtype="float32", count=1, nodata=-9999.0, compress="deflate", predictor=2)
    out_tif = MAP_DIR / f"susceptibility_{out_name}.tif"
    with rasterio.open(out_tif, "w", **profile) as dst:
        dst.write(np.where(np.isfinite(pred), pred, -9999.0).astype("float32"), 1)
        dst.set_band_description(1, f"Landslide susceptibility probability: {out_name}")
    return pred, out_tif


def save_susceptibility_png(pred, ref_transform, title, path):
    left, bottom, right, top = array_bounds(pred.shape[0], pred.shape[1], ref_transform)
    fig, ax = plt.subplots(figsize=(10, 7), dpi=250)
    cmap = LinearSegmentedColormap.from_list(
        "susceptibility",
        ["#1a9850", "#91cf60", "#fee08b", "#fc8d59", "#d73027"],
    )
    im = ax.imshow(pred, cmap=cmap, vmin=0, vmax=1, extent=(left, right, bottom, top), origin="upper")
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel("Easting (m)")
    ax.set_ylabel("Northing (m)")
    cb = fig.colorbar(im, ax=ax, fraction=0.036, pad=0.03)
    cb.set_label("Probability")
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def save_diagnostics(metrics, df, oof_payloads):
    summary = pd.DataFrame(
        [
            {
                "experiment": m["experiment"],
                "features": ", ".join(m["features"]),
                "samples": m["samples"],
                "positive_samples": m["positive_samples"],
                "negative_samples": m["negative_samples"],
                "spatial_blocks": m["spatial_blocks"],
                "cv_folds": m["cv_folds"],
                "AUROC": m["oof_auroc"],
                "F1": m["oof_f1"],
                "precision": m["oof_precision"],
                "recall": m["oof_recall"],
            }
            for m in metrics
        ]
    )
    summary.to_csv(MODEL_DIR / "model_comparison_summary.csv", index=False)
    (MODEL_DIR / "model_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    for name, y, prob, pred in oof_payloads:
        fpr, tpr, _ = roc_curve(y, prob)
        roc_auc = auc(fpr, tpr)
        fig, ax = plt.subplots(figsize=(6, 5), dpi=250)
        RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc, estimator_name=name).plot(ax=ax)
        ax.plot([0, 1], [0, 1], "k--", linewidth=1)
        ax.set_title(f"Spatial CV ROC: {name}")
        fig.savefig(MODEL_DIR / f"roc_{name}.png", bbox_inches="tight", facecolor="white")
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(5, 5), dpi=250)
        ConfusionMatrixDisplay.from_predictions(y, pred, display_labels=["Non-LS", "LS"], cmap="Blues", ax=ax)
        ax.set_title(f"Spatial CV Confusion Matrix: {name}")
        fig.savefig(MODEL_DIR / f"confusion_matrix_{name}.png", bbox_inches="tight", facecolor="white")
        plt.close(fig)


def main():
    ensure_dirs()
    stack, ref_profile, ref_transform, ref_crs, ref_shape = build_aligned_stack()
    aoi_mask = rasterize_aoi(ref_shape, ref_transform, ref_crs)
    landslide_mask = landslide_pixel_mask(ref_shape, ref_transform, ref_crs)
    write_mask(aoi_mask, ref_profile, DATA_DIR / "study_area_mask.tif", "Study area mask")
    write_mask(landslide_mask, ref_profile, DATA_DIR / "landslide_presence_pixels.tif", "Landslide presence pixels")

    df, valid_base, valid_insar, common_valid = build_samples(stack, aoi_mask, landslide_mask, ref_transform)
    write_mask(valid_base, ref_profile, DATA_DIR / "valid_base_predictor_mask.tif", "Valid base predictor mask")
    write_mask(valid_insar, ref_profile, DATA_DIR / "valid_with_insar_predictor_mask.tif", "Valid with-InSAR predictor mask")
    write_mask(common_valid, ref_profile, DATA_DIR / "common_training_mask.tif", "Common training mask")
    df.to_csv(DATA_DIR / "training_samples_common_insar_coverage.csv", index=False)

    base_model, base_metrics, base_prob, base_pred = train_eval_experiment(df, BASE_FEATURES, "without_insar")
    insar_model, insar_metrics, insar_prob, insar_pred = train_eval_experiment(df, INSAR_FEATURES, "with_insar")

    pred_base, base_tif = predict_raster(stack, BASE_FEATURES, valid_base, base_model, ref_profile, "without_insar")
    pred_insar, insar_tif = predict_raster(stack, INSAR_FEATURES, valid_insar, insar_model, ref_profile, "with_insar")
    save_susceptibility_png(pred_base, ref_transform, "Landslide Susceptibility Without InSAR", MAP_DIR / "susceptibility_without_insar.png")
    save_susceptibility_png(pred_insar, ref_transform, "Landslide Susceptibility With InSAR", MAP_DIR / "susceptibility_with_insar.png")

    y = df["class"].to_numpy(dtype=int)
    save_diagnostics(
        [base_metrics, insar_metrics],
        df,
        [
            ("without_insar", y, base_prob, base_pred),
            ("with_insar", y, insar_prob, insar_pred),
        ],
    )

    shutil.copy2(Path(__file__), OUT_DIR / "prepare_joshimath_lsm_rf.py")
    print("Done.")
    print(f"Training samples: {len(df)}; positives: {int(df['class'].sum())}; negatives: {int((df['class']==0).sum())}")
    print(f"Without InSAR AUROC={base_metrics['oof_auroc']:.3f}, F1={base_metrics['oof_f1']:.3f}")
    print(f"With InSAR    AUROC={insar_metrics['oof_auroc']:.3f}, F1={insar_metrics['oof_f1']:.3f}")
    print(f"Saved outputs in: {OUT_DIR}")
    print(f"Susceptibility rasters: {base_tif}; {insar_tif}")


if __name__ == "__main__":
    main()
