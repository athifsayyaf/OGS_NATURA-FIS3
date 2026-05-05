# OGS NATURA-FIS3 Stage 2 Mini-Project

This repository contains the Stage 2 mini-project submission for landslide susceptibility mapping with GIS predictors, Sentinel-2 NDVI, InSAR LOS velocity, and Random Forest spatial cross-validation.

## Study Area

The pilot study area is Joshimath, Uttarakhand, India. The area was selected because it is a steep Himalayan terrain with known slope-instability concerns and a landslide inventory available for rasterization.

## Additional Self-Initiated Work

Beyond the required mini-project tasks, I also developed a GUI-based prototype software for landslide susceptibility and monitoring workflows. The software is shared separately here:

[SlopeGuard AI - GUI software prototype](https://github.com/athifsayyaf/SlopeGuard)

This was built as an additional self-initiated effort to package the GIS, InSAR, model-comparison, map-preview, and report-generation workflow into a more user-friendly software interface.

## Main Result

Two Random Forest models were compared using spatial cross-validation over the common valid InSAR coverage.

| Experiment | Features | AUROC | F1 | Precision | Recall |
|---|---|---:|---:|---:|---:|
| Without InSAR | elevation, slope, aspect, TWI, NDVI | 0.872 | 0.828 | 0.800 | 0.857 |
| With InSAR | elevation, slope, aspect, TWI, NDVI, LOS velocity | 0.893 | 0.857 | 0.857 | 0.857 |

Adding InSAR LOS velocity gave a modest improvement in AUROC and F1-score. The improvement is interpreted cautiously because only 28 samples fell inside the common valid InSAR coverage.

## Workflow

1. Derive DEM terrain layers: slope, aspect, and TWI.
2. Prepare Sentinel-2 NDVI.
3. Rasterize landslide inventory to a binary presence mask.
4. Align all predictors to the same CRS, resolution, and extent.
5. Add aligned InSAR LOS velocity to the analysis stack.
6. Train Random Forest models with spatial cross-validation:
   - baseline terrain/optical model
   - expanded terrain/optical/InSAR model
7. Export AUROC, F1-score, confusion matrices, ROC curves, feature importance, and susceptibility maps.

The scripts currently preserve the original local data paths used during the assignment. The committed `outputs/` folder contains the generated products needed to inspect and evaluate the result without downloading the full raw data archive.

## Important Limitations

- The available InSAR data included LOS velocity but not a full coherence raster product. The InSAR comparison therefore uses aligned velocity, and the report clearly states this limitation.
- The common valid InSAR sample size was small: 28 total samples, 14 landslide and 14 non-landslide.
- Non-landslide samples were treated as stable background samples, not field-verified stable locations.
- The result should be read as a transparent pilot workflow rather than an operational susceptibility product.

## Interpretation

Elevation remained the strongest predictor, while InSAR velocity contributed meaningful additional information in the expanded model. The InSAR-assisted run reduced false positives and slightly improved AUROC/F1, suggesting that deformation information can complement static terrain and optical features where valid InSAR coverage is available.
