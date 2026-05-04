# InSAR-Aware Landslide Susceptibility Mapping for Joshimath, Uttarakhand

**Sansar PhD Assignment Mini-Project**

## Abstract
This mini-project builds an analysis-ready GIS stack and Random Forest classifier for landslide susceptibility mapping in the Joshimath region. DEM-derived terrain layers, Sentinel-2 NDVI, a landslide inventory, and an available InSAR velocity raster were aligned to a common grid. Spatial cross-validation was used to compare models trained with and without InSAR.

## Background
Joshimath lies in steep Himalayan terrain where landslide susceptibility is controlled by relief, drainage concentration, vegetation disturbance, lithologic structure, and surface deformation. The project tests whether an InSAR velocity layer adds useful information beyond conventional terrain and optical predictors.

## Data & Methods
Copernicus DEM data were used to derive slope, aspect, and TWI. Sentinel-2 NDVI was prepared as an optical vegetation index. The landslide inventory was rasterized to presence pixels, and non-landslide samples were drawn outside a local buffer around inventory points. All predictors were aligned to the UTM Zone 44N analysis grid. Random Forest was chosen because it is robust for small tabular geospatial datasets, handles nonlinear feature interactions, and provides interpretable feature importance. Spatial cross-validation used 1 km blocks to reduce leakage from nearby terrain pixels.

## InSAR Processing
The available InSAR product was a mean LOS velocity raster over part of the study area. No mean coherence raster was found in the supplied InSAR folder, so a coherence threshold mask could not be applied. To keep the comparison fair, the two ML runs used the same samples within the common valid InSAR coverage.

## Results
Without InSAR, the spatial-CV AUROC was 0.872 and F1 was 0.828. With InSAR velocity, AUROC increased to 0.893 and F1 increased to 0.857. The comparison suggests that deformation information improved classification in this run, although the InSAR overlap was limited.

| Experiment | AUROC | F1 |
|---|---:|---:|
| without insar | 0.872 | 0.828 |
| with insar | 0.893 | 0.857 |

## Discussion
Feature importance indicates that elevation dominated the terrain/optical model, while elevation remained important after adding InSAR. In the with-InSAR model, velocity importance was 0.188, indicating that deformation contributed non-trivial predictive signal. High-susceptibility zones are expected to concentrate along steep valley flanks, dissected drainage corridors, and terrain with disturbed vegetation or active motion.

## Limitations
The principal limitation is sample size: only 14 landslide presence samples overlapped valid InSAR coverage, so the reported improvement should be treated as preliminary. The inventory is point-based rather than polygon-based, and absence samples are assumed rather than verified stable terrain. With more time, I would add a coherence layer, lithology, distance-to-road/stream/fault variables, larger negative sampling experiments, and independent temporal validation.
