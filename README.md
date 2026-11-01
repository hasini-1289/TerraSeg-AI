# TerraSeg AI: Landslide Segmentation from Satellite Imagery

Author : Derangula Hasini

This repository holds the notebooks, settings and result tables for a study of which inputs help a deep network find landslides in satellite images. It is a research prototype, not a finished product. Some experiments worked and some did not, and both are documented below.

---

## 1. What is this project about?

A landslide covers only a few percent of the pixels in a satellite image (about 2% in Landslide4Sense). We train segmentation networks that mark landslide pixels, and we ask:
- which input channels (RGB, 12 optical bands, slope, elevation) help a ResNet34 U-Net on Landslide4Sense;
- whether focal loss, scSE attention, rotation augmentation and a YOLO11 attention block (C2PSA) add anything;
- whether the benefit of terrain data carries over to a second dataset (Bijie);
- how a YOLO11 segmentation model compares with the U-Net;
- whether Sentinel-1 radar can be fused in (it could not, in our attempts).

All results use fixed, saved splits. Where possible we ran three seeds (42, 123, 777) and report mean and standard deviation.

## 2. Papers and sources used
- Ji et al., 2020, Bijie dataset and attention-boosted CNN, *Landslides* 17.
- Ghorbanzadeh et al., 2022, Landslide4Sense benchmark, *IEEE TGRS* 60.
- Ghorbanzadeh et al., 2022, Landslide4Sense 2022 competition outcome, *IEEE JSTARS* 15, 9927 to 9942.
- U-Net (Ronneberger et al., 2015), ResNet (He et al., 2016), focal loss (Lin et al., 2017), V-Net Dice loss (Milletari et al., 2016), scSE (Roy et al., 2018), Grad-CAM (Selvaraju et al., 2017), YOLO11 (Ultralytics).


## 3. How this differs from existing work
We did not invent a new network. The difference is in method and honesty of reporting: a fixed split, repeated seeds with `gain/std`, false alarms measured at matched recall, the same terrain question tested on two datasets, and a written record of what did not work. The Landslide4Sense 2022 winners used heavier methods (ensembles of transformer models) and a hidden test set, so our numbers are not directly comparable to theirs (first place test F1 74.54%, second place 73.99%; our all-channel U-Net F1 is about 0.756 on our own random test split). **[Confirm the competition numbers against the cited paper.]**

## 4. Methods and formulas used
- ResNet34 U-Net (ImageNet weights, first layer adapted to the number of input channels), `segmentation_models_pytorch`.
- Losses: binary cross-entropy + Dice; focal + Dice (alpha 0.75, gamma 2) as an experiment.
- scSE decoder attention; C2PSA block inserted after the deepest encoder stage.
- YOLO11s-seg (Ultralytics), 3, 4 and 14 input channels on Landslide4Sense; RGB on Bijie.
- Metrics (pixel counts pooled over the test set): IoU = TP/(TP+FP+FN); F1 = 2TP/(2TP+FP+FN); precision, recall; boundary F1 with 2 pixel tolerance.
- Threshold chosen on validation. False alarms at matched recall: highest threshold that reaches the target recall on validation, then count false-positive pixels on test.
- Seed variability: `gain/std` = (mean difference) / pooled standard deviation. Below 2 is treated as inconclusive.
- Full equations are in the paper (`paper/main.tex`).

## 5. Results (verified from the saved result files)

**Bijie baselines (single run, first split)**
| Model | IoU | F1 |
|---|---|---|
| Custom CNN | 0.277 | 0.434 |
| U-Net | 0.262 | 0.415 |
| U-Net + Dice + flips | 0.288 | 0.447 |
| ResNet34 U-Net | 0.756 | 0.861 |

**Landslide4Sense, ResNet34 U-Net, 14 channels (test, three seeds unless noted)**
| Experiment | Seeds | IoU |
|---|---|---|
| E0 baseline | 3 | 0.6098 +/- 0.0072 |
| E1 focal + Dice | 1 | 0.5586 |
| E2 scSE | 3 | 0.6048 +/- 0.0054 |
| E3 rotation | 1 | 0.6082 |
| E4 combined | 3 | 0.5254 +/- 0.0730 |

**Channel study (Landslide4Sense)**
| Input | Seeds | IoU | gain/std vs RGB |
|---|---|---|---|
| RGB | 3 | 0.5716 +/- 0.0024 | n/a |
| RGB + slope | 3 | 0.5882 +/- 0.0122 | 1.9 |
| All 14 channels | 3 | 0.6079 +/- 0.0069 | 7.0 |
| Other three variants | 1 | see `results/phase7_channel_study.csv` | n/a |

False alarms: all 14 channels give 18% fewer false-positive pixels than RGB at validation thresholds (three seeds). At matched 70% recall the reduction is 29% (seed 42 only).

**Bijie terrain (three seeds)**
| Input | IoU |
|---|---|
| RGB | 0.7889 +/- 0.0178 |
| RGB + DEM | 0.7794 +/- 0.0265 |
| RGB + relief | 0.7903 +/- 0.0060 |
| RGB + DEM + relief | 0.7772 +/- 0.0258 |

**YOLO11-seg on Landslide4Sense (seed 42):** RGB IoU 0.231, RGB+slope 0.529. The 14-channel run is excluded because its saved log shows only two epochs.

**Not achieved:** Sentinel-1 radar fusion (IoU 0.01 to 0.12), cloud robustness, georeferenced output, Random Forest and speed results (not found in the saved files).

**[Update this section with Phase 7 and Phase 9 three-seed results and the Phase 8 table when they are final.]**

## 6. Repository layout
```
TerraSeg-AI/
  README.md
  requirements.txt
  .gitignore
  notebooks/
    01_phase1_evaluation_baselines.ipynb
    03_phase3_improvements.ipynb
    05_phase5_explainability_demo.ipynb
    06_yolo11seg_multichannel.ipynb
    07_channel_study_c2psa.ipynb
    08_bijie_yolo_vs_unet.ipynb
    09_bijie_dem_final.ipynb
  results/            small CSV result tables and figures (PNG)
  paper/              main.tex
  docs/               project handoff, notebook guide
```
Model weights, datasets and full YOLO run folders are not stored here (too large). They are re-created by running the notebooks.

## 7. How to run
1. Open a notebook in Google Colab (set the runtime to GPU).
2. Mount Google Drive when asked. Outputs are written to `MyDrive/TerraSeg/...`.
3. Bijie notebooks need a Kaggle API key (`kaggle.json`) that you upload yourself. Never commit it.
4. Run order: 01 (Bijie baselines), 03 (Landslide4Sense improvements, creates `split.npz` and the final model), 05 (demo), 06, 07 (need the Phase 3 split), 08, 09 (Bijie).
5. Seeds are set in the settings cell (`SEEDS = [42, 123, 777]`). Finished runs are skipped on rerun.

## 8. Known limitations
- The Landslide4Sense split is random by patch, not by region, and not the official split.
- Several results use one seed (marked above).
- The 14-channel YOLO run looks unfinished.
- The first Bijie split differs from the later one, and notebooks normalise inputs differently, so numbers across them should not be mixed.
- Boundary F1 is not comparable across datasets.

## 9. Datasets and licences
- Bijie landslide dataset (Kaggle: `ksncik/bijie-landslide-dataset`).
- Landslide4Sense (Hugging Face: `ibm-nasa-geospatial/Landslide4sense`).
- Sen12Landslides (used only in the failed radar attempt).
Check each dataset's own licence before reuse.

## 10. Citation
//need to be added
