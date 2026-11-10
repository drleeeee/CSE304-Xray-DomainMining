# CSE304 X-ray Domain Mining v3

## Research objective
Data mining of **machine-marking adequacy** on NG industrial radiographs. **Not** verified foreign-object presence, detection POD or material identification.

## New 2026-11-10 review labels
`xray-labels-2026-11-10.zip`: 2,809 records (2,643 T, 166 F raw). One F is `draft`, excluded. Final confirmed image-level labels: **2,643 T / 165 F**. Across units SN77128, SN77127, SN12053. Of confirmed F, **104 have zero supporting ROI boxes**, **61 have 1+**; this is a box-availability grouping, **not a validated mechanism**. 3rd-unit T has no ROIs in the newly supplied file, limiting fair ROI-based supervised studies.

## Source inputs
Place exactly eight original KAMP ZIP parts + **one** new `xray-labels-2026-11-10.zip` in Google Drive `MyDrive/CSE304_Xray/input_zips/`. Remove older 2026-11-03 review ZIP; otherwise the loader **intentionally stops**. Keep filename and SN folder paths unchanged. `part008` may not have a valid ZIP central directory; the archive audit explicitly reports this and core BMPs come from the relevant image archives.

## Colab runner
Upload `CSE304_Xray_DomainMining_Colab_Repo_v3.zip` into `MyDrive/CSE304_Xray/`. Open `Run_All_Colab_DomainMining_v3.ipynb`. Run `FAST_TEST=True` first, then set `FAST_TEST=False`. The notebook mounts Drive, installs dependencies and invokes the `.py` package.

## What was added to v3
- `overlay.py`: high-saturation color mask, dilation and Telea inpainting (approximate) plus QC originals | masks | cleaned images.
- `grouping.py`: exact SHA256, pHash and timestamp+perceptual similarity capture groups; label conflict audit; group-disjoint CV.
- `ingest.py`: all three units reviewed, exclude `draft`, separate F with vs without support ROI annotation.
- `statistics_v3.py`: E4 interpretable-feature difference, Mann–Whitney / Cliff δ / BH-FDR + capture-group bootstrap confidence intervals. Uncorrected p-values exploratory; for confirmatory statistical testing consider group permutation.
- `ablation.py`: E5 shortcut control (unit, preset, overlay), global YOLO, interpretable, candidate ROI + interpretable, full combination. Models: logistic / random forest. Each CV fold independently fits imputer + scaler for logistic. AP, ROC-AUC and top-10%-budget Recall. Missing feature families are explicitly marked.
- ROI annotation availability audit. ROI-containing comparisons skipped when below 80% in either class to avoid feature-presence shortcut.
- smoke-tested on 36 *actual* uploaded BMPs (CPU classical features/E4/E5), **not** a full GPU/2,809-image run.

## Run via Python directly
```bash
pip install -r requirements.txt
pip install -e .
OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 python run.py \
  --input /path/to/input_zips \
  --output /path/to/results \
  --work /content/xray_work \
  --with-yolo-roi
```
Use `--no-deep --max-images 50` for an initial smoke test. For a full result, do not set `--max-images` or `--no-deep`.

## Important files
- `audit/label_audit.json`, `audit/groups/capture_groups.csv`, `audit/groups/group_label_conflicts.csv`
- `quality_control/overlay/overlay_*.png` (panels source | mask | approximate cleaned)
- `features/basic_features_v3.parquet`, `features/candidate_features_v3.parquet`, `features/all_image_features.parquet`
- `analysis/e4/e4_group_bootstrap_statistics.csv`
- `analysis/ablations/all_reviewed/feature_family_ablation.csv` and subset/per-unit results

## Limitations
**Crucial**: Overlay removal is lossy and cannot restore the original X-ray pixels occluded by colored machine markings. Review the masks and compare the potential inpainting artifacts. All 2,809 images are NG; there is no clean OK control set. F is heterogeneous (not all confirmed missing foreign bodies). The hand-drawn supporting ROI is not true object segmentation. For Unit 3, T records in the latest ZIP have no candidate boxes, so fair Unit 3 ROI-based T/F evaluation is **not supported**. Global YOLO/ResNet/DINOv2 are **frozen pretrained embeddings**, not X-ray detector finetuning. Candidate-level missed/covered target is **not inferred** from image-level F.
