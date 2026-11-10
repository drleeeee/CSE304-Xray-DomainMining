# v3 validation and change log

- 2026-11-10 review ZIP verified: 2,809 unique review records; T 2,643, F 166 raw; 1 draft F withheld; confirmed F 165; support ROI availability is imbalanced across units.
- CLI smoke test completed with 36 original BMPs: extraction, review joining, overlay removal/gallery, pHash/time groups, classical/domain features, E4 stats and grouped E5 ablation outputs.
- pytest: 6 passed.
- Pyarrow unavailable in local test: explicit `.pkl` fallback; Colab installs pyarrow.
- YOLO11 and DINOv2 pretrained weight download and full 2,809-image GPU run not validated here.
- Overlay inpainting gallery should be reviewed before interpreting class scores; approximate filled pixels show visual changes where rectangles occluded X-ray.
