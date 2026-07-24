# Steel Surface Defect Detection

Object detection model that identifies and localizes 6 types of surface defects on hot-rolled steel, trained on the NEU-DET dataset. Built as an end-to-end pipeline: cloud data storage, dataset preparation, model training, and error analysis.

## Results

Trained YOLOv8n for 100 epochs (early stopping enabled, no improvement observed beyond convergence).

| Metric | Value |
|---|---|
| mAP50 (overall) | 0.740 |
| mAP50-95 (overall) | 0.392 |

**Per-class breakdown:**

| Class | Precision | Recall | mAP50 |
|---|---|---|---|
| patches | 0.826 | 0.881 | 0.938 |
| scratches | 0.575 | 0.901 | 0.881 |
| inclusion | 0.764 | 0.767 | 0.832 |
| pitted_surface | 0.751 | 0.747 | 0.817 |
| rolled-in_scale | 0.425 | 0.448 | 0.496 |
| crazing | 0.614 | 0.290 | 0.475 |

## What works, and what doesn't

Four of six classes reach strong performance (mAP50 0.82–0.94). Two — `crazing` and `rolled-in_scale` — lag significantly behind, even after full convergence.

Investigating `crazing` specifically: visual inspection of false negatives shows the model correctly identifies plausible defect regions (localized boxes with reasonable confidence), but these don't sufficiently overlap with ground truth to count as correct detections. The root cause traces back to the dataset's annotation style for this class — ground truth boxes for `crazing` tend to cover nearly the entire image, treating the defect as a diffuse, image-wide texture rather than a localized object. This creates a mismatch between how the model is trained to behave (predict tight, localized boxes) and what the label expects (near-full-image coverage).

This isn't a data volume issue (both underperforming classes have example counts in line with the others) or an undertraining artifact (confirmed by comparing a 5-epoch and 100-epoch run, where other classes improved substantially while these two plateaued).

**Proposed direction (not implemented in this iteration):** treat `crazing` and `rolled-in_scale` as a semantic segmentation problem rather than object detection, which would natively handle diffuse, contour-less patterns — or reformulate them as patch-level binary classification, closer to how they appear to have been originally annotated.

## Pipeline

GCS (raw/)  →  VOC → YOLO conversion  →  GCS (processed/)  →  local download  →  YOLOv8 training  →  GCS (models/)

- **Data storage**: Google Cloud Storage, structured as `raw/` (source dataset), `processed/` (YOLO-format, ready to train), `models/` (versioned checkpoints)
- **Conversion**: Pascal VOC XML annotations converted to YOLO format (normalized center/width/height), parallelized with a thread pool for I/O-bound GCS operations
- **Training**: YOLOv8n (Ultralytics), trained on Google Colab (T4 GPU)
- **Data integrity**: pipeline handles missing files gracefully (one orphaned annotation found and logged, not silently dropped or crash-inducing)

## Stack

Python · PyTorch · Ultralytics YOLOv8 · Google Cloud Storage · Pascal VOC / YOLO annotation formats

## Reproducing

```bash
git clone https://github.com/modh93/neu-steel-defect-detection.git
cd neu-steel-defect-detection
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Requires a GCS bucket with the NEU-DET dataset converted to YOLO format under `processed/neu-yolo/` (see `src/data/dataset.py` for the conversion pipeline).

```bash
python src/train.py --epochs 100 --patience 15 --run-name my_run
python src/evaluate.py --model runs/detect/runs/my_run/weights/best.pt --check-class crazing
```

## Next steps

- Segmentation-based approach for `crazing` / `rolled-in_scale`
- Evaluate larger YOLOv8 variants (s/m) for potential accuracy gains
- Targeted data augmentation for underperforming classes

## Dataset

[NEU Surface Defect Database](https://www.kaggle.com/datasets/kaustubhdikshit/neu-surface-defect-database) — 6 defect classes on hot-rolled steel surfaces, Pascal VOC annotation format.
