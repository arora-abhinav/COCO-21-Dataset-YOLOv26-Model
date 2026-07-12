# COCO-21 Pose — YOLO26x Fine-Tuned

A fine-tuned YOLO26x-pose model extending the standard COCO-17 human pose skeleton with 4 additional foot keypoints: left big toe, right big toe, left heel, and right heel. Fine-tuned on a filtered subset of COCO-WholeBody annotations where foot keypoints are validated (`foot_valid=True`).

This model was developed as part of the StrideLens project — a research-grounded running biomechanics analysis tool. The motivation for adding foot keypoints is that the standard COCO-17 skeleton only provides ankle keypoints, making it impossible to compute the foot-ground angle at contact — the most clinically meaningful indicator of running strike pattern. COCO-21 addresses this directly.

## Downloads

| Asset | Link |
|---|---|
| `best.pt` — COCO-21 fine-tuned weights (fine-tuned from `yolo26x-pose.pt`) | [Download](https://drive.google.com/file/d/1oo1F9Nrt6mRFGxPhHuommxGrNVz5AcCN/view?usp=sharing) |
| Training annotations (COCO-21 JSON) | [Download](https://drive.google.com/file/d/1uA2iQnog2W1zZDx6TAVr9fQeOTQE9NHX/view?usp=sharing) |
| Validation annotations (COCO-21 JSON) | [Download](https://drive.google.com/file/d/1oo1F9Nrt6mRFGxPhHuommxGrNVz5AcCN/view?usp=sharing) |

Images are sourced from the standard COCO train2017 and val2017 splits, available at [cocodataset.org](https://cocodataset.org).

## Keypoint Schema

COCO-21 extends the standard COCO-17 body keypoints with 4 foot keypoints appended at the end. The full ordered keypoint list is:

| Index | Keypoint |
|---|---|
| 0 | Nose |
| 1 | Left Eye |
| 2 | Right Eye |
| 3 | Left Ear |
| 4 | Right Ear |
| 5 | Left Shoulder |
| 6 | Right Shoulder |
| 7 | Left Elbow |
| 8 | Right Elbow |
| 9 | Left Wrist |
| 10 | Right Wrist |
| 11 | Left Hip |
| 12 | Right Hip |
| 13 | Left Knee |
| 14 | Right Knee |
| 15 | Left Ankle |
| 16 | Right Ankle |
| 17 | Left Big Toe |
| 18 | Right Big Toe |
| 19 | Left Heel |
| 20 | Right Heel |

Each keypoint is stored as `[x, y, v]` where `v` is the standard COCO visibility flag: `0` = not labeled, `1` = labeled but occluded, `2` = labeled and visible. Keypoints with `v=0` have meaningless `x` and `y` values.

The `kpt_shape` for this model is `[21, 3]`.

## Annotation Format

Annotations follow the standard COCO JSON format:

```json
{
    "info": {...},
    "licenses": [...],
    "images": [...],
    "annotations": [...],
    "categories": [...]
}
```

Each annotation object contains:

```json
{
    "id": int,
    "image_id": int,
    "category_id": 1,
    "segmentation": [...],
    "area": float,
    "bbox": [x, y, w, h],
    "iscrowd": 0,
    "keypoints": [x1, y1, v1, ..., x21, y21, v21],
    "num_keypoints": int,
    "foot_valid": bool
}
```

`foot_valid` is inherited from COCO-WholeBody and indicates whether the foot keypoints (indices 17–20) are reliably labeled for that annotation. Annotations where `foot_valid=False` have been retained in the dataset but their foot keypoint visibility values are zeroed out.

## Dataset

- **Source:** COCO-WholeBody annotations filtered to `foot_valid=True`, paired with COCO train2017/val2017 images
- **Training set:** ~10,969 images
- **Validation set:** ~1,175 images
- **Base model:** `yolo26x-pose.pt` — Ultralytics YOLO26x pretrained on COCO-17 (63M parameters, 227 GFLOPs)
- **Fine-tuned model:** `best.pt` — `yolo26x-pose.pt` fine-tuned on COCO-21 annotations
- **Transferred weights:** 1,179/1,263 layers transferred from pretrained checkpoint

## Training

Fine-tuned on a RunPod A40 (48GB VRAM) for 100 epochs.

```python
from ultralytics import YOLO

model = YOLO("yolo26x-pose.pt")

results = model.train(
    data="custom_coco.yaml",
    epochs=100,
    imgsz=640,
    batch=16,
    optimizer="MuSGD",
    device=0,
    workers=4,
)
```

YAML configuration:
```yaml
kpt_shape: [21, 3]
flip_idx: [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15, 18, 17, 20, 19]
nc: 1
names:
  0: person
```

## Validation Metrics

Evaluated on the COCO-21 validation set (1,175 images, 2,280 person instances).

### Box Detection (Person)

| Metric | Value |
|---|---|
| mAP@0.5 | 0.673 |
| Precision | 0.96 |
| Recall | 0.97 |

### Pose Estimation

| Metric | Value |
|---|---|
| mAP@0.5 | 0.616 |
| F1 | 0.66 (at confidence 0.842) |

The pose mAP is lower than box mAP as expected — the 4 new foot keypoints (big toes, heels) are small, frequently occluded by footwear, and the model was fine-tuned on a relatively small subset of COCO-WholeBody. The existing 17 COCO body keypoints retain their original performance from pretraining; the new keypoints are the primary source of pose mAP reduction.

The confusion matrix shows 92% of true person instances correctly detected with 8% missed — consistent with the base model's detection capability being largely preserved through fine-tuning.

## Usage

```python
from ultralytics import YOLO

model = YOLO("best.pt")
results = model("your_image.jpg")

for result in results:
    keypoints = result.keypoints.xy  # (N, 21, 2) — x,y per keypoint per person
    # Index 17: left big toe
    # Index 18: right big toe
    # Index 19: left heel
    # Index 20: right heel
```

## Limitations

- Foot keypoints are only reliable for images where feet are clearly visible and unoccluded
- Performance on running video (fast motion, side-view, treadmill footage) has not been formally benchmarked — the training data is general COCO imagery, not domain-specific running footage
- Training dataset size (~10,969 images) is smaller than the full COCO-17 training set; additional fine-tuning on larger foot-annotated datasets would improve localization precision

## Citation

If you use these annotations or weights, please also cite the original COCO-WholeBody dataset:

```
Jin et al. (2020). Whole-Body Human Pose Estimation in the Wild.
ECCV 2020.
```
