
#!/usr/bin/env python3
"""
Convert 21-keypoint COCO-format annotations (already produced by DataCleanup.ipynb,
which merges left/right big toe + left/right heel onto the base 17 keypoints and
drops foot_kpts) into Ultralytics YOLO Pose labels.

Input annotations are expected to already have:
- keypoints: 63 flat values (21 keypoints x,y,v), ordered as
  17 COCO body keypoints + left_big_toe + right_big_toe + left_heel + right_heel
- no foot_kpts field (already merged in)

Output:
dataset/
    images/
        train/
        val/
    labels/
        train/
        val/

Each label line:
class xc yc w h kp1x kp1y v ... kp21x kp21y v
"""

import json
import shutil
from pathlib import Path
from collections import defaultdict

from PIL import Image

ROOT = Path("dataset")

TRAIN_JSON = ROOT / "annotations" / "training_data_COCO21.json"
VAL_JSON = ROOT / "annotations" / "validation_data_COCO21.json"

TRAIN_IMAGES = ROOT / "images" / "train"
VAL_IMAGES = ROOT / "images" / "val"

OUT_IMAGES = ROOT / "images"
OUT_LABELS = ROOT / "labels"

COPY_IMAGES = False


def ensure_dirs():
    for p in [
        OUT_IMAGES/"train",
        OUT_IMAGES/"val",
        OUT_LABELS/"train",
        OUT_LABELS/"val",
    ]:
        p.mkdir(parents=True, exist_ok=True)


def normalize_bbox(bbox, w, h):
    x, y, bw, bh = bbox
    xc = (x + bw/2) / w
    yc = (y + bh/2) / h
    return xc, yc, bw/w, bh/h


def normalize_keypoints(flat, w, h):
    out = []
    for i in range(0, len(flat), 3):
        x = flat[i] / w
        y = flat[i+1] / h
        v = int(flat[i+2])
        out.extend([
            min(max(x,0.0),1.0),
            min(max(y,0.0),1.0),
            v
        ])
    return out


def convert(json_path, image_dir, split):

    print(f"\nProcessing {split}")

    with open(json_path, "r") as f:
        data = json.load(f)

    image_lookup = {
        im["id"]: im
        for im in data["images"]
    }

    anns_per_image = defaultdict(list)

    skipped = 0
    converted = 0

    for ann in data["annotations"]:

        if ann.get("iscrowd",0):
            continue

        img = image_lookup.get(ann["image_id"])
        if img is None:
            skipped += 1
            continue

        filename = img["file_name"]
        src = image_dir / filename

        if not src.exists():
            skipped += 1
            continue

        w = img["width"]
        h = img["height"]

        kp = ann["keypoints"]

        if len(kp) != 63:
            skipped += 1
            continue

        kp = normalize_keypoints(kp, w, h)

        xc, yc, bw, bh = normalize_bbox(
            ann["bbox"],
            w,
            h
        )

        line = [
            "0",
            f"{xc:.6f}",
            f"{yc:.6f}",
            f"{bw:.6f}",
            f"{bh:.6f}",
        ]

        for i in range(0, len(kp), 3):
            line.append(f"{kp[i]:.6f}")
            line.append(f"{kp[i+1]:.6f}")
            line.append(str(kp[i+2]))

        if len(line) != 68:
            raise RuntimeError(
                f"Expected 68 values, got {len(line)}"
            )

        anns_per_image[filename].append(" ".join(line))
        converted += 1

    print(f"Annotations converted: {converted}")
    print(f"Skipped: {skipped}")

    for filename, lines in anns_per_image.items():

        label_path = OUT_LABELS / split / (Path(filename).stem + ".txt")

        with open(label_path, "w") as f:
            f.write("\n".join(lines))

        if COPY_IMAGES:
            shutil.copy2(
                image_dir / filename,
                OUT_IMAGES / split / filename
            )


def verify(n=5):

    print("\nVerifying labels...")

    train_imgs = list((OUT_IMAGES/"train").glob("*.jpg"))

    if not train_imgs:
        print("No images found.")
        return

    for img_path in train_imgs[:n]:

        label = OUT_LABELS/"train"/(img_path.stem+".txt")

        if not label.exists():
            continue

        img = Image.open(img_path)

        print(img_path.name, img.size)

        with open(label) as f:
            vals = f.readline().split()

        if len(vals) != 68:
            print("Bad label:", img_path.name)
        else:
            print("OK")


def main():

    ensure_dirs()

    convert(
        TRAIN_JSON,
        TRAIN_IMAGES,
        "train"
    )

    convert(
        VAL_JSON,
        VAL_IMAGES,
        "val"
    )

    verify()


if __name__ == "__main__":
    main()
