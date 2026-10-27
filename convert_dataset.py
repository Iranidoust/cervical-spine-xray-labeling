import json
import math
import random
import shutil
import re

from pathlib import Path
from tqdm import tqdm
from PIL import Image


# ---------------------------------------------------------
# CONFIG
# ---------------------------------------------------------

IMAGE_DIR = Path("data/raw/images")
JSON_DIR = Path("data/raw/json")

OUTPUT = Path("data/yolo")

TRAIN_RATIO = 0.80
SEED = 42

CLASS_TO_ID = {
    "C2": 0,
    "C3": 1,
    "C4": 2,
    "C5": 3,
    "C6": 4,
    "C7": 5,
}

CORNER_ORDER = [
    "top left",
    "top right",
    "bottom right",
    "bottom left",
]


# ---------------------------------------------------------
# UTILITIES
# ---------------------------------------------------------

def load_json(path):

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def angle_sort(points):
    """
    Sort points around their centroid.
    Useful for C2 where the annotation isn't always
    the same four-corner structure as C3-C7.
    """

    # remove duplicates
    points = list(dict.fromkeys(
        (float(x), float(y))
        for x, y in points
    ))

    if len(points) < 3:
        return points

    cx = sum(x for x, _ in points) / len(points)
    cy = sum(y for _, y in points) / len(points)

    return sorted(
        points,
        key=lambda p: math.atan2(
            p[1] - cy,
            p[0] - cx
        )
    )


def extract_vertebrae(json_path):

    data = load_json(json_path)

    shapes = data.get("shapes", [])

    vertebrae = {}

    # --------------------------------------------------
    # C3-C7
    # --------------------------------------------------

    for level in ["C3", "C4", "C5", "C6", "C7"]:

        corners = {}

        for shape in shapes:

            label = " ".join(
                str(shape.get("label", "")).strip().lower().split()
            )

            points = shape.get("points", [])

            if not points:
                continue

            for corner in CORNER_ORDER:

                expected = f"{level.lower()} {corner}"

                if label == expected:

                    x, y = points[0][:2]

                    corners[corner] = (
                        float(x),
                        float(y)
                    )

        if all(c in corners for c in CORNER_ORDER):

            vertebrae[level] = [
                corners[c]
                for c in CORNER_ORDER
            ]

    # --------------------------------------------------
    # C2
    # --------------------------------------------------

    c2_points = []

    for shape in shapes:

        label = str(shape.get("label", "")).strip()

        # CSXA may refer to C2 as C2 or Axis
        if re.search(r"\b(C2|Axis)\b", label, re.IGNORECASE):

            for point in shape.get("points", []):

                if len(point) >= 2:

                    c2_points.append(
                        (
                            float(point[0]),
                            float(point[1])
                        )
                    )

    c2_points = angle_sort(c2_points)

    if len(c2_points) >= 3:
        vertebrae["C2"] = c2_points

    return vertebrae


def create_yolo_label(
    json_path,
    width,
    height
):

    vertebrae = extract_vertebrae(json_path)

    lines = []

    for level in [
        "C2",
        "C3",
        "C4",
        "C5",
        "C6",
        "C7"
    ]:

        if level not in vertebrae:
            continue

        points = vertebrae[level]

        if len(points) < 3:
            continue

        class_id = CLASS_TO_ID[level]

        coords = []

        for x, y in points:

            xn = x / width
            yn = y / height

            # safety
            xn = max(0.0, min(1.0, xn))
            yn = max(0.0, min(1.0, yn))

            coords.extend([
                f"{xn:.6f}",
                f"{yn:.6f}"
            ])

        line = (
            str(class_id)
            + " "
            + " ".join(coords)
        )

        lines.append(line)

    return lines


# ---------------------------------------------------------
# PREPARE OUTPUT
# ---------------------------------------------------------

if OUTPUT.exists():
    shutil.rmtree(OUTPUT)

for split in ["train", "val"]:

    (OUTPUT / "images" / split).mkdir(
        parents=True,
        exist_ok=True
    )

    (OUTPUT / "labels" / split).mkdir(
        parents=True,
        exist_ok=True
    )


# ---------------------------------------------------------
# MATCH FILES
# ---------------------------------------------------------

pairs = []

for json_path in JSON_DIR.glob("*.json"):

    image_path = IMAGE_DIR / (
        json_path.stem + ".png"
    )

    if image_path.exists():
        pairs.append(
            (image_path, json_path)
        )


print("Matched image/JSON pairs:", len(pairs))

random.seed(SEED)
random.shuffle(pairs)

split_index = int(
    len(pairs) * TRAIN_RATIO
)

train_pairs = pairs[:split_index]
val_pairs = pairs[split_index:]


print("Training :", len(train_pairs))
print("Validation:", len(val_pairs))


# ---------------------------------------------------------
# PROCESS SPLIT
# ---------------------------------------------------------

def process_split(pairs, split):

    good = 0
    incomplete = 0

    for image_path, json_path in tqdm(
        pairs,
        desc=split
    ):

        with Image.open(image_path) as img:
            width, height = img.size

        lines = create_yolo_label(
            json_path,
            width,
            height
        )

        # Ideally six vertebrae
        if len(lines) != 6:
            incomplete += 1

        target_image = (
            OUTPUT /
            "images" /
            split /
            image_path.name
        )

        target_label = (
            OUTPUT /
            "labels" /
            split /
            f"{image_path.stem}.txt"
        )

        shutil.copy2(
            image_path,
            target_image
        )

        with open(
            target_label,
            "w",
            encoding="utf-8"
        ) as f:

            f.write("\n".join(lines))

        good += 1

    print()
    print(split)
    print("processed:", good)
    print(
        "images not containing exactly 6 polygons:",
        incomplete
    )


process_split(train_pairs, "train")
process_split(val_pairs, "val")

print("\nConversion finished.")