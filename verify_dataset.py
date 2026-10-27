import random
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


IMAGE_DIR = Path(
    "data/yolo/images/train"
)

LABEL_DIR = Path(
    "data/yolo/labels/train"
)

CLASS_NAMES = [
    "C2",
    "C3",
    "C4",
    "C5",
    "C6",
    "C7"
]


images = list(
    IMAGE_DIR.glob("*.png")
)

image_path = random.choice(images)

label_path = (
    LABEL_DIR /
    f"{image_path.stem}.txt"
)

image = cv2.imread(
    str(image_path)
)

height, width = image.shape[:2]

image = cv2.cvtColor(
    image,
    cv2.COLOR_BGR2RGB
)


with open(label_path) as f:

    lines = f.readlines()


plt.figure(figsize=(9, 13))
plt.imshow(image)


for line in lines:

    values = line.strip().split()

    class_id = int(values[0])

    coords = list(
        map(float, values[1:])
    )

    points = []

    for i in range(
        0,
        len(coords),
        2
    ):

        x = coords[i] * width
        y = coords[i + 1] * height

        points.append([x, y])


    polygon = np.array(points)

    closed = np.vstack(
        [
            polygon,
            polygon[0]
        ]
    )

    plt.plot(
        closed[:, 0],
        closed[:, 1],
        linewidth=2
    )

    cx = polygon[:, 0].mean()
    cy = polygon[:, 1].mean()

    plt.text(
        cx,
        cy,
        CLASS_NAMES[class_id],
        fontsize=12,
        fontweight="bold",
        bbox=dict(
            facecolor="white",
            alpha=0.7
        )
    )


plt.title(
    image_path.name
)

plt.axis("off")

plt.tight_layout()

plt.show()