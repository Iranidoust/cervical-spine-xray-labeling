import json
import cv2
import matplotlib.pyplot as plt
from pathlib import Path

IMAGE_DIR = Path("data/raw/images")
JSON_DIR = Path("data/raw/json")

json_file = next(JSON_DIR.glob("*.json"))
image_file = IMAGE_DIR / f"{json_file.stem}.png"

print("JSON :", json_file)
print("IMAGE:", image_file)

image = cv2.imread(str(image_file))

if image is None:
    raise RuntimeError(f"Cannot read {image_file}")

image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

with open(json_file, "r", encoding="utf-8") as f:
    data = json.load(f)

plt.figure(figsize=(10, 14))
plt.imshow(image)

for shape in data["shapes"]:

    label = str(shape.get("label", ""))

    for point in shape.get("points", []):
        x, y = point[:2]

        plt.scatter(x, y, s=20)

        plt.text(
            x + 5,
            y,
            label,
            fontsize=6,
            bbox=dict(alpha=0.4)
        )

plt.axis("off")
plt.tight_layout()
plt.show()