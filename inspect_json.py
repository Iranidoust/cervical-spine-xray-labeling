import json
from pathlib import Path
from collections import Counter

JSON_DIR = Path("data/raw/json")

files = list(JSON_DIR.glob("*.json"))

print("JSON files:", len(files))

if not files:
    raise RuntimeError("No JSON files found")

sample = files[0]

print("\nSample file:")
print(sample)

with open(sample, "r", encoding="utf-8") as f:
    data = json.load(f)

print("\nTop-level keys:")
print(data.keys())

print("\nImage:")
print(data.get("imagePath"))

print("\nNumber of shapes:")
print(len(data.get("shapes", [])))

print("\nFirst 10 shapes:")

for shape in data.get("shapes", [])[:10]:
    print(
        "label =", shape.get("label"),
        "points =", shape.get("points"),
        "type =", shape.get("shape_type")
    )

print("\nAll labels in dataset:")

counter = Counter()

for file in files:
    with open(file, "r", encoding="utf-8") as f:
        d = json.load(f)

    for shape in d.get("shapes", []):
        counter[str(shape.get("label", "")).strip()] += 1

for name, count in sorted(counter.items()):
    print(f"{name:30s} {count}")