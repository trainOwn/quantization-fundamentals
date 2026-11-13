
import fiftyone.zoo as foz
import shutil
from pathlib import Path

OUTPUT_DIR = Path("coco300")
IMAGE_DIR = OUTPUT_DIR / "images"

IMAGE_DIR.mkdir(parents=True, exist_ok=True)

print("Loading already downloaded COCO images...")

# Reuse the existing downloaded dataset
dataset = foz.load_zoo_dataset(
    "coco-2017",
    split="validation",
    max_samples=300,
    shuffle=True,
    seed=42,
    label_types=["detections"],
    dataset_name="coco300_calibration"
)

count = 0

for sample in dataset.iter_samples():
    src = Path(sample.filepath)

    if src.exists():
        shutil.copy2(
            src,
            IMAGE_DIR / src.name
        )
        count += 1

print("\nCompleted!")
print("Images:", count)
print("Location:", IMAGE_DIR.resolve())

