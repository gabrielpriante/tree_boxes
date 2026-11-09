# Import sys so the script can receive the image path and optional run label from the command line.
import sys

# Import Path so file and directory paths are handled cleanly across operating systems.
from pathlib import Path

# Import the main DeepForest model interface.
from deepforest import main

# Import DeepForest's visualization helper so prediction boxes can be drawn on the analyzed image.
from deepforest.visualize import plot_results


# Define the exact pretrained tree detection model repository used for this baseline.
MODEL_NAME = "weecology/deepforest-tree"

# Define the model revision requested from the repository.
MODEL_REVISION = "main"

# Define the image patch size in pixels used during tiled inference.
PATCH_SIZE = 400

# Define the fractional overlap between adjacent inference patches.
PATCH_OVERLAP = 0.05

# Define the IoU threshold used to suppress duplicate detections created across overlapping tiles.
IOU_THRESHOLD = 0.15

# Define the minimum model confidence required for a tree detection to be retained.
SCORE_THRESHOLD = 0.10

# Define the model-level non-maximum suppression threshold.
NMS_THRESHOLD = 0.05


# Read the 7.89 cm/pixel derived image path supplied as the first command-line argument.
image_path = Path(sys.argv[1])

# Read an optional benchmark run label such as "week3".
run_label = sys.argv[2] if len(sys.argv) > 2 else None

# Stop immediately with a clear error if the supplied derived image does not exist.
if not image_path.exists():
    raise FileNotFoundError(f"Image not found: {image_path}")

# Read the experimental condition from the image parent directory.
condition = image_path.parent.name.lower()

# Define the historical base output location for 7.89 cm/pixel DeepForest predictions.
output_root = Path("outputs/experiment_001/deepforest/gsd_7.89")

# Add the benchmark run label when one is explicitly supplied.
output_root = output_root / run_label if run_label else output_root

# Define the condition-specific prediction directory.
output_dir = output_root / condition

# Create the output directory and any missing parent directories.
output_dir.mkdir(parents=True, exist_ok=True)


# Create the DeepForest model wrapper.
model = main.deepforest()

# Load the pretrained individual-tree detection model.
model.load_model(model_name=MODEL_NAME, revision=MODEL_REVISION)

# Explicitly set the frozen model confidence threshold.
model.config.score_thresh = SCORE_THRESHOLD

# Explicitly set the frozen model non-maximum suppression threshold.
model.config.nms_thresh = NMS_THRESHOLD

# Run tiled inference using exactly the same DeepForest parameters used in the existing corrected-GSD baseline.
predictions = model.predict_tile(
    path=str(image_path),  # Supply the frozen 7.89 cm/pixel derived image.
    patch_size=PATCH_SIZE,  # Preserve the frozen 400-pixel patch size.
    patch_overlap=PATCH_OVERLAP,  # Preserve the frozen 5-percent overlap.
    iou_threshold=IOU_THRESHOLD,  # Preserve the frozen cross-tile IoU threshold.
    dataloader_strategy="single",  # Preserve the original single-image dataloader strategy.
)

# Define the raw prediction-table output path.
csv_path = output_dir / f"{image_path.stem}_predictions.csv"

# Save every retained DeepForest prediction without adding a pandas index.
predictions.to_csv(csv_path, index=False)

# Define the deterministic visualization filename stem.
plot_name = f"{image_path.stem}_predictions"

# Draw the retained detections on the exact 7.89 cm/pixel image used for inference.
plot_results(
    results=predictions,  # Supply the retained DeepForest prediction table.
    savedir=str(output_dir),  # Save the visualization inside the Week 3 output namespace.
    basename=plot_name,  # Preserve deterministic scene-based naming.
    image=str(image_path),  # Use the exact derived image analyzed by the model.
    show=False,  # Save without opening an interactive display.
)

# Print the analyzed derived image.
print(f"Image: {image_path}")

# Print the benchmark run label.
print(f"Run label: {run_label}")

# Print the number of retained DeepForest detections.
print(f"Detections: {len(predictions)}")

# Print the raw prediction-table path.
print(f"Prediction CSV: {csv_path}")

# Print the annotated visualization path.
print(f"Prediction image: {output_dir / f'{plot_name}.png'}")