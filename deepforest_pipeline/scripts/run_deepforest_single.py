# Import sys so the script can receive the image path and optional run label from the command line.
import sys

# Import Path so file and directory paths are handled cleanly across operating systems.
from pathlib import Path

# Import the main DeepForest model interface.
from deepforest import main

# Import DeepForest's visualization helper so prediction boxes can be drawn on the original image.
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


# Read the image path supplied as the first command-line argument.
image_path = Path(sys.argv[1])

# Read an optional benchmark run label such as "week3"; use no label when one is not supplied so historical behavior remains available.
run_label = sys.argv[2] if len(sys.argv) > 2 else None

# Stop immediately with a clear error if the supplied image does not exist.
if not image_path.exists():
    raise FileNotFoundError(f"Image not found: {image_path}")

# Read the experimental condition from the image parent directory, such as "before" or "after".
condition = image_path.parent.name.lower()

# Define the historical base output location for native-resolution DeepForest predictions.
output_root = Path("outputs/experiment_001/deepforest/single_image")

# Add the benchmark run label to the output hierarchy when one was explicitly supplied.
output_root = output_root / run_label if run_label else output_root

# Define the condition-specific output directory so BEFORE and AFTER outputs remain separated.
output_dir = output_root / condition

# Create the output directory and any missing parent directories.
output_dir.mkdir(parents=True, exist_ok=True)


# Create the DeepForest model wrapper.
model = main.deepforest()

# Load the pretrained individual-tree detection model.
model.load_model(model_name=MODEL_NAME, revision=MODEL_REVISION)

# Explicitly set the model confidence threshold used for retained predictions.
model.config.score_thresh = SCORE_THRESHOLD

# Explicitly set the model non-maximum suppression threshold.
model.config.nms_thresh = NMS_THRESHOLD

# Run tiled inference on the original image using the frozen native-resolution baseline parameters.
predictions = model.predict_tile(
    path=str(image_path),  # Supply the untouched benchmark image to DeepForest.
    patch_size=PATCH_SIZE,  # Preserve the frozen 400-pixel tile size.
    patch_overlap=PATCH_OVERLAP,  # Preserve the frozen 5-percent tile overlap.
    iou_threshold=IOU_THRESHOLD,  # Preserve the frozen cross-tile IoU suppression threshold.
    dataloader_strategy="single",  # Preserve the original single-image dataloader strategy.
)

# Create the output filename for the raw prediction table.
csv_path = output_dir / f"{image_path.stem}_predictions.csv"

# Save every retained DeepForest prediction without adding a pandas index column.
predictions.to_csv(csv_path, index=False)

# Create a filename stem for the annotated prediction image.
plot_name = f"{image_path.stem}_predictions"

# Draw the DeepForest predictions on the original image and save the visualization.
plot_results(
    results=predictions,  # Supply the frozen raw predictions to the visualization helper.
    savedir=str(output_dir),  # Save the visualization inside the week-specific benchmark directory.
    basename=plot_name,  # Preserve a deterministic filename based on the source image.
    image=str(image_path),  # Draw predictions over the exact benchmark image that was analyzed.
    show=False,  # Save the visualization without opening an interactive display.
)

# Print the exact benchmark image that was analyzed.
print(f"Image: {image_path}")

# Print the benchmark run label used for output separation.
print(f"Run label: {run_label}")

# Print the number of retained DeepForest detections.
print(f"Detections: {len(predictions)}")

# Print the location of the raw prediction CSV.
print(f"Prediction CSV: {csv_path}")

# Print the location of the annotated visualization.
print(f"Prediction image: {output_dir / f'{plot_name}.png'}")