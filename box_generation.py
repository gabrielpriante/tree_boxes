#!/usr/bin/env python3
"""
Tree detection on nadir RGB drone imagery using Claude Opus 5.5 (high effort).

Sends the image plus the annotation prompt to the Anthropic Messages API,
constrains the reply to the required JSON schema with structured outputs,
then renders the bounding boxes onto the original image.

Usage:
    # Put your key in .env next to this script: ANTHROPIC_API_KEY=sk-ant-...
    python tree_detection.py PT_R1.JPG --site "PT R1" --week "WEEK 1" --condition BEFORE

    # Re-render from a saved JSON without calling the API:
    python tree_detection.py PT_R1.JPG --from-json output/PT_R1_trees.json

Requirements:
    pip install anthropic pillow python-dotenv
"""

import argparse
import base64
import csv
import io
import json
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps

MODEL = "claude-opus-5-5"
EFFORT = "high"
MAX_TOKENS = 32000
MAX_IMAGE_BYTES = 5 * 1024 * 1024  # API per-image limit for base64 uploads
MAX_UPLOAD_EDGE = 2576             # longest edge sent to the API

PROMPT_TEMPLATE = """# Tree Detection Instructions

You are an expert tree annotator supporting an ecological and environmental monitoring team. You specialize in identifying individual trees from nadir RGB drone imagery.

You will receive one standardized RGB drone image.

## Image metadata

**Site ID:** {<<SITE>>}  
**Collection week:** {<<WEEK>>}  
**Condition:** {<<CONDITION>>}

The Site ID, collection week, and condition are provided only as identifiers for the image. They do not provide information about the expected number or location of trees.

## Task

Carefully inspect the entire image before producing your final response.

Identify every individual tree that is visually supported by the image.

Use exactly one axis-aligned bounding box for each individual tree.

Each bounding box should tightly enclose the visible canopy associated with that individual tree while excluding neighboring vegetation whenever the crowns can reasonably be separated.

When neighboring or overlapping crowns are visible, create separate bounding boxes only when the imagery provides reasonable visual evidence that multiple individual trees are present.

Do not invent additional trees simply because a canopy is large, irregular, dense, or complex.

If the imagery does not provide sufficient visual evidence to separate a canopy into multiple individual trees, annotate it as one tree rather than guessing an unsupported number of trees.

Annotate trees across the full image. Do not restrict your analysis to a particular portion of the image.

Do not annotate shrubs, herbaceous vegetation, buildings, vehicles, shadows, or other non-tree objects.

Do not use, assume, or infer an expected total number of trees.

Base every annotation only on evidence visible in the provided image.

## Bounding box coordinates

Represent each bounding box using a normalized coordinate system from **0 to 1000**.

The upper-left corner of the image is **(0, 0)**.

The lower-right corner of the image is **(1000, 1000)**.

Each bounding box must use this order:

**[x_min, y_min, x_max, y_max]**

All coordinate values must be integers from 0 through 1000.

For every bounding box:

**x_min < x_max**

**y_min < y_max**

## Verification

Before producing your final response, inspect the complete image again and verify that:

1. Every submitted bounding box corresponds to one visually supported individual tree.
2. No clearly visible tree has been omitted.
3. No tree has been annotated more than once.
4. No obvious non-tree object has been annotated as a tree.
5. Every bounding box is positioned on the tree it is intended to represent.
6. Every bounding box is as tight as reasonably possible around that individual tree's visible canopy.

## Output requirements

Return only valid JSON.

Do not include markdown.

Do not include explanations, commentary, qualifications, summaries, or any other text outside the JSON object.

Number detected trees sequentially beginning with **T001**.

Use exactly this schema:

```json
{
  "site": "{SITE_ID}",
  "week": "{WEEK}",
  "condition": "{BEFORE_OR_AFTER}",
  "trees": [
    {
      "tree_id": "T001",
      "bbox": [x_min, y_min, x_max, y_max]
    },
    {
      "tree_id": "T002",
      "bbox": [x_min, y_min, x_max, y_max]
    }
  ]
}
```

If no individual trees are visually supported by the image, return an empty **"trees"** array.

Do not return an annotated or regenerated image. The bounding boxes will be rendered onto the original image separately using the coordinates you provide."""

# JSON schema enforced through structured outputs (constrained decoding).
# Numeric range checks (0 to 1000, min < max) are validated in Python below.
OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "site": {"type": "string"},
        "week": {"type": "string"},
        "condition": {"type": "string"},
        "trees": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "tree_id": {"type": "string"},
                    "bbox": {"type": "array", "items": {"type": "integer"}},
                },
                "required": ["tree_id", "bbox"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["site", "week", "condition", "trees"],
    "additionalProperties": False,
}


def build_prompt(site: str, week: str, condition: str) -> str:
    return (
        PROMPT_TEMPLATE.replace("<<SITE>>", site)
        .replace("<<WEEK>>", week)
        .replace("<<CONDITION>>", condition)
    )


def load_image(path: Path) -> Image.Image:
    """Open the image and apply EXIF orientation so boxes match what the model sees."""
    img = Image.open(path)
    img = ImageOps.exif_transpose(img)
    return img.convert("RGB")


def encode_for_api(img: Image.Image) -> str:
    """JPEG encode, downscaling if needed to stay under API limits.
    Normalized 0 to 1000 coordinates are scale invariant, so this is safe."""
    work = img.copy()
    if max(work.size) > MAX_UPLOAD_EDGE:
        work.thumbnail((MAX_UPLOAD_EDGE, MAX_UPLOAD_EDGE), Image.LANCZOS)
    for quality in (95, 90, 85, 80, 70):
        buf = io.BytesIO()
        work.save(buf, format="JPEG", quality=quality)
        if buf.tell() <= MAX_IMAGE_BYTES:
            return base64.standard_b64encode(buf.getvalue()).decode("ascii")
    raise ValueError("Could not encode image under the 5 MB API limit.")


def call_claude(img: Image.Image, prompt: str) -> dict:
    import anthropic
    from dotenv import load_dotenv

    # Load ANTHROPIC_API_KEY from the .env file next to this script.
    load_dotenv(Path(__file__).resolve().parent / ".env")
    client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY
    image_b64 = encode_for_api(img)

    # Streaming avoids HTTP timeouts on long, high effort requests.
    with client.messages.stream(
        model=MODEL,
        max_tokens=MAX_TOKENS,
        thinking={"type": "adaptive"},
        output_config={
            "effort": EFFORT,
            "format": {"type": "json_schema", "schema": OUTPUT_SCHEMA},
        },
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/jpeg",
                            "data": image_b64,
                        },
                    },
                    {"type": "text", "text": prompt},
                ],
            }
        ],
    ) as stream:
        message = stream.get_final_message()

    if message.stop_reason == "max_tokens":
        raise RuntimeError("Response hit max_tokens before finishing; raise MAX_TOKENS.")
    if message.stop_reason == "refusal":
        raise RuntimeError("The model declined the request.")

    text = "".join(b.text for b in message.content if b.type == "text").strip()
    print(
        f"Usage: input={message.usage.input_tokens} output={message.usage.output_tokens}",
        file=sys.stderr,
    )
    return json.loads(text)


def validate_trees(result: dict) -> list:
    """Keep well formed boxes, clamp to 0..1000, drop degenerate ones."""
    clean = []
    for t in result.get("trees", []):
        bbox = t.get("bbox", [])
        if len(bbox) != 4:
            print(f"Skipping {t.get('tree_id')}: bbox has {len(bbox)} values", file=sys.stderr)
            continue
        x0, y0, x1, y1 = (max(0, min(1000, int(v))) for v in bbox)
        if x0 >= x1 or y0 >= y1:
            print(f"Skipping {t.get('tree_id')}: degenerate bbox {bbox}", file=sys.stderr)
            continue
        clean.append({"tree_id": t["tree_id"], "bbox": [x0, y0, x1, y1]})
    return clean


def draw_boxes(img: Image.Image, trees: list) -> Image.Image:
    out = img.copy()
    draw = ImageDraw.Draw(out)
    w, h = out.size
    line_w = max(2, round(min(w, h) / 300))
    font_size = max(14, round(min(w, h) / 55))
    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", font_size)
    except OSError:
        font = ImageFont.load_default()

    box_color = (255, 0, 255)  # magenta stands out against canopy green
    for t in trees:
        x0, y0, x1, y1 = t["bbox"]
        px = [x0 * w / 1000, y0 * h / 1000, x1 * w / 1000, y1 * h / 1000]
        draw.rectangle(px, outline=box_color, width=line_w)

        label = t["tree_id"]
        tb = draw.textbbox((0, 0), label, font=font)
        tw, th = tb[2] - tb[0], tb[3] - tb[1]
        pad = line_w
        lx = px[0]
        ly = px[1] - th - 2 * pad if px[1] - th - 2 * pad >= 0 else px[1]
        draw.rectangle([lx, ly, lx + tw + 2 * pad, ly + th + 2 * pad], fill=box_color)
        draw.text((lx + pad, ly + pad - tb[1]), label, fill=(255, 255, 255), font=font)
    return out


COUNT_FIELDS = ["image", "site", "week", "condition", "tree_count"]


def update_counts_csv(csv_path: Path, image_name: str, result: dict) -> None:
    """Record the tree count for this image, replacing any earlier row for it."""
    rows = []
    if csv_path.exists():
        with csv_path.open(newline="") as f:
            rows = [r for r in csv.DictReader(f) if r.get("image") != image_name]
    rows.append({
        "image": image_name,
        "site": result.get("site", ""),
        "week": result.get("week", ""),
        "condition": result.get("condition", ""),
        "tree_count": len(result["trees"]),
    })
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COUNT_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("image", type=Path)
    ap.add_argument("--site", default="PT R1")
    ap.add_argument("--week", default="WEEK 1")
    ap.add_argument("--condition", default="BEFORE")
    ap.add_argument("--output-dir", type=Path, default=Path("output"))
    ap.add_argument("--from-json", type=Path, help="Skip the API and render an existing result JSON")
    args = ap.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    img = load_image(args.image)
    stem = args.image.stem

    if args.from_json:
        result = json.loads(args.from_json.read_text())
    else:
        prompt = build_prompt(args.site, args.week, args.condition)
        print(f"Calling {MODEL} (effort={EFFORT}) on {args.image.name} ...", file=sys.stderr)
        result = call_claude(img, prompt)

    result["trees"] = validate_trees(result)

    json_path = args.output_dir / f"{stem}_trees.json"
    json_path.write_text(json.dumps(result, indent=2))

    annotated = draw_boxes(img, result["trees"])
    img_path = args.output_dir / f"{stem}_annotated.jpg"
    annotated.save(img_path, quality=92)

    csv_path = args.output_dir / "tree_counts.csv"
    update_counts_csv(csv_path, args.image.name, result)

    print(f"{len(result['trees'])} trees")
    print(f"JSON:  {json_path}")
    print(f"Image: {img_path}")
    print(f"CSV:   {csv_path}")


if __name__ == "__main__":
    main()