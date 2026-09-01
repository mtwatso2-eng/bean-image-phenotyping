# bean-image-phenotyping

A Posit Shiny web app in Python for automated bean segmentation and phenotyping from images.

The app uses INT8-quantized SAM3 ONNX models from [rusen/sam3-browser-int8](https://huggingface.co/rusen/sam3-browser-int8) and measures width, height, aspect ratio, mean RGB, and color variance for each bean.

## Setup

1. Clone this repo.
2. Create a virtual environment with Python 3.10 or newer:
   ```bash
   python3.11 -m venv .venv
   source .venv/bin/activate
   ```
3. Install dependencies: `pip install -r requirements.txt`
4. Run the app: `shiny run app.py`

On first launch, the app downloads the ONNX models into the local Hugging Face cache (not into the repo). By default it uses the lower-memory `compact-448` variant for 4 GB hosts.

No Hugging Face token is required for the ONNX models.

## Deployment

- Do not commit model weights to GitHub.
- Default variant: `SAM3_ONNX_VARIANT=compact-448` (best for ~4 GB RAM).
- Limit input size on small instances: `MAX_IMAGE_DIMENSION=1024` (default).
- Higher quality on larger instances: `SAM3_ONNX_VARIANT=int8`.
- Optional: set `SAM3_ONNX_DIR` to a directory containing pre-downloaded ONNX files.

## Usage

Open the web app and select a folder of bean images to begin analysis. The app will produce a `bean_analysis.csv` file with per-bean measurements and display segmentation overlays as each image is processed.
