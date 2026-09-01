# bean-image-phenotyping

A Posit Shiny web app in Python for automated bean segmentation and phenotyping from images.

The app uses SAM3 semantic segmentation to detect beans and measures width, height, aspect ratio, mean RGB, and color variance for each bean.

## Setup

Requires **Python 3.10+** and **PyTorch 2.3+** for SAM3.

1. Clone this repo.
2. Create a virtual environment with Python 3.10 or newer:
   ```bash
   python3.11 -m venv .venv
   source .venv/bin/activate
   ```
3. Install dependencies: `pip install -r requirements.txt`
3. Request access to the SAM3 weights at [huggingface.co/facebook/sam3](https://huggingface.co/facebook/sam3).
4. Copy `tokens.env.example` to `tokens.env` and add your Hugging Face token:
   ```bash
   cp tokens.env.example tokens.env
   ```
   ```env
   HF_TOKEN=hf_your_token_here
   ```
   Quotes are not required around the token value.
5. Run the app: `shiny run app.py`

On first launch, the app downloads `sam3.pt` into the local Hugging Face cache (not into the repo). The same cached file is reused on later runs, and the model is loaded into memory when the server starts so inference is fast.

## Deployment

- Do not commit `sam3.pt` to GitHub.
- Set `HF_TOKEN` (or `HUGGINGFACE_HUB_TOKEN`) in your deployment environment.
- Optionally set `SAM3_MODEL_PATH` to a pre-downloaded weights file if your host provides persistent storage outside the app bundle.

## Usage

Open the web app and select a folder of bean images to begin analysis. The app will produce a `bean_analysis.csv` file with per-bean measurements and display segmentation overlays as each image is processed.
