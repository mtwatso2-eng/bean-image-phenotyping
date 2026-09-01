import os
from pathlib import Path

from dotenv import load_dotenv

APP_DIR = Path(__file__).parent
SAM3_REPO_ID = "facebook/sam3"
SAM3_FILENAME = "sam3.pt"

load_dotenv(APP_DIR / "tokens.env")

_model_path = None


def get_model_path() -> Path:
    """Return the cached SAM3 weights path, downloading from Hugging Face if needed."""
    global _model_path
    if _model_path is not None:
        return _model_path

    env_path = os.environ.get("SAM3_MODEL_PATH")
    if env_path:
        path = Path(env_path).expanduser()
        if not path.exists():
            raise FileNotFoundError(f"SAM3_MODEL_PATH does not exist: {path}")
        _model_path = path
        return _model_path

    local_path = APP_DIR / SAM3_FILENAME
    if local_path.exists():
        _model_path = local_path
        return _model_path

    from huggingface_hub import hf_hub_download

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    downloaded_path = hf_hub_download(
        repo_id=SAM3_REPO_ID,
        filename=SAM3_FILENAME,
        token=token,
    )
    _model_path = Path(downloaded_path)
    return _model_path
