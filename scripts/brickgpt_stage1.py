"""Local BrickGPT setup checks and baseline inference; no ICL experiments."""
import importlib.metadata
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'temp' / 'runtime'))
sys.path.insert(0, str(ROOT / 'BrickGPT' / 'src'))
os.environ.setdefault('HF_HUB_CACHE', str(ROOT / 'temp' / 'hf-cache' / 'hub'))
os.environ.setdefault('HF_XET_CACHE', str(ROOT / 'temp' / 'hf-cache' / 'xet'))
os.environ.setdefault('MPLCONFIGDIR', str(ROOT / 'temp' / 'matplotlib'))


def resolve_base_model(path):
    if path:
        return Path(path).resolve()
    manual = ROOT / 'temp/models/Llama-3.2-1B-Instruct'
    if (manual / 'config.json').exists() and list(manual.glob('*.safetensors')):
        return manual
    from huggingface_hub import snapshot_download
    return Path(snapshot_download('meta-llama/Llama-3.2-1B-Instruct', local_files_only=True))
