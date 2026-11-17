"""MELD paths and split preserving manifests."""
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
LABELS = ["anger", "disgust", "fear", "joy", "neutral", "sadness", "surprise"]


def load_rows(split: str, root: Path = ROOT) -> pd.DataFrame:
    if split not in ("train", "dev", "test"):
        raise ValueError("Unknown split")
    rows = pd.read_csv(root / "data" / "labels" / f"{split}_sent_emo.csv")
    rows["clip_id"] = [f"dia{d}_utt{u}" for d, u in zip(rows.Dialogue_ID, rows.Utterance_ID)]
    return rows


def media_index(root: Path = ROOT) -> dict:
    index = {}
    for split in ("train", "dev", "test"):
        folder = root / "data" / "media" / split
        index[split] = {p.stem: p for p in folder.rglob("*.mp4")}
    return index
