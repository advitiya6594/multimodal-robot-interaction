"""Download official MELD and safely extract utterance clips, preserving splits."""
import argparse
from pathlib import Path
import tarfile
import urllib.request
import shutil

ROOT = Path(__file__).resolve().parents[1]
URL = "https://web.eecs.umich.edu/~mihalcea/downloads/MELD.Raw.tar.gz"


def extract_clips(archive, destination):
    destination.mkdir(parents=True, exist_ok=True)
    count = 0
    with tarfile.open(fileobj=archive, mode="r|gz") as tar:
        for member in tar:
            if member.isfile() and member.name.endswith(".mp4"):
                target = destination / Path(member.name).name
                with tar.extractfile(member) as source, target.open("wb") as output:
                    shutil.copyfileobj(source, output)
                count += 1
    print(destination.name, count, "clips", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", type=Path, default=ROOT / "data/raw/MELD.Raw.tar.gz")
    args = parser.parse_args()
    labels = ROOT / "data/labels"
    labels.mkdir(parents=True, exist_ok=True)
    for split in ("train", "dev", "test"):
        target = labels / f"{split}_sent_emo.csv"
        if not target.exists():
            urllib.request.urlretrieve("https://raw.githubusercontent.com/declare-lab/MELD/e8cedf27b5d2877e198332c957127e16eb214afe/data/MELD/" + target.name, target)
    if not args.archive.exists():
        args.archive.parent.mkdir(parents=True, exist_ok=True)
        partial = args.archive.with_suffix(".partial")
        print("Downloading 10.9 GB official media archive", flush=True)
        urllib.request.urlretrieve(URL, partial)
        partial.replace(args.archive)
    with tarfile.open(args.archive, "r|gz") as outer:
        for member in outer:
            name = Path(member.name).name
            split = next((s for s in ("train", "dev", "test") if name == s + ".tar.gz"), None)
            if split:
                print("Extracting", name, flush=True)
                # Streaming nested archives avoids another 11 GB copy on disk.
                extract_clips(outer.extractfile(member), ROOT / "data/media" / split)
    for split in ("train", "dev", "test"):
        if not list((ROOT / "data/media" / split).glob("*.mp4")):
            raise RuntimeError(f"No {split} media found. Inspect archive layout before training.")


if __name__ == "__main__":
    main()
