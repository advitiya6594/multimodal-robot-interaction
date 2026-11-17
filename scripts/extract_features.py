"""Cache deterministic features once, with an explicit missing/failed clip manifest."""
from concurrent.futures import ThreadPoolExecutor
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from robot_interaction.audio import decode_audio, features, FEATURE_VERSION
from robot_interaction.data import load_rows, media_index

ROOT = Path(__file__).resolve().parents[1]


def extract(item):
    clip_id, path = item
    try:
        return clip_id, features(decode_audio(path)), None
    except Exception as error:
        return clip_id, None, str(error)[:400]


def main():
    index = media_index(ROOT)
    manifest = {"feature_version": FEATURE_VERSION, "splits": {}}
    for split in ("train", "dev", "test"):
        rows = load_rows(split, ROOT)
        items = [(cid, index[split][cid]) for cid in rows.clip_id if cid in index[split]]
        missing = [cid for cid in rows.clip_id if cid not in index[split]]
        vectors, ids, errors = [], [], []
        with ThreadPoolExecutor(max_workers=4) as pool:
            for i, (cid, vector, error) in enumerate(pool.map(extract, items)):
                if error:
                    errors.append({"clip_id": cid, "error": error})
                else:
                    ids.append(cid)
                    vectors.append(vector)
                if (i+1) % 500 == 0:
                    print(split, i+1, "/", len(items), flush=True)
        if not vectors:
            raise RuntimeError(f"No usable {split} audio")
        np.savez_compressed(ROOT / f"data/{split}_features.npz", ids=np.array(ids), features=np.stack(vectors))
        manifest["splits"][split] = {"csv_rows": len(rows), "usable": len(ids), "missing": missing, "failed": errors}
        print(split, len(ids), "usable", flush=True)
    (ROOT / "reports/data_manifest.json").write_text(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
