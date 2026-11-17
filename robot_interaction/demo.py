"""Replay ordered MELD turns or process a supplied transcript and local audio."""
import argparse
import json
import time
from .data import load_rows, media_index
from .engine import Engine


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=["train", "dev", "test"], default="dev")
    parser.add_argument("--dialogue", type=int, default=0)
    parser.add_argument("--mode", choices=["text", "audio", "fusion"], default="fusion")
    parser.add_argument("--text")
    parser.add_argument("--audio")
    parser.add_argument("--pace", action="store_true", help="Wait each clip's duration before processing")
    args = parser.parse_args()
    engine = Engine()
    if args.text:
        print(json.dumps(engine.predict(args.text, args.audio, args.mode), ensure_ascii=False))
        return
    rows = load_rows(args.split)
    rows = rows[rows.Dialogue_ID == args.dialogue].sort_values("Utterance_ID")
    if rows.empty:
        parser.error("Dialogue does not exist")
    index = media_index()[args.split]
    for row in rows.itertuples():
        path = index.get(row.clip_id)
        if args.mode != "text" and path is None:
            print(json.dumps({"turn_id": row.clip_id, "error": "missing_audio"}), flush=True)
            continue
        if args.pace and path:
            from .audio import decode_audio, SAMPLE_RATE
            time.sleep(len(decode_audio(path))/SAMPLE_RATE)
        result = engine.predict(row.Utterance, path, args.mode, row.clip_id)
        print(json.dumps({"transcript": row.Utterance, **result}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
