"""Fit on train, select fusion strength on dev, evaluate held-out test once."""
import json
import time

import joblib
import numpy as np
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from .data import ROOT, LABELS, load_rows
from .audio import FEATURE_VERSION


def read_split(split):
    rows = load_rows(split)
    with np.load(ROOT / f"data/{split}_features.npz") as cache:
        lookup = {cid: i for i, cid in enumerate(cache["ids"])}
        rows = rows[rows.clip_id.isin(lookup)].copy()
        acoustic = cache["features"][[lookup[cid] for cid in rows.clip_id]]
    return rows, acoustic


def classifier():
    return LogisticRegression(C=1.0, class_weight="balanced", max_iter=700, random_state=42)


def fuse(text, audio, weight):
    return hstack([text, csr_matrix(audio * weight)], format="csr")


def score(model, x, y):
    predicted = model.predict(x)
    return {
        "n": len(y), "accuracy": float(accuracy_score(y, predicted)),
        "macro_f1": float(f1_score(y, predicted, labels=LABELS, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y, predicted, labels=LABELS, average="weighted", zero_division=0)),
        "per_class": classification_report(y, predicted, labels=LABELS, output_dict=True, zero_division=0),
        "confusion_matrix": confusion_matrix(y, predicted, labels=LABELS).tolist(),
    }


def parameter_count(bundle):
    learned = {}
    for name in ("text", "audio", "fusion"):
        model = bundle[name]
        learned[name + "_classifier"] = int(model.coef_.size + model.intercept_.size)
    learned["tfidf_idf"] = int(bundle["vectorizer"].idf_.size)
    learned["audio_scaler_statistics"] = int(sum(getattr(bundle["scaler"], name).size for name in ("mean_", "var_", "scale_")))
    return {"components": learned, "total_required_for_demo": sum(learned.values()),
            "note": "Conservative count includes all three demo classifiers, learned IDF weights and scaler statistics. Acoustic extraction and response templates have zero learned parameters. Vocabulary strings are also stored in the artifact."}


def main():
    start = time.perf_counter()
    datasets = {split: read_split(split) for split in ("train", "dev", "test")}
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=20000,
                                 sublinear_tf=True, strip_accents="unicode")
    scaler = StandardScaler()
    train_rows, train_acoustic = datasets["train"]
    text = {"train": vectorizer.fit_transform(train_rows.Utterance.fillna(""))}
    audio = {"train": scaler.fit_transform(train_acoustic)}
    for split in ("dev", "test"):
        rows, acoustic = datasets[split]
        text[split] = vectorizer.transform(rows.Utterance.fillna(""))
        audio[split] = scaler.transform(acoustic)
    target = {split: rows.Emotion.to_numpy() for split, (rows, _) in datasets.items()}
    text_model = classifier().fit(text["train"], target["train"])
    audio_model = classifier().fit(audio["train"], target["train"])
    candidates, best_model, best_weight, best_score = [], None, None, -1
    for weight in (0.03, 0.1, 0.3, 1.0):
        model = classifier().fit(fuse(text["train"], audio["train"], weight), target["train"])
        value = f1_score(target["dev"], model.predict(fuse(text["dev"], audio["dev"], weight)), labels=LABELS, average="macro", zero_division=0)
        candidates.append({"audio_weight": weight, "dev_macro_f1": float(value)})
        print(candidates[-1], flush=True)
        if value > best_score:
            best_model, best_weight, best_score = model, weight, value
    bundle = {"vectorizer": vectorizer, "scaler": scaler, "text": text_model,
              "audio": audio_model, "fusion": best_model, "audio_weight": best_weight,
              "feature_version": FEATURE_VERSION, "seed": 42, "labels": LABELS}
    report = {"protocol": "Official MELD splits, usable audio intersection for every model. Training-only preprocessing. Fusion weight selected on dev macro-F1. No test tuning.",
              "labels": LABELS, "fusion_selection": candidates, "selected_audio_weight": best_weight,
              "counts": {s: len(r) for s, (r, _) in datasets.items()}, "dev": {}, "test": {}}
    for split in ("dev", "test"):
        report[split]["text"] = score(text_model, text[split], target[split])
        report[split]["audio"] = score(audio_model, audio[split], target[split])
        report[split]["fusion"] = score(best_model, fuse(text[split], audio[split], best_weight), target[split])
        majority = train_rows.Emotion.mode()[0]
        report[split]["majority"] = {"emotion": majority, "macro_f1": float(f1_score(target[split], [majority]*len(target[split]), labels=LABELS, average="macro", zero_division=0)), "accuracy": float(np.mean(target[split] == majority))}
    report["parameters"] = parameter_count(bundle)
    report["training_wall_seconds"] = time.perf_counter() - start
    ROOT.joinpath("artifacts").mkdir(exist_ok=True)
    joblib.dump(bundle, ROOT / "artifacts/model.joblib", compress=3)
    ROOT.joinpath("reports/evaluation.json").write_text(json.dumps(report, indent=2))
    # Per-turn outcomes support honest case studies, including mistakes.
    rows = datasets["test"][0]
    records = []
    predictions = {"text": text_model.predict(text["test"]), "audio": audio_model.predict(audio["test"]), "fusion": best_model.predict(fuse(text["test"], audio["test"], best_weight))}
    for i, row in enumerate(rows.itertuples()):
        records.append({"clip_id": row.clip_id, "dialogue_id": int(row.Dialogue_ID), "utterance_id": int(row.Utterance_ID),
                        "label": row.Emotion, **{k: str(v[i]) for k, v in predictions.items()}})
    ROOT.joinpath("reports/test_predictions.json").write_text(json.dumps(records, indent=2))
    print(json.dumps({k: {m: round(v["macro_f1"],4) for m,v in report[k].items()} for k in ("dev", "test")}), flush=True)


if __name__ == "__main__":
    with threadpool_limits(limits=1):
        main()
