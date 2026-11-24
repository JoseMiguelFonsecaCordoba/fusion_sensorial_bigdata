"""Entrena una MLP con el dataset GTZAN."""
import argparse
from pathlib import Path

import joblib
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder

from . import config
from .music_classifier import extract_features


def load_gtzan(root):
    root = Path(root)
    rows = []
    labels = []
    tracks = 0
    for genre_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        for path in sorted(genre_dir.iterdir()):
            if path.suffix.lower() not in (".wav", ".au", ".aiff", ".flac", ".ogg", ".mp3"):
                continue
            try:
                features = extract_features(path).reshape(-1)
                if features.shape != (72,) or not np.isfinite(features).all():
                    raise ValueError("caracteristicas invalidas")
                rows.append(features)
                labels.append(genre_dir.name)
                tracks += 1
            except Exception as exc:
                print(f"Se omite {path.name}: {exc}")
    if tracks < 20:
        raise ValueError(
            f"Solo se encontraron {tracks} archivos validos. "
            "Descarga GTZAN en data/gtzan/genres_original/"
        )
    return np.asarray(rows), np.asarray(labels), tracks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        default=config.ROOT / "data" / "gtzan" / "genres_original",
        type=Path,
    )
    parser.add_argument(
        "--model",
        default=config.MUSIC_MODEL_PATH,
        type=Path,
    )
    args = parser.parse_args()

    X, y, tracks = load_gtzan(args.dataset)
    if not np.isfinite(X).all():
        raise ValueError("La matriz GTZAN contiene valores no finitos")
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y)
    train_x, test_x, train_y, test_y = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    scaler = StandardScaler()
    train_x = scaler.fit_transform(train_x)
    test_x = scaler.transform(test_x)
    model = MLPClassifier(
        hidden_layer_sizes=(128, 64),
        activation="relu",
        solver="adam",
        alpha=1e-3,
        batch_size=32,
        learning_rate_init=1e-3,
        max_iter=300,
        early_stopping=True,
        validation_fraction=0.15,
        random_state=42,
    )
    model.fit(train_x, train_y)
    accuracy = model.score(test_x, test_y)

    args.model.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model": model,
            "scaler": scaler,
            "classes": list(label_encoder.classes_[model.classes_.astype(int)]),
        },
        args.model,
    )
    print(f"Archivos procesados: {tracks}")
    print(f"Precision de prueba: {accuracy:.1%}")
    print(f"Modelo guardado en: {args.model}")


if __name__ == "__main__":
    main()
