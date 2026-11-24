"""Clasificacion musical basada en caracteristicas de GTZAN.

El modelo recibe un archivo de audio, extrae caracteristicas espectrales y
devuelve un genero GTZAN junto con su preset sensorial equivalente.
"""
import re
from pathlib import Path

import joblib
import librosa
import numpy as np

from . import config

GENRE_TO_PRESET = {
    "blues": "bolero",
    "classical": "clasica",
    "country": "regional",
    "disco": "electronica",
    "hiphop": "reggaeton",
    "jazz": "jazz",
    "metal": "metal",
    "pop": "pop",
    "reggae": "cumbia",
    "rock": "rock",
}

SPECIAL_SONGS = (
    {
        "name": "Sembrando flores",
        "aliases": ("sembrando flores", "los cojolines", "cojolines"),
        "preset": "regional",
        "region": "Veracruz",
    },
)


def _normalize(text):
    text = text.lower()
    text = "".join(c for c in text if c.isalnum() or c.isspace())
    return re.sub(r"\s+", " ", text).strip()


def special_song_preset(title=None, filename=None):
    """Devuelve el preset especial conocido, sin modificar el modelo general."""
    text = _normalize(" ".join(v for v in (title, filename) if v))
    compact = text.replace(" ", "")
    for song in SPECIAL_SONGS:
        aliases = tuple(_normalize(v).replace(" ", "") for v in song["aliases"])
        if any(alias in compact for alias in aliases):
            return song["name"], song["preset"]
    return None, None


def extract_features(path):
    """Extrae un vector fijo para que el MLP pueda clasificar el audio."""
    y, sr = librosa.load(str(path), sr=22050, mono=True)
    if len(y) < sr:
        raise ValueError("El archivo de audio es demasiado corto")

    y = y[: sr * 30]
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    zcr = librosa.feature.zero_crossing_rate(y)

    features = []
    for matrix in (
        mfcc,
        chroma,
        spectral_centroid,
        spectral_bandwidth,
        spectral_rolloff,
        zcr,
    ):
        features.extend(np.mean(matrix, axis=1))
        features.extend(np.std(matrix, axis=1))
    vector = np.asarray(features, dtype=np.float32).reshape(1, -1)
    vector = np.nan_to_num(vector, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    return vector


def load_classifier(model_path=None):
    path = Path(model_path or config.MUSIC_MODEL_PATH)
    if not path.exists():
        raise FileNotFoundError(
            f"No existe el modelo musical en {path}. "
            "Entrenalo con: python -m src.train_music_classifier"
        )
    return joblib.load(path)


def classify_audio(path, title=None, model_path=None):
    special_name, special_preset = special_song_preset(
        title=title,
        filename=Path(path).name,
    )
    if special_name:
        return {
            "source": "especial",
            "matched_song": special_name,
            "gtzan_genre": "country",
            "preset": special_preset,
            "region": "Veracruz",
            "region": "Veracruz",
            "confidence": 1.0,
            "override": True,
        }

    bundle = load_classifier(model_path)
    features = extract_features(path)
    features = bundle["scaler"].transform(features)
    probabilities = bundle["model"].predict_proba(features)[0]
    index = int(np.argmax(probabilities))
    genre = bundle["classes"][index]
    confidence = float(probabilities[index])
    return {
        "source": "mlp",
        "gtzan_genre": genre,
        "preset": GENRE_TO_PRESET.get(genre, "pop"),
        "confidence": round(confidence, 3),
        "override": False,
    }
