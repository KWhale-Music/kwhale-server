"""Audio-embedding backends for `track_emb_audio` (recommender side).

`track_emb_audio.vec` is `vector(512)` — the width reserved for a real audio
model (CLAP). Until such a model runs here, the default backend reuses the
normalized 17-dim DSP vector the indexer already computes (`feature_vector`)
and zero-pads it to the column width: cosine distance does not change under
zero-padding, so scores stay meaningful and no schema migration is needed.

Extension point for a real model (e.g. Discogs-EffNet via essentia-tensorflow):
implement `AudioEmbedBackend`, register an instance in `BACKENDS`, select it
with `AUDIO_EMBED_BACKEND=<name>`, and bump `MODEL_VERSIONS["audio"]` in
`queue.py` so old and new vectors never share a model_version. Heavy imports
(essentia, onnxruntime, torch) must happen inside `embed()`: only the indexer
process may load the audio stack, the API image stays light.
"""

from __future__ import annotations

import os
from typing import Any, Protocol

from .features import feature_vector, get_cached_stats

# Column width of track_emb_audio.vec in schema.sql (the planned CLAP model).
EMBED_DIM = 512


class AudioEmbedBackend(Protocol):
    """Turns one track into the vector stored in `track_emb_audio`."""

    async def embed(
        self, track_id: int, features: dict[str, Any] | None, file_path: str | None
    ) -> list[float] | None:
        """Track vector, or None when the inputs have not landed yet."""
        ...


class DspFeaturesBackend:
    """dsp-zscore-v2: Z-scored L2-normalized 22-dim DSP vector, zero-padded to 512."""

    async def embed(
        self, track_id: int, features: dict[str, Any] | None, file_path: str | None
    ) -> list[float] | None:
        return feature_vector(features, get_cached_stats())


BACKENDS: dict[str, AudioEmbedBackend] = {
    "dsp-zscore": DspFeaturesBackend(),
    "dsp17": DspFeaturesBackend(),
}
DEFAULT_BACKEND = "dsp-zscore"


def get_audio_backend(name: str | None = None) -> AudioEmbedBackend:
    key = (name or os.environ.get("AUDIO_EMBED_BACKEND") or DEFAULT_BACKEND).strip().lower()
    backend = BACKENDS.get(key)
    if backend is None:
        raise ValueError(f"unknown AUDIO_EMBED_BACKEND: {key}")
    return backend


def pad_to_column(vec: list[float]) -> list[float]:
    """Zero-pad a vector to the fixed column width (cosine-invariant)."""
    if len(vec) > EMBED_DIM:
        raise ValueError(f"audio embedding has {len(vec)} dims, column holds {EMBED_DIM}")
    return vec + [0.0] * (EMBED_DIM - len(vec))
