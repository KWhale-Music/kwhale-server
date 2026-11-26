"""Queue helpers for the feature indexer.

Kept free of heavy imports (no Essentia) so the API process can enqueue work
without pulling the audio stack into its image.
"""

from __future__ import annotations

from ... import db

MODEL_VERSIONS = {
    "dsp": "essentia-2.1b6-v2",
    "tags": "tags-v1",
    "clap": "clap-music-1",
    "text": "giga-v2-rich",
    # Audio embedding: Z-scored full DSP vector (services/rec/audio_embed.py).
    "audio": "dsp-zscore-v2",
}

DEFAULT_KINDS = ("dsp", "tags", "audio", "text")


async def enqueue_track(
    track_id: int,
    kinds: tuple[str, ...] = DEFAULT_KINDS,
    *,
    force: bool = False,
) -> None:
    """Queue pipeline jobs for a track.

    `force=True` re-queues even `done` jobs: after a DSP re-index the tags and
    audio embedding were computed from stale features and must be rebuilt.
    """
    for kind in kinds:
        version = MODEL_VERSIONS.get(kind)
        if not version:
            continue
        await db.execute(
            """
            INSERT INTO index_jobs (track_id, kind, model_version)
            VALUES (:t, :k, :v)
            ON CONFLICT (track_id, kind, model_version) DO UPDATE
            SET status = 'pending',
                attempts = 0,
                last_error = NULL,
                updated_at = NOW()
            WHERE :force OR index_jobs.status = 'failed'
            """,
            {"t": track_id, "k": kind, "v": version, "force": force},
        )
