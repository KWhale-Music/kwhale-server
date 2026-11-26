"""Playlist sequencing: artist/album spacing and smooth energy/BPM arc ordering."""

from __future__ import annotations

import math

from .features import cosine
from .scoring import Scored, octave_bpm_diff


def _arc_energy(progress: float, arc: str, prev_energy: float) -> float:
    if arc == "build_up":
        return 0.40 + 0.50 * progress
    if arc == "wind_down":
        return 0.72 - 0.45 * progress
    if arc == "wave":
        return 0.54 + 0.20 * math.sin(progress * math.pi * 2.5)
    return prev_energy


def _artists_of(item: Scored) -> set[str]:
    cand = item.candidate
    if cand.artists:
        out = {a.strip().lower() for a in cand.artists if a and a.strip()}
        if out:
            return out
    if cand.artist and cand.artist.strip():
        return {cand.artist.strip().lower()}
    return set()


def sequence_playlist(
    items: list[Scored],
    arc: str = "wave",
    seed: int = 0,
    artist_window: int = 3,
    album_window: int = 2,
) -> list[Scored]:
    """Order already-selected playlist tracks for cohesive sonic flow and separation.

    Preserves the exact multiset of `items`:
      * Slot 0 starts strong (from top 3 highest-scored candidates, seeded lightly)
      * Penalizes back-to-back or nearby repeats of the same artist/album
      * Smooths step-to-step energy/BPM transitions while following `arc`
    """
    n = len(items)
    if n <= 2:
        return list(items)

    pool = list(items)
    # Pick a strong opener from top candidates within a narrow relevance margin
    top_k = min(3, n)
    top_score = pool[0].score
    close_top = [i for i in range(top_k) if top_score - pool[i].score <= 0.04]
    opener_idx = close_top[seed % len(close_top)] if len(close_top) > 1 else 0

    ordered: list[Scored] = [pool.pop(opener_idx)]

    while pool:
        pos = len(ordered)
        progress = pos / max(1, n - 1)
        prev = ordered[-1]
        prev_energy = prev.candidate.energy if prev.candidate.energy is not None else 0.52
        prev_bpm = prev.candidate.bpm
        target_e = _arc_energy(progress, arc, prev_energy)

        recent_artists: list[set[str]] = [_artists_of(x) for x in ordered[-artist_window:]]
        recent_albums: list[str] = [
            (x.candidate.album or "").strip().lower()
            for x in ordered[-album_window:]
            if x.candidate.album
        ]

        best_idx = 0
        best_val = -1e9
        for idx, cand_item in enumerate(pool):
            cand = cand_item.candidate
            c_artists = _artists_of(cand_item)
            c_album = (cand.album or "").strip().lower()

            val = 0.85 * cand_item.score

            # Artist separation
            for back_i, prev_set in enumerate(reversed(recent_artists), start=1):
                if c_artists & prev_set:
                    if back_i == 1:
                        val -= 50.0
                    else:
                        val -= 12.0 / back_i
                    break

            # Album separation
            if c_album and c_album in recent_albums:
                val -= 4.0

            # Energy continuity + arc alignment
            if cand.energy is not None:
                e = float(cand.energy)
                val -= 0.45 * abs(e - prev_energy)
                val -= 0.35 * abs(e - target_e)

            # Octave-aware BPM smoothness
            if prev_bpm is not None and cand.bpm is not None:
                diff = octave_bpm_diff(float(cand.bpm), float(prev_bpm))
                if diff > 15.0:
                    val -= 0.25 * min(1.0, (diff - 15.0) / 45.0)

            if val > best_val:
                best_val = val
                best_idx = idx

        ordered.append(pool.pop(best_idx))

    return ordered


def _pick_closest_bridge(
    anchor_or_prev: Scored,
    pool: list[Scored],
    recent_artists: list[set[str]],
    recent_albums: list[str],
    rng_bias: float = 0.0,
) -> int:
    prev_vec = anchor_or_prev.candidate.vec
    prev_e = anchor_or_prev.candidate.energy if anchor_or_prev.candidate.energy is not None else 0.52
    prev_bpm = anchor_or_prev.candidate.bpm

    best_idx = 0
    best_val = -1e9
    for idx, item in enumerate(pool):
        c = item.candidate
        c_artists = _artists_of(item)
        c_album = (c.album or "").strip().lower()
        val = 0.65 * item.score

        if prev_vec and c.vec:
            sim = cosine(prev_vec, c.vec)
            if sim is not None:
                val += 0.35 * ((sim + 1.0) / 2.0)

        for back_i, prev_set in enumerate(reversed(recent_artists), start=1):
            if c_artists & prev_set:
                val -= 50.0 if back_i == 1 else (12.0 / back_i)
                break
        if c_album and c_album in recent_albums:
            val -= 4.0

        if c.energy is not None:
            val -= 0.25 * abs(float(c.energy) - prev_e)
        if prev_bpm is not None and c.bpm is not None:
            diff = octave_bpm_diff(float(c.bpm), float(prev_bpm))
            if diff > 15.0:
                val -= 0.15 * min(1.0, (diff - 15.0) / 45.0)

        # Tiny tie-break preference toward earlier sampled candidates
        val -= rng_bias * idx
        if val > best_val:
            best_val = val
            best_idx = idx
    return best_idx


def sequence_discovery(
    items: list[Scored],
    seed: int = 0,
    *,
    anchor_ratio: float = 0.25,
    bridge_len: int = 3,
    artist_window: int = 3,
    album_window: int = 2,
) -> list[Scored]:
    """Interleave familiar anchors (~25%) followed by acoustically adjacent discoveries."""
    # Recently played tracks are eligible (no hard wall) but never lead a shelf.
    recent = [item for item in items if item.reason_code == "recent_play"]
    pool = [item for item in items if item.reason_code != "recent_play"]
    n = len(pool)
    if n <= 3:
        return list(pool) + recent

    anchors: list[Scored] = []
    discoveries: list[Scored] = []
    for item in pool:
        c = item.candidate
        # A like without a play is not a familiar anchor: large favorite
        # libraries would otherwise turn the whole shelf into "anchors".
        if item.reason_code == "familiar_anchor" and c.plays > 0:
            anchors.append(item)
        else:
            discoveries.append(item)

    max_anchors = min(len(anchors), max(1, round(n * anchor_ratio))) if anchors and discoveries else 0
    if max_anchors == 0:
        return sequence_playlist(pool, arc="wave", seed=seed) + recent

    # Extra unused anchors still belong in the overflow tail only after discoveries.
    selected_anchors = list(anchors[:max_anchors])
    extra_anchors = list(anchors[max_anchors:])

    ordered: list[Scored] = []
    while len(ordered) < n and (selected_anchors or discoveries):
        recent_artists = [_artists_of(x) for x in ordered[-artist_window:]]
        recent_albums = [
            (x.candidate.album or "").strip().lower()
            for x in ordered[-album_window:]
            if x.candidate.album
        ]

        if selected_anchors and discoveries:
            # Start each 4-track mini-arc with a familiar anchor that avoids artist clashes
            a_idx = 0
            for idx, anc in enumerate(selected_anchors):
                if not any(_artists_of(anc) & prev for prev in recent_artists):
                    a_idx = idx
                    break
            anchor = selected_anchors.pop(a_idx)
            ordered.append(anchor)
            anchor_artist = anchor.candidate.artist or ""

            for _ in range(bridge_len):
                if not discoveries:
                    break
                r_art = [_artists_of(x) for x in ordered[-artist_window:]]
                r_alb = [
                    (x.candidate.album or "").strip().lower()
                    for x in ordered[-album_window:]
                    if x.candidate.album
                ]
                d_idx = _pick_closest_bridge(ordered[-1], discoveries, r_art, r_alb, rng_bias=0.002)
                picked = discoveries.pop(d_idx)
                if picked.reason_code != "rediscover":
                    picked.reason_code = "bridge"
                    picked.reason_detail = anchor_artist
                ordered.append(picked)
        elif discoveries:
            d_idx = _pick_closest_bridge(ordered[-1], discoveries, recent_artists, recent_albums, rng_bias=0.002) if ordered else 0
            ordered.append(discoveries.pop(d_idx))
        else:
            ordered.append(selected_anchors.pop(0))

    return ordered + extra_anchors + recent
