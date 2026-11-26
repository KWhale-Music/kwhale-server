"""Query-aware candidate generation (SQL + pgvector HNSW, per-artist capped)."""

from __future__ import annotations

import logging
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any

from ... import db
from ..genres import family_set, genre_families
from .audio_embed import pad_to_column
from .features import cosine, feature_vector, load_corpus_stats
from .scoring import Candidate

if TYPE_CHECKING:
    from .llm import QueryPlan

log = logging.getLogger("kwhale.rec.candidates")

LOCAL = """
    EXISTS (SELECT 1 FROM track_sources s
            WHERE s.track_id = t.id AND s.connector = 'local' AND s.available)
    AND NOT EXISTS (SELECT 1 FROM dislikes d WHERE d.user_id = :u AND d.track_id = t.id)
    AND NOT EXISTS (SELECT 1 FROM track_quality q WHERE q.track_id = t.id AND q.error IS NOT NULL)
"""


def _vec_literal(vec: list[float]) -> str:
    return "[" + ",".join(f"{v:.6f}" for v in vec) + "]"


async def _ids(sql: str, params: dict[str, Any]) -> list[int]:
    rows = await db.fetch_all(sql, params)
    return [int(r["id"]) for r in rows if r.get("id") is not None]


def _artist_affinity(artists: list[str], artist_map: dict[str, float]) -> float | None:
    if not artists or not artist_map:
        return None
    vals: list[float] = []
    for idx, name in enumerate(artists):
        key = name.strip().lower()
        if key in artist_map:
            vals.append(artist_map[key] * (1.0 if idx == 0 else 0.7))
    return max(vals) if vals else None


def _genre_affinity(families: list[str], genre_map: dict[str, float]) -> float | None:
    values = [genre_map[family] for family in families if family in genre_map]
    return max(values) if values else None


async def hydrate(
    user_id: int,
    profile: dict[str, Any],
    ids: list[int],
    *,
    sem_sims: dict[int, float] | None = None,
    ref_sims: dict[int, float] | None = None,
) -> list[Candidate]:
    """Load track metadata, DSP features, semantic tags, and per-user stats."""
    if not ids:
        return []

    corpus_stats = await load_corpus_stats()
    rows = await db.fetch_all(
        """
        SELECT t.id,
               t.artists,
               t.artists[1] AS artist,
               t.album,
               t.year,
               t.genre,
               t.genres_norm,
               tf.bpm,
               tf.beat_confidence,
               tf.loudness_lufs,
               tf.dynamic_complexity,
               tf.danceability,
               tf.onset_rate,
               tf.spectral_centroid,
               tf.spectral_rolloff,
               tf.spectral_flux,
               tf.mfcc_mean,
               tf.energy,
               tf.valence,
               tf.dsp_extra,
               tf.subgenres,
               tf.moods,
               tf.textures,
               tf.scenes,
               tf.language,
               tf.vibe_summary,
               tf.acousticness,
               tf.instrumentalness,
               (tf.track_id IS NOT NULL) AS has_tf,
               COALESCE(st.plays, 0)     AS plays,
               COALESCE(st.completes, 0) AS completes,
               COALESCE(st.skips, 0)     AS skips,
               COALESCE(st.repeats, 0)   AS repeats,
               st.last_played,
               (f.track_id IS NOT NULL)  AS liked
        FROM tracks t
        LEFT JOIN track_features tf ON tf.track_id = t.id
        LEFT JOIN mv_user_track_stats st ON st.user_id = :u AND st.track_id = t.id
        LEFT JOIN favorites f ON f.user_id = :u AND f.track_id = t.id
        WHERE t.id = ANY(:ids)
        """,
        {"u": user_id, "ids": ids},
    )

    artist_map = profile.get("artists") or {}
    genre_map = profile.get("genres") or {}
    sem_map = sem_sims or {}
    ref_map = ref_sims or {}

    by_id: dict[int, Candidate] = {}
    for row in rows:
        tid = int(row["id"])
        artists = [str(a).strip() for a in (row.get("artists") or []) if str(a).strip()]
        artist = (row.get("artist") or (artists[0] if artists else "")).strip()
        genres = list(row.get("genres_norm") or [])
        if not genres:
            genres = genre_families(row.get("genre"))

        acousticness = row.get("acousticness")
        if acousticness is None and isinstance(row.get("dsp_extra"), dict):
            acousticness = row["dsp_extra"].get("acoustic_proxy")

        subgenres = list(row.get("subgenres") or [])
        moods = list(row.get("moods") or [])
        textures = list(row.get("textures") or [])
        scenes = list(row.get("scenes") or [])
        has_features = bool(
            row.get("has_tf")
            and (row.get("energy") is not None or row.get("bpm") is not None or subgenres or moods)
        )

        by_id[tid] = Candidate(
            track_id=tid,
            artist=artist or None,
            artists=artists or None,
            album=(row.get("album") or "").strip() or None,
            year=int(row["year"]) if row.get("year") is not None else None,
            genre=(row.get("genre") or "").strip() or None,
            genres=genres,
            vec=feature_vector(row, corpus_stats),
            energy=float(row["energy"]) if row.get("energy") is not None else None,
            valence=float(row["valence"]) if row.get("valence") is not None else None,
            bpm=float(row["bpm"]) if row.get("bpm") is not None else None,
            danceability=float(row["danceability"]) if row.get("danceability") is not None else None,
            acousticness=float(acousticness) if acousticness is not None else None,
            instrumentalness=(
                float(row["instrumentalness"])
                if row.get("instrumentalness") is not None
                else None
            ),
            subgenres=subgenres or None,
            moods=moods or None,
            textures=textures or None,
            scenes=scenes or None,
            language=row.get("language"),
            vibe_summary=row.get("vibe_summary"),
            has_features=has_features,
            plays=int(row["plays"] or 0),
            completes=int(row["completes"] or 0),
            skips=int(row["skips"] or 0),
            repeats=int(row["repeats"] or 0),
            liked=bool(row.get("liked")),
            last_played=row.get("last_played"),
            artist_affinity=_artist_affinity(artists, artist_map),
            genre_affinity=_genre_affinity(genres, genre_map),
            sem_query_sim=sem_map.get(tid),
            ref_sim=ref_map.get(tid),
        )

    return [by_id[tid] for tid in ids if tid in by_id]


async def _audio_vec_neighbors(
    user_id: int,
    vec: list[float] | None,
    limit: int = 150,
    *,
    exclude_track_id: int | None = None,
    exclude_genres: list[str] | None = None,
) -> list[int]:
    if not vec:
        return []
    padded = _vec_literal(pad_to_column(vec))
    exc_clause = "AND NOT (t.genres_norm && :exc)" if exclude_genres else ""
    tid_clause = "AND t.id <> :ex_tid" if exclude_track_id is not None else ""
    return await _ids(
        f"""
        SELECT e.track_id AS id
        FROM track_emb_audio e
        JOIN tracks t ON t.id = e.track_id
        WHERE {LOCAL} {exc_clause} {tid_clause}
        ORDER BY e.vec <=> CAST(:v AS vector)
        LIMIT :l
        """,
        {
            "u": user_id,
            "v": padded,
            "l": limit,
            "exc": exclude_genres or [],
            "ex_tid": exclude_track_id,
        },
    )


async def _text_vec_neighbors(
    user_id: int,
    vec: list[float] | None,
    limit: int = 200,
    *,
    exclude_track_id: int | None = None,
    exclude_genres: list[str] | None = None,
) -> dict[int, float]:
    if not vec:
        return {}
    lit = _vec_literal(vec)
    exc_clause = "AND NOT (t.genres_norm && :exc)" if exclude_genres else ""
    tid_clause = "AND t.id <> :ex_tid" if exclude_track_id is not None else ""
    rows = await db.fetch_all(
        f"""
        SELECT e.track_id AS id, (1.0 - (e.vec <=> CAST(:v AS vector))) AS sim
        FROM track_emb_text e
        JOIN tracks t ON t.id = e.track_id
        WHERE {LOCAL} {exc_clause} {tid_clause}
        ORDER BY e.vec <=> CAST(:v AS vector)
        LIMIT :l
        """,
        {
            "u": user_id,
            "v": lit,
            "l": limit,
            "exc": exclude_genres or [],
            "ex_tid": exclude_track_id,
        },
    )
    return {
        int(r["id"]): float(r["sim"])
        for r in rows
        if r.get("id") is not None and r.get("sim") is not None
    }


async def _cooc_map(user_id: int, seeds: list[int], limit: int = 120) -> dict[int, float]:
    if not seeds:
        return {}
    rows = await db.fetch_all(
        f"""
        SELECT q.id, SUM(q.w) AS w
        FROM (
            SELECT c.track_b AS id, c.weight AS w
            FROM mv_item_cooccurrence c
            WHERE c.track_a = ANY(:seeds)
            UNION ALL
            SELECT c.track_a AS id, c.weight AS w
            FROM mv_item_cooccurrence c
            WHERE c.track_b = ANY(:seeds)
        ) q
        JOIN tracks t ON t.id = q.id
        WHERE {LOCAL} AND NOT (t.id = ANY(:seeds))
        GROUP BY q.id
        ORDER BY w DESC
        LIMIT :l
        """,
        {"u": user_id, "seeds": seeds[:50], "l": limit},
    )
    if not rows:
        return {}
    peak = max(float(r["w"]) for r in rows) or 1.0
    return {int(r["id"]): min(1.0, float(r["w"]) / peak) for r in rows}


_FAMILY_SUBGENRES: dict[str, list[str]] = {
    "rap": ["rap", "russian_rap", "trap", "drill", "phonk", "dark_phonk", "boom_bap", "jazz_rap", "acoustic_rap", "cloud_rap", "pop_rap", "melodic_rap"],
    "jazz": ["jazz", "nu_jazz", "smooth_jazz", "cool_jazz", "bebop", "swing", "acid_jazz", "bossa_nova"],
    "folk": ["folk", "acoustic_folk", "indie_folk", "chanson", "acoustic", "singer_songwriter"],
    "rock": ["rock", "indie_rock", "alt_rock", "post_punk", "punk", "metal", "grunge", "shoegaze"],
    "pop": ["pop", "indie_pop", "synthpop", "dance_pop", "art_pop", "bedroom_pop", "dream_pop", "chillwave"],
    "electronic": ["electronic", "house", "techno", "dnb", "ambient", "synthwave", "retrowave", "downtempo", "trip_hop", "lo_fi_beats", "chillwave"],
    "rnb": ["rnb", "neo_soul", "soul", "funk"],
    "classical": ["classical", "neoclassical", "orchestral", "piano"],
    "soundtrack": ["soundtrack", "score", "orchestral"],
}


def _expand_family_subgenres(families: list[str], extra_subs: list[str] | None = None) -> list[str]:
    out: set[str] = set(extra_subs or [])
    for fam in families:
        out.add(fam)
        out.update(_FAMILY_SUBGENRES.get(fam, ()))
    return sorted(out)


async def _collect_query_driven(
    user_id: int,
    profile: dict[str, Any],
    plan: QueryPlan,
    query_vec: list[float] | None,
    *,
    limit: int = 600,
    seed: int = 0,
) -> tuple[list[int], dict[int, float], dict[int, float]]:
    ids: list[int] = []
    seen: set[int] = set()
    sem_sims: dict[int, float] = {}
    ref_sims: dict[int, float] = {}

    def _extend(chunk: list[int]) -> None:
        for tid in chunk:
            if tid not in seen:
                seen.add(tid)
                ids.append(tid)

    exc_genres = list(plan.exclude_genres or [])
    # Hard exclusions are applied later in Python (mixes._apply_filters), where
    # "без рэпа" can keep jazz-rap. Filtering by raw genres here would drop
    # jazz-rap before the pure-rap exemption ever runs.
    exc_clause = ""
    inc_subs = _expand_family_subgenres(list(plan.include_genres or []), list(plan.target_subgenres or [])) if plan.include_genres else []

    # 1. Giga-Embeddings semantic HNSW search
    if query_vec:
        sem_map = await _text_vec_neighbors(user_id, query_vec, limit=240)
        sem_sims.update(sem_map)
        _extend(list(sem_map.keys()))

    # 2. Structured Tag + Genre + Year GIN/attribute match (per-artist capped!)
    has_tag_filters = any(
        (
            plan.include_genres,
            plan.target_subgenres,
            plan.target_moods,
            plan.target_textures,
            plan.target_scenes,
            plan.target_language,
            plan.year_min,
            plan.year_max,
        )
    )
    if has_tag_filters:
        lang_clause = "AND (tf.language = :lang OR tf.language IS NULL)" if plan.target_language else ""
        yr_lo_clause = "AND (t.year >= :ymin OR t.year IS NULL)" if plan.year_min else ""
        yr_hi_clause = "AND (t.year <= :ymax OR t.year IS NULL)" if plan.year_max else ""

        tag_terms: list[str] = []
        tag_params: dict[str, Any] = {"u": user_id, "exc": exc_genres, "seed": str(seed)}
        if plan.target_subgenres:
            tag_terms.append("CASE WHEN tf.subgenres && :subs THEN 3 ELSE 0 END")
            tag_params["subs"] = list(plan.target_subgenres)
        if plan.target_textures:
            tag_terms.append("CASE WHEN tf.textures && :texs THEN 3 ELSE 0 END")
            tag_params["texs"] = list(plan.target_textures)
        if plan.target_scenes:
            tag_terms.append("CASE WHEN tf.scenes && :scns THEN 2 ELSE 0 END")
            tag_params["scns"] = list(plan.target_scenes)
        if plan.target_moods:
            tag_terms.append("CASE WHEN tf.moods && :mods THEN 2 ELSE 0 END")
            tag_params["mods"] = list(plan.target_moods)
        if plan.include_genres:
            tag_terms.append("CASE WHEN (t.genres_norm && :inc OR tf.subgenres && :inc_subs) THEN 2 ELSE 0 END")
            tag_params["inc"] = list(plan.include_genres)
            tag_params["inc_subs"] = inc_subs
        if plan.target_language:
            tag_terms.append("CASE WHEN tf.language = :lang THEN 1 ELSE 0 END")
            tag_params["lang"] = plan.target_language
        if plan.year_min:
            tag_params["ymin"] = plan.year_min
        if plan.year_max:
            tag_params["ymax"] = plan.year_max

        tag_expr = " + ".join(tag_terms) if tag_terms else "0"
        # Always require a tag hit.  The old genre-only branch dropped
        # jazz-rap from a "джаз" request because its genre says "rap" while
        # its texture says "jazz_samples".
        having_filter = "WHERE tag_hits > 0"

        tag_rows = await _ids(
            f"""
            WITH scored AS (
                SELECT t.id,
                       lower(COALESCE(t.artists[1], '')) AS art,
                       ({tag_expr}) AS tag_hits,
                       md5(t.id::text || ':' || :seed) AS rng
                FROM tracks t
                LEFT JOIN track_features tf ON tf.track_id = t.id
                WHERE {LOCAL} {exc_clause} {lang_clause} {yr_lo_clause} {yr_hi_clause}
            ),
            capped AS (
                SELECT id, tag_hits, rng,
                       ROW_NUMBER() OVER (PARTITION BY art ORDER BY tag_hits DESC, rng) AS rn
                FROM scored
                {having_filter}
            )
            SELECT id FROM capped
            WHERE rn <= 5
            ORDER BY tag_hits DESC, rng
            LIMIT 250
            """,
            tag_params,
        )
        _extend(tag_rows)

    # 3. Audio Target distance retrieval (only tracks that actually have features!)
    if plan.audio_target or plan.hard_bounds:
        t_energy = plan.audio_target.get("energy")
        t_valence = plan.audio_target.get("valence")
        t_bpm = plan.audio_target.get("bpm")
        t_acoustic = plan.audio_target.get("acousticness")

        dist_terms: list[str] = []
        audio_params: dict[str, Any] = {"u": user_id, "exc": exc_genres, "seed": str(seed)}
        if t_energy is not None:
            dist_terms.append("CASE WHEN tf.energy IS NOT NULL THEN ABS(tf.energy - :te) ELSE 0.25 END")
            audio_params["te"] = float(t_energy)
        if t_valence is not None:
            dist_terms.append("CASE WHEN tf.valence IS NOT NULL THEN ABS(tf.valence - :tv) ELSE 0.25 END")
            audio_params["tv"] = float(t_valence)
        if t_acoustic is not None:
            dist_terms.append(
                "CASE WHEN tf.acousticness IS NOT NULL THEN ABS(tf.acousticness - :ta) ELSE 0.25 END"
            )
            audio_params["ta"] = float(t_acoustic)
        if t_bpm is not None:
            dist_terms.append(
                "CASE WHEN tf.bpm IS NOT NULL THEN LEAST(ABS(tf.bpm - :tb), ABS(2.0 * tf.bpm - :tb), ABS(0.5 * tf.bpm - :tb)) / 100.0 ELSE 0.25 END"
            )
            audio_params["tb"] = float(t_bpm)
        dist_expr = " + ".join(dist_terms) if dist_terms else "0.0"

        audio_rows = await _ids(
            f"""
            WITH dists AS (
                SELECT t.id,
                       lower(COALESCE(t.artists[1], '')) AS art,
                       ({dist_expr}) AS dist,
                       md5(t.id::text || ':' || :seed) AS rng
                FROM tracks t
                JOIN track_features tf ON tf.track_id = t.id
                WHERE {LOCAL} {exc_clause}
                  AND tf.energy IS NOT NULL
            ),
            capped AS (
                SELECT id, dist, rng,
                       ROW_NUMBER() OVER (PARTITION BY art ORDER BY dist ASC, rng) AS rn
                FROM dists
            )
            SELECT id FROM capped
            WHERE rn <= 5
            ORDER BY dist ASC, rng
            LIMIT 200
            """,
            audio_params,
        )
        _extend(audio_rows)

    # 4. Reference artists ("как Nujabes / Скриптонит") -> artist tracks + HNSW neighbors
    if plan.reference_artists:
        ref_rows = await db.fetch_all(
            f"""
            SELECT t.id, ta.vec::text AS a_vec, te.vec::text AS t_vec
            FROM tracks t
            LEFT JOIN track_emb_audio ta ON ta.track_id = t.id
            LEFT JOIN track_emb_text te ON te.track_id = t.id
            WHERE {LOCAL} {exc_clause}
              AND EXISTS (
                  SELECT 1 FROM unnest(t.artists) AS an(name)
                  WHERE EXISTS (
                      SELECT 1 FROM unnest(:refs) AS rn(rname)
                      WHERE artist_matches(an.name, rn.rname)
                  )
              )
            LIMIT 40
            """,
            {"u": user_id, "exc": exc_genres, "refs": plan.reference_artists},
        )
        ref_ids = [int(r["id"]) for r in ref_rows]
        for rid in ref_ids:
            ref_sims[rid] = 1.0
        _extend(ref_ids)
        if ref_ids:
            nn_rows = await db.fetch_all(
                f"""
                SELECT e2.track_id AS id, MAX(1.0 - (e2.vec <=> e1.vec)) AS sim
                FROM track_emb_audio e1
                JOIN track_emb_audio e2 ON e2.track_id <> e1.track_id
                JOIN tracks t ON t.id = e2.track_id
                WHERE e1.track_id = ANY(:refs) AND {LOCAL} {exc_clause}
                GROUP BY e2.track_id
                ORDER BY sim DESC
                LIMIT 100
                """,
                {"u": user_id, "refs": ref_ids[:10], "exc": exc_genres},
            )
            for nr in nn_rows:
                nid = int(nr["id"])
                ref_sims[nid] = max(ref_sims.get(nid, 0.0), float(nr["sim"] or 0.0))
                _extend([nid])

    # 5. Taste supplement ONLY proportional to taste_blend and filtered by exclusions/include_genres
    if plan.taste_blend >= 0.18 and len(ids) < limit:
        artists = sorted((profile.get("artists") or {}).keys(), key=lambda k: profile["artists"][k], reverse=True)[:20]
        if artists:
            inc_clause = "AND (t.genres_norm && :inc)" if plan.include_genres else ""
            taste_rows = await _ids(
                f"""
                WITH pool AS (
                    SELECT t.id, lower(COALESCE(t.artists[1], '')) AS art,
                           md5(t.id::text || ':' || :seed) AS rng
                    FROM tracks t
                    WHERE {LOCAL} {exc_clause} {inc_clause}
                      AND EXISTS (
                          SELECT 1 FROM unnest(t.artists) AS a(name)
                          WHERE lower(a.name) = ANY(:artists)
                      )
                ),
                capped AS (
                    SELECT id, rng, ROW_NUMBER() OVER (PARTITION BY art ORDER BY rng) AS rn
                    FROM pool
                )
                SELECT id FROM capped WHERE rn <= 4 ORDER BY rng LIMIT 100
                """,
                {
                    "u": user_id,
                    "exc": exc_genres,
                    "inc": list(plan.include_genres or []),
                    "artists": artists,
                    "seed": str(seed),
                },
            )
            _extend(taste_rows)

    return ids[:limit], sem_sims, ref_sims


async def collect_for_request(
    user_id: int,
    profile: dict[str, Any],
    kind: str = "personal",
    value: str = "",
    *,
    plan: QueryPlan | None = None,
    query_vec: list[float] | None = None,
    limit: int = 550,
    seed: int = 0,
) -> tuple[list[Candidate], dict[int, float]]:
    """Query-aware multi-channel recall. Returns (hydrated candidates, cooc_map)."""
    seeds = [int(v) for v in (profile.get("seeds") or [])]
    cooc_map = await _cooc_map(user_id, seeds, limit=150)

    if kind in ("prompt", "mood") and plan is not None:
        q_ids, sem_sims, ref_sims = await _collect_query_driven(
            user_id, profile, plan, query_vec, limit=limit, seed=seed
        )
        cands = await hydrate(user_id, profile, q_ids, sem_sims=sem_sims, ref_sims=ref_sims)
        for c in cands:
            c.cooc = cooc_map.get(c.track_id)
        return cands, cooc_map

    ids: list[int] = []
    seen: set[int] = set()

    def _extend(chunk: list[int]) -> None:
        for tid in chunk:
            if tid not in seen:
                seen.add(tid)
                ids.append(tid)

    if kind == "genre":
        target_fams = sorted(family_set([value]))
        rows = await _ids(
            f"""
            WITH matched AS (
                SELECT t.id, lower(COALESCE(t.artists[1], '')) AS art,
                       COALESCE(st.completes, 0) + COALESCE(st.plays, 0) AS pop,
                       md5(t.id::text || ':' || :seed) AS rng
                FROM tracks t
                LEFT JOIN track_features tf ON tf.track_id = t.id
                LEFT JOIN mv_user_track_stats st ON st.user_id = :u AND st.track_id = t.id
                WHERE {LOCAL}
                  AND (
                      t.genres_norm && :fams
                      OR lower(COALESCE(t.genre, '')) = lower(:val)
                      OR tf.subgenres && :fams
                  )
            ),
            capped AS (
                SELECT id, pop, rng,
                       ROW_NUMBER() OVER (PARTITION BY art ORDER BY pop DESC, rng) AS rn
                FROM matched
            )
            SELECT id FROM capped
            WHERE rn <= 5
            ORDER BY pop DESC, rng
            LIMIT :l
            """,
            {"u": user_id, "fams": target_fams or [value.lower()], "val": value, "seed": str(seed), "l": limit},
        )
        _extend(rows)
        cands = await hydrate(user_id, profile, ids)
        for c in cands:
            c.cooc = cooc_map.get(c.track_id)
        return cands, cooc_map

    if kind == "artist":
        try:
            artist_id = int(value)
        except ValueError:
            return [], cooc_map
        ar_row = await db.fetch_one("SELECT name FROM artists WHERE id = :id", {"id": artist_id})
        if not ar_row or not ar_row.get("name"):
            return [], cooc_map
        artist_name = str(ar_row["name"])
        rows = await _ids(
            f"""
            SELECT t.id
            FROM tracks t
            LEFT JOIN mv_user_track_stats st ON st.user_id = :u AND st.track_id = t.id
            WHERE {LOCAL}
              AND (
                  EXISTS (SELECT 1 FROM track_artists ta WHERE ta.track_id = t.id AND ta.artist_id = :aid)
                  OR EXISTS (SELECT 1 FROM unnest(t.artists) AS an(name) WHERE artist_matches(an.name, :aname))
              )
            ORDER BY COALESCE(st.completes, 0) DESC, COALESCE(st.plays, 0) DESC, t.year DESC NULLS LAST, t.id DESC
            LIMIT :l
            """,
            {"u": user_id, "aid": artist_id, "aname": artist_name, "l": limit},
        )
        _extend(rows)
        cands = await hydrate(user_id, profile, ids)
        for c in cands:
            c.cooc = cooc_map.get(c.track_id)
        return cands, cooc_map

    if kind == "discovery":
        # Channel A: real familiar anchors — actually played, not just liked.
        # A like without a play is a "saved but unheard" item, not an anchor.
        _extend(
            await _ids(
                f"""
                WITH anchor_pool AS (
                    SELECT t.id, lower(COALESCE(t.artists[1], '')) AS art,
                           COALESCE(EXTRACT(EPOCH FROM (NOW() - st.last_played)) / 86400.0, 21.0) AS age_days,
                           (CASE WHEN f.track_id IS NOT NULL THEN 3.0 ELSE 0.0 END
                            + COALESCE(st.completes, 0)
                            + 0.6 * COALESCE(st.repeats, 0)
                            + 0.4 * COALESCE(st.plays, 0)) AS aff,
                           md5(t.id::text || ':' || :seed) AS rng
                    FROM tracks t
                    LEFT JOIN mv_user_track_stats st ON st.track_id = t.id AND st.user_id = :u
                    LEFT JOIN favorites f ON f.track_id = t.id AND f.user_id = :u
                    WHERE {LOCAL}
                      AND st.last_played IS NOT NULL
                      AND st.last_played < NOW() - INTERVAL '5 days'
                      AND (
                          COALESCE(st.plays, 0) > 0
                          OR COALESCE(st.completes, 0) > 0
                          OR COALESCE(st.repeats, 0) > 0
                      )
                      AND NOT (
                          COALESCE(st.skips, 0) >= 2
                          AND COALESCE(st.completes, 0) = 0
                          AND f.track_id IS NULL
                      )
                ),
                capped AS (
                    SELECT id, age_days, aff, rng,
                           ROW_NUMBER() OVER (PARTITION BY art ORDER BY (age_days * GREATEST(0.6, aff)) DESC, rng) AS rn
                    FROM anchor_pool
                )
                SELECT id FROM capped WHERE rn <= 4 ORDER BY (age_days * GREATEST(0.6, aff)) DESC, rng LIMIT 220
                """,
                {"u": user_id, "seed": str(seed)},
            )
        )
        # Channel A2: liked but never played — honest "saved, not heard yet".
        _extend(
            await _ids(
                f"""
                WITH liked_unheard AS (
                    SELECT t.id, lower(COALESCE(t.artists[1], '')) AS art,
                           md5(t.id::text || ':' || :seed) AS rng
                    FROM tracks t
                    JOIN favorites f ON f.track_id = t.id AND f.user_id = :u
                    LEFT JOIN mv_user_track_stats st ON st.track_id = t.id AND st.user_id = :u
                    WHERE {LOCAL}
                      AND COALESCE(st.plays, 0) = 0
                ),
                capped AS (
                    SELECT id, rng, ROW_NUMBER() OVER (PARTITION BY art ORDER BY rng) AS rn
                    FROM liked_unheard
                )
                SELECT id FROM capped WHERE rn <= 3 ORDER BY rng LIMIT 140
                """,
                {"u": user_id, "seed": str(seed)},
            )
        )
        # Channel B: Unheard taste gems (plays == 0) via audio/text HNSW + cooc + genres
        taste = profile.get("taste") or {}
        if taste.get("taste"):
            _extend(await _audio_vec_neighbors(user_id, taste["taste"], limit=160))
        if taste.get("text_taste"):
            _extend(list((await _text_vec_neighbors(user_id, taste["text_taste"], limit=160)).keys()))
        _extend(list(cooc_map.keys())[:100])
        genres = sorted((profile.get("genres") or {}).keys(), key=lambda g: profile["genres"][g], reverse=True)[:5]
        if genres:
            _extend(
                await _ids(
                    f"""
                    WITH unheard AS (
                        SELECT t.id, lower(COALESCE(t.artists[1], '')) AS art,
                               md5(t.id::text || ':' || :seed) AS rng
                        FROM tracks t
                        LEFT JOIN mv_user_track_stats st ON st.track_id = t.id AND st.user_id = :u
                        WHERE {LOCAL}
                          AND COALESCE(st.plays, 0) = 0
                          AND t.genres_norm && :genres
                    ),
                    capped AS (
                        SELECT id, rng, ROW_NUMBER() OVER (PARTITION BY art ORDER BY rng) AS rn
                        FROM unheard
                    )
                    SELECT id FROM capped WHERE rn <= 3 ORDER BY rng LIMIT 150
                    """,
                    {"u": user_id, "genres": genres, "seed": str(seed)},
                )
            )
        # Channel C: Unheard serendipity tail across the rest of the local catalog
        _extend(
            await _ids(
                f"""
                WITH serendipity AS (
                    SELECT t.id, lower(COALESCE(t.artists[1], '')) AS art,
                           md5(t.id::text || ':' || :seed) AS rng
                    FROM tracks t
                    LEFT JOIN mv_user_track_stats st ON st.track_id = t.id AND st.user_id = :u
                    WHERE {LOCAL}
                ),
                capped AS (
                    SELECT id, rng, ROW_NUMBER() OVER (PARTITION BY art ORDER BY rng) AS rn
                    FROM serendipity
                )
                SELECT id FROM capped WHERE rn <= 2 ORDER BY rng LIMIT :n
                """,
                {"u": user_id, "seed": str(seed), "n": 100},
            )
        )
        cands = await hydrate(user_id, profile, ids[:limit])
        for c in cands:
            c.cooc = cooc_map.get(c.track_id)
        return cands, cooc_map

    # Default: `personal` multi-channel ordered & capped recall
    artists = sorted((profile.get("artists") or {}).keys(), key=lambda k: profile["artists"][k], reverse=True)[:25]
    genres = sorted((profile.get("genres") or {}).keys(), key=lambda k: profile["genres"][k], reverse=True)[:8]
    taste = profile.get("taste") or {}

    if artists:
        _extend(
            await _ids(
                f"""
                WITH ranked AS (
                    SELECT t.id, lower(COALESCE(t.artists[1], '')) AS art,
                           COALESCE(st.completes, 0) * 2.0
                           + CASE WHEN f.track_id IS NOT NULL THEN 3.0 ELSE 0.0 END
                           - CASE WHEN st.last_played > NOW() - INTERVAL '1 day' THEN 5.0 ELSE 0.0 END AS s,
                           md5(t.id::text || ':' || :seed) AS rng
                    FROM tracks t
                    LEFT JOIN mv_user_track_stats st ON st.track_id = t.id AND st.user_id = :u
                    LEFT JOIN favorites f ON f.track_id = t.id AND f.user_id = :u
                    WHERE {LOCAL}
                      AND EXISTS (
                          SELECT 1 FROM unnest(t.artists) AS a(name)
                          WHERE lower(a.name) = ANY(:artists)
                      )
                ),
                capped AS (
                    SELECT id, s, rng,
                           ROW_NUMBER() OVER (PARTITION BY art ORDER BY s DESC, rng) AS rn
                    FROM ranked
                )
                SELECT id FROM capped WHERE rn <= 4 ORDER BY s DESC, rng LIMIT 160
                """,
                {"u": user_id, "artists": artists, "seed": str(seed)},
            )
        )

    if genres:
        _extend(
            await _ids(
                f"""
                WITH ranked AS (
                    SELECT t.id, lower(COALESCE(t.artists[1], '')) AS art,
                           md5(t.id::text || ':' || :seed) AS rng
                    FROM tracks t
                    WHERE {LOCAL} AND t.genres_norm && :genres
                ),
                capped AS (
                    SELECT id, rng, ROW_NUMBER() OVER (PARTITION BY art ORDER BY rng) AS rn
                    FROM ranked
                )
                SELECT id FROM capped WHERE rn <= 4 ORDER BY rng LIMIT 150
                """,
                {"u": user_id, "genres": genres, "seed": str(seed)},
            )
        )

    if taste.get("taste"):
        _extend(await _audio_vec_neighbors(user_id, taste["taste"], limit=140))
    if taste.get("recent"):
        _extend(await _audio_vec_neighbors(user_id, taste["recent"], limit=80))
    if taste.get("text_taste"):
        _extend(list((await _text_vec_neighbors(user_id, taste["text_taste"], limit=120)).keys()))

    _extend(list(cooc_map.keys())[:100])

    # Fresh local tracks + deterministic exploration fill
    _extend(
        await _ids(
            f"""
            SELECT t.id FROM tracks t
            WHERE {LOCAL}
            ORDER BY t.last_seen_at DESC
            LIMIT 80
            """,
            {"u": user_id},
        )
    )
    if len(ids) < limit:
        _extend(
            await _ids(
                f"""
                SELECT t.id FROM tracks t
                WHERE {LOCAL}
                ORDER BY md5(t.id::text || ':' || :seed)
                LIMIT :n
                """,
                {"u": user_id, "seed": str(seed), "n": limit - len(ids)},
            )
        )

    cands = await hydrate(user_id, profile, ids[:limit])
    for c in cands:
        c.cooc = cooc_map.get(c.track_id)
    return cands, cooc_map


async def collect(user_id: int, profile: dict, limit: int = 500) -> list[Candidate]:
    """Backward-compatible wrapper for personal candidate recall."""
    cands, _cooc = await collect_for_request(
        user_id, profile, kind="personal", value="", limit=limit, seed=user_id
    )
    return cands


async def neighbours(track_id: int, limit: int = 150, user_id: int | None = None) -> list[int]:
    dis_clause = (
        "AND NOT EXISTS (SELECT 1 FROM dislikes d WHERE d.user_id = :u AND d.track_id = t.id)"
        if user_id is not None
        else ""
    )
    return await _ids(
        f"""
        SELECT q.id, SUM(q.w) AS w
        FROM (
            SELECT c.track_b AS id, c.weight AS w FROM mv_item_cooccurrence c WHERE c.track_a = :s
            UNION ALL
            SELECT c.track_a AS id, c.weight AS w FROM mv_item_cooccurrence c WHERE c.track_b = :s
        ) q
        JOIN tracks t ON t.id = q.id
        WHERE EXISTS (SELECT 1 FROM track_sources s WHERE s.track_id = t.id AND s.connector = 'local' AND s.available)
          AND NOT EXISTS (SELECT 1 FROM track_quality tq WHERE tq.track_id = t.id AND tq.error IS NOT NULL)
          {dis_clause}
        GROUP BY q.id
        ORDER BY w DESC
        LIMIT :l
        """,
        {"s": track_id, "l": limit, "u": user_id},
    )


async def audio_neighbours(user_id: int, track_id: int, limit: int = 150) -> list[int]:
    """Local tracks that SOUND like the seed — pgvector HNSW over `track_emb_audio`."""
    emb_row = await db.fetch_one(
        "SELECT vec::text AS vec FROM track_emb_audio WHERE track_id = :s", {"s": track_id}
    )
    if emb_row and emb_row.get("vec"):
        rows = await _ids(
            f"""
            SELECT e.track_id AS id
            FROM track_emb_audio e
            JOIN tracks t ON t.id = e.track_id
            WHERE e.track_id <> :s AND {LOCAL}
            ORDER BY e.vec <=> CAST(:v AS vector)
            LIMIT :l
            """,
            {"u": user_id, "s": track_id, "v": emb_row["vec"], "l": limit},
        )
        if rows:
            return rows

    # Fallback when `track_emb_audio` is not populated yet (e.g. unit test fixtures)
    seed_row = await db.fetch_one(
        "SELECT * FROM track_features WHERE track_id = :s", {"s": track_id}
    )
    corpus_stats = await load_corpus_stats()
    seed_vec = feature_vector(seed_row, corpus_stats)
    if seed_vec is None:
        return []
    rows_all = await db.fetch_all(
        f"""
        SELECT tf.track_id AS id, tf.*
        FROM track_features tf
        JOIN tracks t ON t.id = tf.track_id
        WHERE tf.track_id <> :s AND {LOCAL}
        """,
        {"u": user_id, "s": track_id},
    )
    scored: list[tuple[float, int]] = []
    for row in rows_all:
        sim = cosine(seed_vec, feature_vector(row, corpus_stats))
        if sim is not None:
            scored.append((sim, int(row["id"])))
    scored.sort(reverse=True)
    return [track for _, track in scored[:limit]]


async def text_neighbours(user_id: int, track_id: int, limit: int = 150) -> list[int]:
    """Local tracks semantically close to `track_id` via `track_emb_text` HNSW."""
    emb_row = await db.fetch_one(
        "SELECT vec::text AS vec FROM track_emb_text WHERE track_id = :s", {"s": track_id}
    )
    if not emb_row or not emb_row.get("vec"):
        return []
    return await _ids(
        f"""
        SELECT e.track_id AS id
        FROM track_emb_text e
        JOIN tracks t ON t.id = e.track_id
        WHERE e.track_id <> :s AND {LOCAL}
        ORDER BY e.vec <=> CAST(:v AS vector)
        LIMIT :l
        """,
        {"u": user_id, "s": track_id, "v": emb_row["vec"], "l": limit},
    )


async def same_artist_or_genre(user_id: int, track_id: int, limit: int = 150) -> list[int]:
    return await _ids(
        f"""
        SELECT t.id
        FROM tracks t
        JOIN tracks seed ON seed.id = :s
        WHERE t.id <> :s AND {LOCAL}
          AND (
              (seed.genres_norm <> '{{}}' AND t.genres_norm && seed.genres_norm)
              OR lower(COALESCE(t.genre, '')) = lower(COALESCE(seed.genre, ''))
              OR EXISTS (
                  SELECT 1 FROM unnest(seed.artists) AS sa(name)
                  WHERE EXISTS (
                      SELECT 1 FROM unnest(t.artists) AS ta(name)
                      WHERE artist_matches(ta.name, sa.name)
                  )
              )
          )
        ORDER BY t.year DESC NULLS LAST, t.id DESC
        LIMIT :l
        """,
        {"u": user_id, "s": track_id, "l": limit},
    )


def within_days(last_played: datetime | None, days: int) -> bool:
    if last_played is None:
        return False
    last = last_played if last_played.tzinfo else last_played.replace(tzinfo=UTC)
    return (datetime.now(UTC) - last) <= timedelta(days=days)
