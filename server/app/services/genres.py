"""Canonical genre families.

Providers send raw genre strings («Hip-Hop & Rap», «rusrap», «хип-хоп»,
«R&B/Soul») that never compare equal. Every consumer — candidate selection,
prompt filters, taste profiles — matches FAMILIES, never raw strings. The
family key is what lands in tracks.genres_norm at ingest time.
"""

from __future__ import annotations

import re

GENRE_FAMILIES: dict[str, tuple[str, ...]] = {
    "rap": (
        "rap",
        "хип-хоп",
        "хип хоп",
        "hip-hop",
        "hip hop",
        "hiphop",
        "rusrap",
        "рэп",
        "trap",
        "трэп",
        "drill",
        "дриллинг",
        "grime",
        "грайм",
        "phonk",
        "фонк",
        "boom bap",
        "cloud rap",
    ),
    "pop": (
        "pop",
        "поп",
        "попса",
        "ruspop",
        "kpop",
        "jpop",
        "dance-pop",
        "electropop",
        "synth-pop",
        "synthpop",
        "k-pop",
        "j-pop",
        "britpop",
        "dream pop",
        "indie pop",
        "эстрада",
    ),
    "rock": (
        "rock",
        "рок",
        "rusrock",
        "alt-rock",
        "alt rock",
        "alternative",
        "альтернатив",
        "indie",
        "инди",
        "punk",
        "панк",
        "metal",
        "метал",
        "grunge",
        "гранж",
        "shoegaze",
        "шугейз",
        "пост-панк",
        "post-punk",
        "new wave",
    ),
    "electronic": (
        "electronic",
        "электрон",
        "house",
        "хаус",
        "techno",
        "техно",
        "trance",
        "транс",
        "drum and bass",
        "drum & bass",
        "dnb",
        "dubstep",
        "дабстеп",
        "idm",
        "ambient",
        "эмбиент",
        "edm",
        "breakbeat",
        "synthwave",
        "синтвейв",
        "retrowave",
        "downtempo",
        "trip-hop",
        "трип-хоп",
        "uk garage",
        "chillout",
    ),
    "jazz": (
        "jazz",
        "джаз",
        "bebop",
        "swing",
        "свинг",
        "bossa nova",
        "босса",
        "blues",
        "блюз",
        "lounge",
        "nu-jazz",
        "acid jazz",
    ),
    "classical": ("classical", "классик", "orchestra", "opera", "baroque", "симфон", "неоклассик"),
    "folk": ("folk", "фолк", "шансон", "chanson", "bard", "бард", "country", "кантри", "ethnic", "этно"),
    "rnb": ("rnb", "r'n'b", "r&b", "рнб", "soul", "соул", "funk", "фанк", "neo-soul"),
    "reggae": ("reggae", "регги", "ska", "dub", "даб"),
    "soundtrack": ("soundtrack", "score", "ost", "саундтрек", "anime"),
    "experimental": ("experimental", "эксперимент", "noise", "industrial", "avant"),
    "instrumental": ("instrumental", "инструментал", "lo-fi", "lofi", "лоу-фай", "beats", "chillhop"),
}

# Short ASCII tokens that cause false positives as substrings (e.g. "dub" in "dubstep",
# "ska" in "skazka", "ost" in "post-punk", "rap" in "trapezoid").
_WORD_BOUNDARY_TOKENS = frozenset({"dub", "ska", "ost", "edm", "idm", "rap", "pop", "rock", "rnb"})
_BOUNDARY_PATTERNS: dict[str, re.Pattern[str]] = {
    tok: re.compile(rf"(?<![a-zа-яё0-9]){re.escape(tok)}(?![a-zа-яё0-9])", re.IGNORECASE)
    for tok in _WORD_BOUNDARY_TOKENS
}

# Order matters only for the returned list order.
FAMILY_KEYS = tuple(GENRE_FAMILIES)


def _matches_token(text: str, token: str) -> bool:
    pat = _BOUNDARY_PATTERNS.get(token)
    if pat is not None:
        return pat.search(text) is not None
    return token in text


def genre_families(raw: str | None) -> list[str]:
    """Canonical families for one raw genre string (may be empty)."""
    if not raw:
        return []
    text = str(raw).strip().lower()
    if not text:
        return []
    return [
        family
        for family, tokens in GENRE_FAMILIES.items()
        if any(_matches_token(text, token) for token in tokens)
    ]


def normalize_genres(values: list[str] | None) -> list[str]:
    """Union of families over several raw genre strings, order preserved."""
    out: list[str] = []
    for value in values or []:
        for family in genre_families(value):
            if family not in out:
                out.append(family)
    return out


def family_set(raw_values: list[str] | tuple[str, ...] | set[str] | None) -> set[str]:
    """Families referenced by a prompt phrase list («русский рэп» -> {"rap"})."""
    return {family for value in (raw_values or ()) for family in genre_families(value)}
