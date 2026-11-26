"""Shared normalization: titles, artist lists, NFKC — one place for all.

Used by dedup (identity keys), tagging (clean tags) and providers (search
queries). Rules are conservative: junk that services append to titles goes
away (feat-clauses, "(Explicit)", "(2011 Remaster)"), but meaningful
differences (Live, Remix, cover versions) stay — they are different tracks.
"""

from __future__ import annotations

import re
import unicodedata

_SPACES_RE = re.compile(r"\s+")

# "(feat. X)" / "[ft. X]" / "(featuring X)" / "(при уч. X)" / "(w/ X)"
_FEAT_PAREN_RE = re.compile(
    r"\s*[([]\s*(?:(?:feat|ft|featuring|with|w/)\.?|(?:при\s+уч(?:астии)?|уч|п\.у)\.?)[\s.:/-]+([^)\]]+)[)\]]\s*",
    re.IGNORECASE,
)
_FEAT_RE = re.compile(
    r"\s*[([]\s*(?:(?:feat|ft|featuring)\b|(?:при\s+уч(?:астии)?|уч|п\.у)\b\.?|(?:with|w/)\s+)[^)\]]*[)\]]\s*",
    re.IGNORECASE,
)
_FEAT_INLINE_RE = re.compile(
    r"\s+(?:(?:feat|ft|featuring)\.?|(?:при\s+уч(?:астии)?|п\.у)\.?)\s+\S.*$",
    re.IGNORECASE,
)

# "(prod. by X)" / "[prod. X]" / trailing "prod. by X" (keeps TYPE BEAT intact!)
_PROD_PAREN_RE = re.compile(
    r"\s*[([]\s*(?:prod\.?(?:\s+by)?|produced\s+by|music\s+by|prod\s+by)\b[^)\]]*[)\]]\s*",
    re.IGNORECASE,
)
_PROD_TAIL_RE = re.compile(
    r"\s+(?:prod\.?\s+by|produced\s+by|prod\.?)\s+(?!.*\btype\s+beat\b)\S.*$",
    re.IGNORECASE,
)

# trailing version junk: "(Explicit)", "(Clean)", "(Album Version)", "[Official Audio]"...
# Remastered, Radio Edit, Live, Remix, Acoustic, Instrumental stay distinct!
_VERSION_JUNK_RE = re.compile(
    r"\s*[([]\s*(?:"
    r"explicit|clean|dirty|album\s+version|single\s+version|"
    r"original\s+mix|original\s+version|bonus\s+track\s*\d*|"
    r"official\s+(?:music\s+)?(?:audio|video|lyric\s+video|visualizer)|"
    r"lyric\s+video|visualizer|lyrics|audio|текст\w*\s+вшит\w*"
    r")\s*[)\]]",
    re.IGNORECASE,
)
_VERSION_TAIL_RE = re.compile(
    r"\s*[-–—]\s*(?:explicit|clean|dirty|album\s+version|single\s+version|original\s+mix)\s*$",
    re.IGNORECASE,
)
_SLOWED_REVERB_RE = re.compile(
    r"\bslowed\s*(?:&|\+|and|n)?\s*(?:&amp;)?\s*reverb\b", re.IGNORECASE
)
_SPED_UP_RE = re.compile(r"\bspe+d\s*[-– ]?\s*up(?:\s+ver(?:sion)?\.?)?\b", re.IGNORECASE)
_RMX_WORD_RE = re.compile(r"\brmx\b", re.IGNORECASE)

# artist separators: "A feat. B", "A & B", "A, B", "A vs B", "A x B"
# Require non-empty tokens on both sides of "x" so "MONSTA X" / "Lil Nas X" never split.
_ARTIST_SPLIT_RE = re.compile(
    r"(?:\s*(?:,|&|\bvs\b\.?|\bfeaturing\b|\bfeat\b\.?|\bft\b\.?|\bпри\s+уч\.?|\bпри\s+участии\b)\s*|(?<=\S)\s+[x×х]\s+(?=\S))",
    re.IGNORECASE,
)
_QUOTE_CHARS = "\"'“”‘’«»"

# Cyrillic letters that are visually identical to Latin ones. Provider
# metadata mixes them freely ("Мashа Hima" vs "Masha Hima"), so dedup keys
# must fold them or the same recording gets two identities (2341/2357).
_HOMOGLYPHS = str.maketrans(
    {
        "А": "A", "В": "B", "Е": "E", "К": "K", "М": "M", "Н": "H",
        "О": "O", "Р": "P", "С": "C", "Т": "T", "У": "Y", "Х": "X",
        "І": "I", "Ј": "J", "Ѕ": "S",
        "а": "a", "в": "b", "е": "e", "к": "k", "м": "m", "н": "h",
        "о": "o", "р": "p", "с": "c", "т": "t", "у": "y", "х": "x",
        "і": "i", "ј": "j", "ѕ": "s", "ё": "e", "Ё": "E",
    }
)
# Non-decomposing Latin extensions folded similarly by SQL unaccent()
_LATIN_EXT_FOLD = str.maketrans(
    {
        "ø": "o", "Ø": "O", "ł": "l", "Ł": "L", "đ": "d", "Đ": "D",
        "й": "и", "Й": "И",
    }
)
_CYRILLIC_RE = re.compile(r"[а-яё]", re.IGNORECASE)
_LATIN_RE = re.compile(r"[a-z]", re.IGNORECASE)
_RU_TRANSLIT = {
    "а": "a", "б": "b", "в": "v", "г": "g", "д": "d", "е": "e", "ё": "e",
    "ж": "zh", "з": "z", "и": "i", "й": "i", "к": "k", "л": "l", "м": "m",
    "н": "n", "о": "o", "п": "p", "р": "r", "с": "s", "т": "t", "у": "u",
    "ф": "f", "х": "h", "ц": "ts", "ч": "ch", "ш": "sh", "щ": "shch",
    "ъ": "", "ы": "y", "ь": "", "э": "e", "ю": "yu", "я": "ya",
    "a": "a", "b": "b", "c": "c", "d": "d", "e": "e", "f": "f", "g": "g",
    "h": "h", "i": "i", "j": "j", "k": "k", "l": "l", "m": "m", "n": "n",
    "o": "o", "p": "p", "q": "q", "r": "r", "s": "s", "t": "t", "u": "u",
    "v": "v", "w": "w", "x": "x", "y": "y", "z": "z",
}


def _fold_homoglyphs(text: str) -> str:
    return text.translate(_HOMOGLYPHS)


def _strip_accents_and_fold(text: str) -> str:
    """Fold NFKD combining marks + й→и BEFORE punctuation regexes.

    Decomposing й→и+\\u0306 before `[^\\w\\s]` replacement used to turn combining
    marks into mid-word spaces ("Майор"→"маи ор", "Beyoncé"→"beyonc e").
    """
    nfkd = unicodedata.normalize("NFKD", text)
    no_marks = "".join(ch for ch in nfkd if unicodedata.category(ch) != "Mn")
    return no_marks.translate(_LATIN_EXT_FOLD)


def normalize_text(text: str) -> str:
    """NFKC + whitespace collapse + quote trim (display-level cleanup)."""
    text = unicodedata.normalize("NFKC", text or "")
    text = _SPACES_RE.sub(" ", text).strip()
    return text.strip(_QUOTE_CHARS + " ")


def extract_featured_artists(title: str) -> tuple[str, list[str]]:
    """Pull featured collaborators out of a title before stripping the clause."""
    out = normalize_text(title)
    featured: list[str] = []

    def _collect(match: re.Match[str]) -> str:
        raw = match.group(1)
        for part in split_artists_strict(raw):
            if part and part not in featured:
                featured.append(part)
        return " "

    out = _FEAT_PAREN_RE.sub(_collect, out)
    if " (" not in out and " [" not in out:
        m = _FEAT_INLINE_RE.search(out)
        if m:
            raw_inline = re.sub(
                r"^\s*(?:(?:feat|ft|featuring)\.?|(?:при\s+уч(?:астии)?|п\.у)\.?)\s*",
                "",
                m.group(0),
                flags=re.IGNORECASE,
            )
            for part in split_artists_strict(raw_inline):
                if part and part not in featured:
                    featured.append(part)
            out = out[: m.start()]
    return normalize_text(out), featured


def strip_version_junk(title: str) -> str:
    """Drop feat-clauses, prod-credits, and explicit/upload noise (dedup identity)."""
    out = normalize_text(title)
    out = _FEAT_RE.sub(" ", out)
    # inline "feat. X" / "при уч. X" only when it's clearly a suffix ("Title feat. X")
    if " (" not in out and " [" not in out:
        out = _FEAT_INLINE_RE.sub("", out)
    for _ in range(3):  # suffixes stack: "(prod. X) (Explicit) (2011 Remaster)"
        cleaned = _VERSION_JUNK_RE.sub("", out)
        cleaned = _PROD_PAREN_RE.sub(" ", cleaned)
        cleaned = _PROD_TAIL_RE.sub("", cleaned)
        cleaned = _VERSION_TAIL_RE.sub("", cleaned)
        stripped = normalize_text(cleaned)
        if stripped == out:
            break
        out = stripped
    return normalize_text(out)


def split_artists(text: str) -> list[str]:
    """"A feat. B & C" -> ["A", "B", "C"]; empty parts dropped, order kept."""
    text = normalize_text(text)
    if not text:
        return []
    parts = [normalize_text(p) for p in _ARTIST_SPLIT_RE.split(text)]
    out: list[str] = []
    for p in parts:
        if p and p not in out:
            out.append(p)
    return out


# Entity-level splitting: only unambiguous collab separators. The bare "x" is
# deliberately absent — it is part of real band names ("MONSTA X", "AC/DC"
# semantics), and cutting it would materialise a wrong artist row. The loose
# `split_artists` above stays for dedup keys only, where a false cut is cheap.
_ARTIST_SPLIT_STRICT_RE = re.compile(
    r"\s*(?:,|&|\bvs\b\.?|\bfeaturing\b|\bfeat\b\.?|\bft\b\.?|\bпри\s+уч\.?|\bпри\s+участии\b)\s*",
    re.IGNORECASE,
)

# Noise tags appended by SoundCloud/YT artist profiles that should not split identity
_ARTIST_PROFILE_NOISE_RE = re.compile(
    r"\s*[([]\s*(?:official(?:\s+audio|\s+channel|\s+page)?|music)\s*[)\]]\s*$",
    re.IGNORECASE,
)

# Suffixes that appear on the right side of a real "Title - Tag" dash and must NOT
# be treated as an "Artist - Title" uploader prefix split.
_NON_TITLE_DASH_RIGHT_RE = re.compile(
    r"^(?:single|ep|lp|explicit|clean|dirty|deluxe(?:\s+edition)?|"
    r"remaster(?:ed)?(?:\s+\d{4})?|\d{4}\s+remaster(?:ed)?|"
    r"live(?:\s+at\s+.+)?|acoustic(?:\s+version)?|instrumental|"
    r"radio\s+edit|extended\s+mix|club\s+mix|original\s+mix|bonus\s+track.*)$",
    re.IGNORECASE,
)


def split_artists_strict(text: str) -> list[str]:
    """"A feat. B & C" -> ["A","B","C"]; "MONSTA X" stays whole."""
    text = normalize_text(text)
    if not text:
        return []
    parts = [normalize_text(p) for p in _ARTIST_SPLIT_STRICT_RE.split(text)]
    out: list[str] = []
    for p in parts:
        if p and p not in out:
            out.append(p)
    return out


def flatten_artist_names(artists: list[str] | None, *, loose: bool = False) -> list[str]:
    """Collab strings inside the array get split too.

    Sources disagree on packing: one sends ["Тимати,MOT,ЕГОР КРИД"], another
    ["Тимати","MOT","ЕГОР КРИД"]. Both must yield the same participant list.
    When `loose=True` (dedup comparison only, never entity creation), also
    splits "A x B" collabs cleanly.
    """
    splitter = split_artists if loose else split_artists_strict
    out: list[str] = []
    for a in artists or []:
        for part in splitter(a):
            if part not in out:
                out.append(part)
    return out


def dedup_artist_key(text: str) -> str:
    """Lowercased artist string for identity matching.

    Only feat-clauses are dropped ("Daft Punk feat. X" == "Daft Punk");
    list separators collapse to spaces so "A, B" == "A & B" (historical
    dedup_key behaviour kept).
    """
    joined = _fold_homoglyphs(normalize_text(text or ""))
    joined = _FEAT_RE.sub(" ", joined)
    joined = _FEAT_INLINE_RE.sub("", joined)
    joined = _strip_accents_and_fold(joined)
    joined = re.sub(r"[^\w\s]", " ", joined.lower())
    return _SPACES_RE.sub(" ", joined).strip()


def compact_artist_key(text: str) -> str:
    """Space-stripped artist identity matching SQL `artist_norm_imm(name)`.

    Collapses whitespace drift ("100 gecs" == "100gecs", "MONSTA X" == "MONSTAX",
    "TylerTheCreator(Official)" == "Tyler, The Creator") while preserving script.
    """
    cleaned = _ARTIST_PROFILE_NOISE_RE.sub("", normalize_text(text or ""))
    key = dedup_artist_key(cleaned).replace(" ", "")
    return key


def translit_ru_to_lat(text: str) -> str:
    """Deterministic Cyrillic->Latin transliteration for cross-script matching."""
    folded = _strip_accents_and_fold(normalize_text(text or "")).lower()
    out: list[str] = []
    for ch in folded:
        if ch in _RU_TRANSLIT:
            out.append(_RU_TRANSLIT[ch])
        elif ch.isalnum() or ch.isspace():
            out.append(ch)
        else:
            out.append(" ")
    return _SPACES_RE.sub(" ", "".join(out)).strip()


def _phonetic_translit_skeleton(text: str) -> str:
    """Normalize minor Latin/Cyrillic spelling differences in transliterated names."""
    lat = translit_ru_to_lat(text)
    lat = _SPACES_RE.sub("", lat)
    # x+ -> ks ("Oxxxymiron" <-> "Оксимирон"), kh/h, yu/iu, ya/ia, y/i, trailing silent 'e'
    lat = re.sub(r"x+", "ks", lat)
    lat = lat.replace("shch", "sch").replace("kh", "h")
    lat = lat.replace("yu", "iu").replace("ya", "ia").replace("y", "i")
    if len(lat) >= 7 and lat.endswith("e"):
        lat = lat[:-1]
    lat = re.sub(r"(.)\1+", r"\1", lat)
    return lat


def artist_identity_keys(text: str, *, allow_translit: bool = True) -> set[str]:
    """Candidate artist equivalence keys (exact compact, mixed-script, and translit).

    - Always emits compact key matching `artist_norm_imm`.
    - Mixed Cyrillic+Latin ("Zемфира") folds into Cyrillic/translit keys so
      one-letter homoglyph look-alikes don't stay orphan cards.
    - Pure Cyrillic/Latin translit keys emitted when >= 4 chars so "Kino"/"КИНО",
      "Zemfira"/"Земфира", "Skryptonite"/"Скриптонит" agree cleanly.
    """
    raw = _ARTIST_PROFILE_NOISE_RE.sub("", normalize_text(text or ""))
    if not raw:
        return set()
    keys: set[str] = set()
    base = compact_artist_key(raw)
    if base:
        keys.add(base)
    has_cyr = bool(_CYRILLIC_RE.search(raw))
    has_lat = bool(_LATIN_RE.search(raw))
    if has_cyr and has_lat:
        # Mixed script inside one token ("Zемфира"): map via translit regardless of length
        tlat = _SPACES_RE.sub("", translit_ru_to_lat(raw))
        if tlat:
            keys.add(f"tr:{tlat}")
            keys.add(f"ph:{_phonetic_translit_skeleton(raw)}")
    if allow_translit and len(base) >= 4:
        tlat = _SPACES_RE.sub("", translit_ru_to_lat(raw))
        if len(tlat) >= 4:
            keys.add(f"tr:{tlat}")
        skel = _phonetic_translit_skeleton(raw)
        if len(skel) >= 6:
            keys.add(f"ph:{skel}")
    return keys


def artists_compatible(artists_a: list[str] | None, artists_b: list[str] | None, *, allow_subset: bool = True) -> bool:
    """True when two participant lists represent the same track performance.

    Handles:
    - Collab packing: ["SALUKI & тёмный принц"] == ["SALUKI", "тёмный принц"]
    - Reordered features / guests when participant sets intersect
    - One side dropping featured credits (["SALUKI"] + "(feat. B)" vs ["SALUKI", "B"])
    - Homoglyphs, spacing drift, mixed scripts, and RU<->LAT transliteration
    Does NOT falsely merge distinct cover artists (disjoint participant keys).
    """
    flat_a = flatten_artist_names(artists_a, loose=True)
    flat_b = flatten_artist_names(artists_b, loose=True)
    if not flat_a and not flat_b:
        return True
    if not flat_a or not flat_b:
        return False

    sets_a = [artist_identity_keys(a) for a in flat_a]
    sets_b = [artist_identity_keys(b) for b in flat_b]
    sets_a = [s for s in sets_a if s]
    sets_b = [s for s in sets_b if s]
    if not sets_a or not sets_b:
        return False

    # Same primary artist (handles primary match with optional featured guests)
    if sets_a[0] & sets_b[0]:
        return True

    if allow_subset:
        # Check if one side's full participant list overlaps cleanly where the shorter
        # list is a subset of the longer list (e.g. ["SALUKI", "ANIKV"] vs ["ANIKV", "SALUKI"],
        # or ["BATO", "SALUKI"] vs ["SALUKI", "BATO"], or feat extraction mismatch).
        smaller, larger = (sets_a, sets_b) if len(sets_a) <= len(sets_b) else (sets_b, sets_a)
        if all(any(sa & sb for sb in larger) for sa in smaller):
            return True
    else:
        # Exact participant set match (order-insensitive)
        if len(sets_a) == len(sets_b) and all(any(sa & sb for sb in sets_b) for sa in sets_a):
            return True
    return False


def split_uploader_title(title: str) -> tuple[str, str] | None:
    """Split 'Artist — Track' / 'Artist - Track [tags]' into (left, right).

    Guards against false splits when the right side is only a release/edition
    qualifier ('Track - Radio Edit', 'Album - Single', 'X - Live').
    """
    text = normalize_text(title or "")
    m = re.match(r"^(.+?)\s+[—–-]\s+(.+)$", text)
    if not m:
        return None
    left, right = m.group(1).strip(), m.group(2).strip()
    if not left or not right:
        return None
    if _NON_TITLE_DASH_RIGHT_RE.match(right):
        return None
    return left, right


def dedup_title_key(text: str) -> str:
    """Lowercased, junk-free title for identity matching."""
    cleaned = _fold_homoglyphs(strip_version_junk(text))
    cleaned = _SLOWED_REVERB_RE.sub("slowed reverb", cleaned)
    cleaned = _SPED_UP_RE.sub("sped up", cleaned)
    cleaned = _RMX_WORD_RE.sub("remix", cleaned)
    cleaned = _strip_accents_and_fold(cleaned)
    cleaned = re.sub(r"[^\w\s]", " ", cleaned.lower())
    return _SPACES_RE.sub(" ", cleaned).strip() or text.strip().lower()


# ── Album release type ───────────────────────────────────────────────────────

_ALBUM_TYPE_ALIASES = {
    "single": "single",
    "ep": "ep",
    "album": "album",
    "compilation": "compilation",
    "compile": "compilation",
    "live": "live",
}

_ALBUM_TYPE_HINT_RE = re.compile(
    r"(?:\(\s*(?:single|ep)\s*\)|[-–—]\s*(?:single|ep)\s*$)", re.IGNORECASE
)


def derive_album_type(
    title: str | None = None,
    track_count: int | None = None,
    explicit: str | None = None,
) -> str:
    """Best-effort release type for the album/single/EP split on artist pages.

    Sources rarely expose a release type, so this is a heuristic: an explicit
    signal (connector field, "(Single)" or " - Single" in the title) wins, then
    track count — 1-2 tracks is a single, 3-6 an EP, 7+ a full album. The value
    only labels the UI (Album.type in the API contract), never identity.
    """
    if explicit:
        mapped = _ALBUM_TYPE_ALIASES.get(str(explicit).strip().lower())
        if mapped:
            return mapped
    if title and _ALBUM_TYPE_HINT_RE.search(title):
        return (
            "ep"
            if re.search(r"(?:\(\s*ep\s*\)|[-–—]\s*ep\s*$)", title, re.IGNORECASE)
            else "single"
        )
    if track_count is not None:
        if track_count <= 2:
            return "single"
        if track_count <= 6:
            return "ep"
        return "album"
    return "album"


_RELEASE_TAIL_RE = re.compile(
    r"\s*[-–—]\s*(?:single|ep|lp|deluxe(?:\s+edition)?|expanded(?:\s+edition)?)\s*$", re.IGNORECASE
)
_ALBUM_EDITION_PARENS_RE = re.compile(
    r"\s*[([]\s*(?:deluxe|expanded|special|collector'?s?|\d+(?:st|nd|rd|th)\s+anniversary|anniversary)(?:\s+(?:edition|version))?\s*[)\]]",
    re.IGNORECASE,
)
_SOUNDTRACK_PARENS_RE = re.compile(
    r"\s*[([]\s*original\s+(?:motion\s+picture\s+)?soundtrack\s*[)\]]",
    re.IGNORECASE,
)


def album_title_key(title: str | None) -> str:
    """Album identity title: version junk, deluxe/anniversary editions, and release-type tails dropped.

    «LOCA LA VIDA - Single» / «WILD EA$T (Deluxe)» and «WILD EA$T» are one release
    seen through two sources; without stripping edition tags they stay duplicate
    rows on the artist page. Track titles intentionally keep edition markers.
    """
    out = strip_version_junk(title or "")
    out = _SOUNDTRACK_PARENS_RE.sub("", out)
    out = _ALBUM_EDITION_PARENS_RE.sub("", out)
    for _ in range(2):
        stripped = normalize_text(_RELEASE_TAIL_RE.sub("", out))
        stripped = normalize_text(_ALBUM_EDITION_PARENS_RE.sub("", stripped))
        if stripped == out:
            break
        out = stripped
    return dedup_title_key(out)
