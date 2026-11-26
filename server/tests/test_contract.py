"""Contract drift guard: every public /v1 route must be documented in openapi.yaml.

The server previously grew endpoints that were missing from the frozen contract
(/mixes/prompt, /tracks/{id}/similar, /scoring/*). This test fails whenever a
/server/ route appears that the contract does not describe. Path params are
normalised so {trackId} vs {track_id} naming does not matter. Skips cleanly when
the sibling kwhale-android tree (which holds openapi.yaml) is not present.
"""

import re
from pathlib import Path

import pytest
import yaml

from app.main import app


def _norm(p: str) -> str:
    return re.sub(r"\{[^}]*\}", "{}", p)


def _openapi_path() -> Path | None:
    p = Path(__file__).resolve().parents[3] / "kwhale-android" / "api" / "openapi.yaml"
    return p if p.is_file() else None


def test_server_v1_routes_documented_in_openapi():
    oa = _openapi_path()
    if oa is None:
        pytest.skip("openapi.yaml not found in sibling kwhale-android tree")
    documented = {_norm(p) for p in yaml.safe_load(oa.read_text())["paths"]}

    server: set[str] = set()
    for r in app.routes:
        path = getattr(r, "path", None)
        if not path or not path.startswith("/v1/"):
            continue  # skip /admin, /docs, /openapi.json, static
        server.add(_norm(path[len("/v1") :]))

    undocumented = sorted(p for p in server if p not in documented)
    assert not undocumented, f"серверные /v1 пути вне openapi.yaml: {undocumented}"
