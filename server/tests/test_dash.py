"""DASH manifest parsing (offline, synthetic manifests like Tidal's)."""

from __future__ import annotations

import base64
import os

os.environ.setdefault("DATABASE_URL", "postgresql+asyncpg://kwhale:kwhale_v3_pg@localhost:5433/kwhale")

from app.connectors.dash import decode_data_uri, parse_manifest  # noqa: E402

_MPD_SEGMENT_LIST = """<?xml version="1.0"?>
<MPD xmlns="urn:mpeg:dash:schema:mpd:2011" type="static">
  <Period>
    <AdaptationSet mimeType="audio/mp4">
      <Representation id="1" bandwidth="1411200" codecs="fLaC">
        <SegmentList>
          <Initialization sourceURL="init.mp4"/>
          <SegmentURL media="seg-1.m4s"/>
          <SegmentURL media="seg-2.m4s"/>
        </SegmentList>
      </Representation>
    </AdaptationSet>
  </Period>
</MPD>
"""

_MPD_SEGMENT_TEMPLATE = """<?xml version="1.0"?>
<MPD xmlns="urn:mpeg:dash:schema:mpd:2011">
  <Period duration="PT12S">
    <AdaptationSet mimeType="audio/flac">
      <Representation id="audio" bandwidth="900000">
        <SegmentTemplate timescale="44100" initialization="$RepresentationID$/init.mp4"
                         media="$RepresentationID$/$Number%05d$.m4s" startNumber="1"
                         segmentDuration="176400"/>
      </Representation>
    </AdaptationSet>
  </Period>
</MPD>
"""

_MPD_TIMELINE = """<?xml version="1.0"?>
<MPD xmlns="urn:mpeg:dash:schema:mpd:2014">
  <Period>
    <AdaptationSet>
      <Representation id="x">
        <SegmentTemplate initialization="init.mp4" media="s$Number$.m4s" startNumber="0">
          <SegmentTimeline>
            <S t="0" d="4" r="2"/>
          </SegmentTimeline>
        </SegmentTemplate>
      </Representation>
    </AdaptationSet>
  </Period>
</MPD>
"""

_BTS = '{"reps": [{"mimeType": "audio/mp4", "urls": ["https://a/x1", "https://a/x2"]}]}'


def test_segment_list_manifest():
    urls = parse_manifest(_MPD_SEGMENT_LIST, "https://cdn.tidal.com/path/")
    assert urls == [
        "https://cdn.tidal.com/path/init.mp4",
        "https://cdn.tidal.com/path/seg-1.m4s",
        "https://cdn.tidal.com/path/seg-2.m4s",
    ]


def test_segment_template_with_timeline():
    urls = parse_manifest(_MPD_TIMELINE, "https://cdn/")
    # S r=2 -> 3 segments, startNumber=0
    assert urls == ["https://cdn/init.mp4", "https://cdn/s0.m4s", "https://cdn/s1.m4s", "https://cdn/s2.m4s"]


def test_segment_template_duration_count():
    urls = parse_manifest(_MPD_SEGMENT_TEMPLATE, "https://cdn/")
    # period 12s / segment 176400/44100=4s -> 3 media segments + init
    assert urls[0] == "https://cdn/audio/init.mp4"
    assert urls[1:] == [
        "https://cdn/audio/00001.m4s",
        "https://cdn/audio/00002.m4s",
        "https://cdn/audio/00003.m4s",
    ]


def test_bts_json_manifest():
    urls = parse_manifest(_BTS, "")
    assert urls == ["https://a/x1", "https://a/x2"]


def test_decode_data_uri():
    payload = base64.b64encode(b"<MPD/>").decode()
    mime, text = decode_data_uri(f"data:application/dash+xml;base64,{payload}")
    assert mime == "application/dash+xml"
    assert text == "<MPD/>"
