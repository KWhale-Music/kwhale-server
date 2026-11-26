FROM python:3.12-slim

# bun = JS runtime for yt-dlp; fpcalc = AcoustID fingerprinting
# BUN_ALL_PROXY — опциональный SOCKS/HTTP-прокси только для скачивания bun
# (из RU GitHub отдаёт ~16 КБ/с; пустое значение = прямое соединение).
ARG BUN_ALL_PROXY=
RUN apt-get update && apt-get install -y --no-install-recommends curl ca-certificates unzip zip libchromaprint-tools ffmpeg \
    && ALL_PROXY="${BUN_ALL_PROXY}" curl -fsSL https://bun.sh/install | ALL_PROXY="${BUN_ALL_PROXY}" bash -s "bun-v1.2.23" \
    && ln -s /root/.bun/bin/bun /usr/local/bin/bun \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

WORKDIR /srv

# Dependencies first, code after. With the old order (`COPY app` before pip)
# every code change invalidated the dependency layer and pip re-downloaded the
# whole world — 30-45 min per build on this link (~18 kB/s). The stub package
# satisfies pip's project build in the cached layer; the real code lands in the
# cheap layer below and is reinstalled with --no-deps (no network) so the
# installed `app` package in site-packages — which uvicorn imports — is real.
#
# PIP_PROXY — HTTP(S) proxy for dependency downloads. Fastly/PyPI is badly
# throttled from this network (measured: 0.2-0.4 MB/s direct vs 9 MB/s to
# Cloudflare); the local sing-box mixed inbound on :2080 is noticeably faster
# for pythonhosted. Build with:
#   docker compose build --build-arg PIP_PROXY=http://172.17.0.1:2080 api
ARG PIP_PROXY=
COPY pyproject.toml ./
RUN mkdir -p app && touch app/__init__.py \
    && pip install ${PIP_PROXY:+--proxy "$PIP_PROXY"} --no-cache-dir ".[dev,indexer]" setuptools wheel
COPY app ./app
RUN pip install --no-cache-dir --no-build-isolation --no-deps .

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]