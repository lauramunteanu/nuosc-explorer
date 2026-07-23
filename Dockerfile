# Headless image: regenerates figures without a display (CI / batch).
# The interactive GUI needs a desktop session and is not served from here.
FROM python:3.12-slim

ENV MPLBACKEND=Agg \
    PIP_NO_CACHE_DIR=1 \
    NUOSC_FIGDIR=/out

RUN apt-get update && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY pyproject.toml README.md ./
COPY nuosc_explorer ./nuosc_explorer

RUN pip install git+https://github.com/pgranger23/jaxnu-osc.git \
    && pip install .

VOLUME /out
ENTRYPOINT ["nuosc-explorer"]
CMD ["--snapshot", "--outdir", "/out"]
