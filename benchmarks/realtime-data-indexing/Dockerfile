# The serviette image for the benchmark, built from the repo checkout with
# the released Pathway from PyPI. The build context is the repo root — see
# docker-compose.yml; `docker compose build` (or the first `up`) is all the
# setup there is.
FROM python:3.12-slim

COPY pyproject.toml README.md LICENSE /src/
COPY serviette /src/serviette
RUN pip install --no-cache-dir "/src[qdrant,local]"

WORKDIR /bench
