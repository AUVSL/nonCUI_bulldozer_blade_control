# Docker Official Image maintained by Canonical. Review before pulling/building.
# https://hub.docker.com/_/ubuntu
FROM docker.io/library/ubuntu:24.04@sha256:69cecf4bbf72d2d44a9eef1b71fb98c7fb973d78af11399deccef19beb008ad9

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MPLBACKEND=Agg \
    MPLCONFIGDIR=/tmp/matplotlib \
    PATH="/opt/venv/bin:$PATH"

RUN apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends python3 python3-venv ca-certificates git \
    && rm -rf /var/lib/apt/lists/* \
    && python3 -m venv /opt/venv

WORKDIR /app
COPY requirements.txt ./
RUN python -m pip install --no-cache-dir --only-binary=:all: \
    --index-url https://pypi.org/simple -r requirements.txt \
    && python -m pip check

RUN groupadd --gid 10001 simulator \
    && useradd --uid 10001 --gid simulator --create-home --shell /bin/bash simulator \
    && mkdir -p /app/figures \
    && chown simulator:simulator /app/figures
COPY *.py ./
COPY LICENSE ./

USER simulator
CMD ["python", "util.py"]
