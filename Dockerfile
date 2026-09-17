# Waku dashboard in a container. Same program as `make dashboard`, with two
# differences: it binds 0.0.0.0 so Docker can publish 7777, and runtime state
# lives at WAKU_HOME=/app/.waku (bind-mounted to ./.waku by compose.yaml).
#
#   docker compose up --build
#   open http://localhost:7777
FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

# Hatch needs the license files at install time (PEP 639). Skills live at the
# repo root — an editable install does not copy them into the package, so they
# have to be in the image or procedural memory starts empty.
COPY pyproject.toml README.md LICENSE LICENSE-BRAND ./
COPY waku ./waku
COPY skills ./skills

RUN uv venv && uv pip install --no-cache -e .

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    WAKU_DASHBOARD_HOST=0.0.0.0 \
    WAKU_HOME=/app/.waku

EXPOSE 7777

CMD ["waku", "dashboard"]
