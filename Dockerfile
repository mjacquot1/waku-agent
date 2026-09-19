# Waku dashboard in a container. Same program as `make dashboard`, with two
# differences: it binds 0.0.0.0 so Docker can publish 7777, and runtime state
# lives at WAKU_HOME=/app/.waku (bind-mounted to ./.waku by compose.yaml).
#
#   docker compose up --build
#   open http://localhost:7777
#
# save_html(render=true) needs Playwright + Chromium. The Python extra and the
# browser OS libraries are installed here; docker-entrypoint.sh re-runs
# `uv pip install -e '.[browser]'` and `playwright install chromium` on every
# start so an older image still heals itself.
FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

# Hatch needs the license files at install time (PEP 639). Skills live at the
# repo root — an editable install does not copy them into the package, so they
# have to be in the image or procedural memory starts empty.
COPY pyproject.toml README.md LICENSE LICENSE-BRAND ./
COPY waku ./waku
COPY skills ./skills
COPY docker-entrypoint.sh /docker-entrypoint.sh

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    WAKU_DASHBOARD_HOST=0.0.0.0 \
    WAKU_HOME=/app/.waku \
    PLAYWRIGHT_BROWSERS_PATH=/ms-playwright

RUN chmod +x /docker-entrypoint.sh \
    && uv venv \
    && uv pip install --no-cache -e '.[browser]' \
    && playwright install --with-deps chromium

EXPOSE 7777

ENTRYPOINT ["/docker-entrypoint.sh"]
CMD ["waku", "dashboard"]
