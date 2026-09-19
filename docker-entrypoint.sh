#!/bin/sh
# Ensure the [browser] extra and Chromium are present, then start the CMD.
# Cheap when the image already has them; heals a container started from an
# older build that did not. OS libraries for Chromium are installed at image
# build (playwright install --with-deps) — this step only fetches the browser
# binary if it is missing.
set -e
uv pip install --no-cache -e '.[browser]'
playwright install chromium
exec "$@"
