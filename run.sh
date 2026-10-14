#!/bin/sh
set -eu
cd "$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
if [ ! -x .venv/bin/visualizer ]; then
  echo 'Environment missing. See README.md for installation instructions.' >&2
  exit 1
fi
exec .venv/bin/visualizer serve --config visualizer.toml "$@"
