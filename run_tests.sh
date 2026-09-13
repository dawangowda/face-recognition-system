#!/usr/bin/env bash
# Logic + API tests. Neither needs model weights - they use synthetic embeddings
# and a stub engine, so they run in seconds on any machine.
set -e
export PYTHONPATH="$(dirname "$0")"
echo "=== matching / threshold-sweep logic ==="
python3 tests/test_matching.py
echo
echo "=== API endpoints ==="
python3 tests/test_api.py
