#!/usr/bin/env bash

set -euo pipefail

ROOT="/nobackup/nd276333/Workspace/Alternance/cdsaxs/src/roughness"

# Create directory tree
mkdir -p \
    "$ROOT/data/profiles" \
    "$ROOT/data/cdsaxs/patterns" \
    "$ROOT/data/cdsem/inputs" \
    "$ROOT/data/cdsem/images" \
    "$ROOT/src/profile" \
    "$ROOT/src/cdsaxs" \
    "$ROOT/src/cdsem" \
    "$ROOT/src/shared" \
    "$ROOT/scripts" \
    "$ROOT/figures" \
    "$ROOT/tests"

# Create top-level files
touch \
    "$ROOT/README.md" \
    "$ROOT/config.yaml"

# Create source files
touch \
    "$ROOT/src/profile/generate.py" \
    "$ROOT/src/cdsaxs/simulate.py" \
    "$ROOT/src/cdsaxs/extract.py" \
    "$ROOT/src/cdsem/prepare.py" \
    "$ROOT/src/cdsem/run_nebula.py" \
    "$ROOT/src/cdsem/extract.py" \
    "$ROOT/src/shared/fitting.py" \
    "$ROOT/src/shared/compare.py"

# Create script files
touch \
    "$ROOT/scripts/run_all.py" \
    "$ROOT/scripts/run_nebula_pipeline.py"

# Create test files
touch \
    "$ROOT/tests/test_profile.py" \
    "$ROOT/tests/test_cdsaxs.py" \
    "$ROOT/tests/test_cdsem.py"

# Optional package initialization files
find "$ROOT/src" -type d -exec touch {}/__init__.py \;

echo "Project structure created at:"
echo "  $ROOT"