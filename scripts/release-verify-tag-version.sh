#!/usr/bin/env bash
# Validate GITHUB_REF_NAME is vN.N.N and matches [project] version in pyproject.toml.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

tag="${GITHUB_REF_NAME:-}"
if [[ ! "${tag}" =~ ^v[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
    echo "error: release tag must match vN.N.N, got: ${tag:-<empty>}" >&2
    exit 1
fi

tag_ver="${tag#v}"
pyproject_ver="$(python3 - <<'EOF'
import tomllib, pathlib, sys
data = tomllib.loads(pathlib.Path("pyproject.toml").read_text(encoding="utf-8"))
ver = data.get("project", {}).get("version") or data.get("tool", {}).get("poetry", {}).get("version")
if not ver:
    print("error: could not find version in pyproject.toml", file=sys.stderr)
    sys.exit(1)
print(ver)
EOF
)"

if [[ "${tag_ver}" != "${pyproject_ver}" ]]; then
    echo "error: tag version ${tag_ver} != pyproject.toml version ${pyproject_ver}" >&2
    exit 1
fi

if [[ -n "${GITHUB_ENV:-}" ]]; then
    echo "VERSION=${tag_ver}" >> "${GITHUB_ENV}"
fi
echo "Extracted version: ${tag_ver}"
