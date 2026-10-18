#!/usr/bin/env bash
# Create .venv with a supported Python and install requirements.txt (macOS / Linux).
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="${REPO}/.venv"

PY=""
for v in 3.13 3.12 3.11; do
    if command -v "python${v}" >/dev/null 2>&1; then PY="python${v}"; break; fi
done
if [ -z "${PY}" ]; then
    echo "Python 3.11, 3.12 or 3.13 is required (3.14 has no numpy 2.2.6 wheel)."
    exit 1
fi

rm -rf "${VENV}"
"${PY}" -m venv "${VENV}"
source "${VENV}/bin/activate"
python -m pip install --upgrade pip --quiet
python -m pip install -r "${REPO}/requirements.txt"
echo "Done. Activate with: source ${VENV}/bin/activate"
