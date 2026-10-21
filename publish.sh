#!/usr/bin/env bash
# Push live fund outputs so the Streamlit Cloud viewer refreshes. Called after live --execute / --snapshot.
set -euo pipefail
cd "$(dirname "$0")"
git add finance.db trades.csv reports/ 2>/dev/null || true
if git diff --cached --quiet -- finance.db trades.csv reports/; then echo "publish: nothing new"; exit 0; fi
git commit -q -m "Publish fund state $(date '+%Y-%m-%d %H:%M')" -- finance.db trades.csv reports/
git push -q && echo "publish: pushed"
