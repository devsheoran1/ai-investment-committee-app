#!/bin/bash
# Stage 8: copy the deliverables that are NOT in git (finance.db is gitignored) to backups/<YYYY-MM-DD_HHMM>/.
# Called automatically after every --execute and --snapshot; safe to run by hand.
cd "$(dirname "$0")" || exit 1
dest="backups/$(TZ=Europe/Madrid date +%Y-%m-%d_%H%M)"
mkdir -p "$dest"
for f in finance.db trades.csv prices.csv; do [ -f "$f" ] && cp "$f" "$dest/"; done
[ -d reports ] && cp -R reports "$dest/"
echo "backup -> $dest"
