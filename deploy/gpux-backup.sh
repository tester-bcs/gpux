#!/bin/bash
# gpux backup — pulls the central gallery store + usage.db from niceguy to
# ms-7c75's HDD. Runs from ms-7c75 user crontab (niceguy has no keys to push).
#   - store/  : incremental mirror (fast, always current)
#   - usage.db: dated copies, keep last N
#   - weekly  : dated tarball snapshot of the mirror for point-in-time restore
set -uo pipefail

REMOTE=niceguy
R_STORE=/home/avk/gpux/store/
R_DB=/home/avk/gpux/router/usage.db
DEST=/mnt/hdd/data/gpux-backups
KEEP=21

mkdir -p "$DEST/store" "$DEST/snapshots"
ts=$(date +%Y%m%d-%H%M%S)

rsync -a --delete --timeout=60 "$REMOTE:$R_STORE" "$DEST/store/" \
  && echo "$ts store mirror ok" || echo "$ts store mirror FAILED"

scp -q -o ConnectTimeout=20 "$REMOTE:$R_DB" "$DEST/usage-$ts.db" \
  && echo "$ts usage.db ok" || echo "$ts usage.db FAILED"
ls -1t "$DEST"/usage-*.db 2>/dev/null | tail -n +$((KEEP+1)) | xargs -r rm -f

# weekly point-in-time snapshot (Mondays)
if [ "$(date +%u)" = "1" ]; then
  tar czf "$DEST/snapshots/store-$ts.tgz" -C "$DEST" store \
    && echo "$ts weekly snapshot ok"
  ls -1t "$DEST"/snapshots/store-*.tgz 2>/dev/null | tail -n +9 | xargs -r rm -f
fi
