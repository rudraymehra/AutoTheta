#!/bin/bash
# AutoTheta — daily startup script (called by launchd at 9:10 AM IST)
# Retries up to 5 times if network/auth fails

export PATH="/usr/local/bin:/usr/bin:/bin:$PATH"
cd /Users/rudraym/Trader

VENV_PYTHON="/Users/rudraym/Trader/.venv/bin/python3"
TODAY=$(date +%Y-%m-%d)
mkdir -p "logs/$TODAY"

LOG="logs/$TODAY/console.log"

# Download fresh instrument master (with retry)
for i in 1 2 3 4 5; do
    $VENV_PYTHON -c "
import requests, json
from pathlib import Path
print('Downloading instrument master...')
r = requests.get('https://margincalculator.angelbroking.com/OpenAPI_File/files/OpenAPIScripMaster.json', timeout=120)
r.raise_for_status()
Path('data').mkdir(exist_ok=True)
with open('data/instruments.json', 'w') as f:
    json.dump(r.json(), f)
print(f'Done: {len(r.json())} instruments')
" >> "$LOG" 2>&1 && break
    echo "[$(date '+%H:%M:%S')] Instrument download failed (attempt $i/5), retrying in 30s..." >> "$LOG"
    sleep 30
done

# Run paper trading bot with retry (if auth/network fails, wait and retry)
for i in 1 2 3 4 5; do
    echo "[$(date '+%H:%M:%S')] Starting paper_live.py (attempt $i/5)..." >> "$LOG"
    $VENV_PYTHON paper_live.py >> "$LOG" 2>&1
    EXIT_CODE=$?

    # If it exited cleanly (auto-stop at 3:30 PM), don't retry
    if [ $EXIT_CODE -eq 0 ]; then
        break
    fi

    # If market is already closed, don't retry
    HOUR=$(date +%H)
    if [ "$HOUR" -ge 16 ]; then
        echo "[$(date '+%H:%M:%S')] Market closed, not retrying." >> "$LOG"
        break
    fi

    echo "[$(date '+%H:%M:%S')] Bot crashed (exit=$EXIT_CODE), retrying in 60s..." >> "$LOG"
    sleep 60
done
