#!/bin/bash
# AutoTheta — daily startup script (called by launchd at 9:10 AM IST)

export PATH="/usr/local/bin:/usr/bin:/bin:$PATH"
cd /Users/rudraym/Trader

VENV_PYTHON="/Users/rudraym/Trader/.venv/bin/python3"
TODAY=$(date +%Y-%m-%d)
mkdir -p "logs/$TODAY"

# Download fresh instrument master (updates daily at 8:30 AM)
$VENV_PYTHON -c "
import requests, json
from pathlib import Path
print('Downloading instrument master...')
r = requests.get('https://margincalculator.angelbroking.com/OpenAPI_File/files/OpenAPIScripMaster.json', timeout=120)
Path('data').mkdir(exist_ok=True)
with open('data/instruments.json', 'w') as f:
    json.dump(r.json(), f)
print(f'Done: {len(r.json())} instruments')
" >> "logs/$TODAY/console.log" 2>&1

# Run paper trading bot (runs until 3:30 PM auto-stop)
exec $VENV_PYTHON paper_live.py >> "logs/$TODAY/console.log" 2>&1
