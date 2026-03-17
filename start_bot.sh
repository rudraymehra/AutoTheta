#!/bin/bash
# AutoTheta — daily startup script (called by cron at 9:10 AM IST)

cd /Users/rudraym/Trader
source .venv/bin/activate

# Download fresh instrument master (updates daily at 8:30 AM)
python3 -c "
import requests, json
from pathlib import Path
print('Downloading instrument master...')
r = requests.get('https://margincalculator.angelbroking.com/OpenAPI_File/files/OpenAPIScripMaster.json', timeout=120)
Path('data').mkdir(exist_ok=True)
with open('data/instruments.json', 'w') as f:
    json.dump(r.json(), f)
print(f'Done: {len(r.json())} instruments')
" 2>&1

# Run paper trading bot (runs until market close or Ctrl+C)
# Output goes to today's log file
TODAY=$(date +%Y-%m-%d)
mkdir -p "logs/$TODAY"
python3 paper_live.py >> "logs/$TODAY/console.log" 2>&1
